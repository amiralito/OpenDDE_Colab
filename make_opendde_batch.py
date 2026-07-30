#!/usr/bin/env python3
"""Generate OpenDDE batch input JSON(s).

An OpenDDE input file is a top-level LIST of job dicts; `opendde msa` / `opendde pred`
iterate over every entry. Entities use `count` (and optionally explicit `id` lists):
proteinChain / dnaSequence / rnaSequence / ligand / ion. The same schema is used by
Protenix, so these files work with either tool.

Modes
-----
  monomer    one job per sequence                      --fasta seqs.fasta
  homomer    each sequence as an N-mer                 --fasta nlrs.fasta --copies 6
  combos     N-way cartesian product across FASTAs     --fastas a.fasta b.fasta [c.fasta ...]
  all_pairs  pairwise, within one FASTA or across two  --fasta_a x.fasta [--fasta_b y.fasta]
  pairs      explicit pairs from a TSV/CSV (idA,idB)   --pairs p.tsv --fasta_a x.fasta [--fasta_b y.fasta]

Examples
--------
# combinatorial effector x NLR screen, one bundle to feed the batch notebook
python make_opendde_batch.py combos --fastas effectors.fasta nlrs.fasta -o opendde_inputs

# hexameric resistosome with ATP + Mg, three seeds
python make_opendde_batch.py homomer --fasta nlrs.fasta --copies 6 --ligand CCD_ATP --ion MG \
       --seeds 101,102,103 -o opendde_inputs

# ligand panel: one job per ligand, plus an apo control, split into per-ligand folders
python make_opendde_batch.py monomer --fasta targets.fasta --ligand_file cofactors.txt \
       --ligand-mode each --include-apo --split-by-ligand -o opendde_inputs

Ligands (--ligand / --ligand_file): one or more, added to EVERY job.
  forms   "CCD_ATP" / "ccd:ATP" / "ATP" -> CCD code;  "smiles:CC(=O)O" -> SMILES
          "file:/path/lig.sdf" / "FILE_/path/lig.sdf" -> ligand file (OpenDDE-specific)
          "CCD_NAG_BMA_BGC" -> underscore-joined CCD codes, passed through as-is
  counts  per-ligand via "@N", e.g. CCD_ATP@6      (otherwise --ligand_copies)
  CLI     --ligand CCD_ATP CCD_ADP MG          (space- or comma-separated)
  file    --ligand_file cofactors.txt          (one spec per line, "# ..." comments,
                                                optional count as "CCD_ATP@6" or "CCD_ATP 6")
Ions (--ion): one or more bare CCD codes, e.g. --ion MG ZN  (also MG@2).

Ligand panel layout (--ligand-mode):
  all   (default) every ligand co-present in ONE job     -> N inputs = N jobs
  each            one job per ligand (all-vs-all panel)  -> N inputs x M ligands jobs
                  add --include-apo for a ligand-free control per input
  Ions are constant background: they are added to every variant in both modes.

Notes
-----
  MSAs        run `opendde msa` on the output once; it writes <file>-update-msa.json
              next to the input and unique sequences are searched only once.
  templates   OpenDDE reads templates from `templatesPath` in the JSON, not from a CLI
              flag; this generator does not emit template blocks.
  chain ids   omitted by default (OpenDDE assigns them from `count`). Pass --ids to
              write explicit id lists (A, B, ... Z, AA, AB, ...).
"""
import argparse, csv, difflib, hashlib, json, os, re, sys
from itertools import product


# --------------------------------------------------------------------------- #
# FASTA
# --------------------------------------------------------------------------- #
def read_fasta(path):
    seqs, name, buf = {}, None, []
    try:
        with open(path, encoding='utf-8-sig') as f:      # utf-8-sig drops a leading BOM
            for line in f:
                line = line.rstrip()
                if not line:
                    continue
                if line.startswith('>'):
                    if name is not None:
                        seqs[name] = ''.join(buf)
                    name = line[1:].split()[0]      # first whitespace-delimited token
                    buf = []
                else:
                    buf.append(line.strip())
    except FileNotFoundError:
        sys.exit(f'Error: could not find FASTA file: {path}')
    if name is not None:
        seqs[name] = ''.join(buf)
    if not seqs:
        sys.exit(f'No sequences parsed from {path}')
    return seqs


# --------------------------------------------------------------------------- #
# naming / sequence helpers
# --------------------------------------------------------------------------- #
_UNSAFE = re.compile(r'[^A-Za-z0-9_]+')

def sanitize(s):
    s = s.replace('.', '_').replace('^', '_').replace('-', '_')
    return _UNSAFE.sub('_', s).strip('_') or 'job'

def clean_seq(s):
    return re.sub(r'\s+', '', s).upper()

def guess_type(seq):
    s = clean_seq(seq)
    if s and all(c in 'ACGT' for c in s):
        return 'dna'
    if s and all(c in 'ACGU' for c in s):
        return 'rna'
    return 'protein'

def parse_ligand(spec, default_count=1, label=None):
    if '|' in spec:                      # 'smiles:C#N|HCN' -> label HCN ('|' is not valid in SMILES)
        spec, _, lbl = spec.rpartition('|')
        label = label or lbl.strip() or None
    """Parse one ligand spec into {'ccd'|'smiles': ..., 'count': N}.

    Accepted:  CCD_ATP | ccd:ATP | ATP | smiles:CC(=O)O
    Optional per-ligand count via a trailing '@N', e.g. CCD_ATP@6.
    ('@' is also SMILES chirality, so we only split when the tail is all digits.)
    """
    if not spec:
        return None
    s = spec.strip()
    count = default_count
    base, at, tail = s.rpartition('@')
    if at and base and tail.isdigit():          # guard protects [C@H] style SMILES
        s, count = base.strip(), int(tail)
    low = s.lower()
    if low.startswith('file:'):
        out = {'file': s[5:]}
    elif s.upper().startswith('FILE_'):
        out = {'file': s[5:]}
    elif low.startswith('smiles:'):
        out = {'smiles': s[7:]}
    elif low.startswith('ccd:'):
        out = {'ccd': s[4:].upper()}
    elif s.upper().startswith('CCD_'):
        out = {'ccd': s[4:].upper()}
    else:
        out = {'ccd': s.upper()}                # bare token -> assume CCD code
    out['count'] = int(count)
    if label:
        out['label'] = label
    return out


def _split_specs(tokens):
    """Flatten CLI tokens, allowing comma-separated lists.
    SMILES are never comma-split (commas aren't valid SMILES, but be safe)."""
    out = []
    for t in tokens or []:
        t = t.strip()
        if not t:
            continue
        if t.lower().startswith('smiles:'):
            out.append(t)
        else:
            out.extend(x for x in (p.strip() for p in t.split(',')) if x)
    return out


def read_spec_file(path):
    """One ligand per line:  SPEC [count] [label]

        CCD_ATP                     -> default count, name tag 'ATP'
        CCD_ATP@6                   -> 6 copies
        CCD_ATP 6                   -> 6 copies
        smiles:OC(=O)... 1 OG7      -> 1 copy, name tag 'OG7'
        smiles:OC(=O)... OG7        -> default count, name tag 'OG7'

    A label makes job names readable instead of a SMILES hash. Blank lines and
    whole-line '#' comments are skipped; an inline '#' is NOT a comment because
    '#' is a triple bond in SMILES (e.g. C#N). SMILES contain no spaces, so
    whitespace splitting is safe."""
    out = []
    try:
        with open(path, encoding='utf-8-sig') as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                spec, count, label = parts[0], None, None
                rest = parts[1:]
                if rest and rest[0].isdigit():
                    count = rest.pop(0)
                if rest:
                    label = rest[0]
                if count:
                    spec = f'{spec}@{count}'
                for sp in _split_specs([spec]):
                    out.append((sp, label))
    except FileNotFoundError:
        sys.exit(f'Error: could not find ligand file: {path}')
    if not out:
        sys.exit(f'No ligand specs parsed from {path}')
    return out


def collect_ligands(a):
    """CLI --ligand tokens plus --ligand_file lines -> list of parsed ligand dicts."""
    pairs = [(s, None) for s in _split_specs(a.ligand)]
    if a.ligand_file:
        pairs += read_spec_file(a.ligand_file)
    ligs = [parse_ligand(sp, a.ligand_copies, lbl) for sp, lbl in pairs]
    for p in ligs:
        if 'smiles' in p and p.get('label'):
            _SMILES_LABELS[p['smiles']] = p['label']
    return ligs


def collect_ions(a):
    """CLI --ion tokens (bare CCD codes, optional @N) -> list of ion dicts."""
    out = []
    for s in _split_specs(a.ion):
        code, at, tail = s.rpartition('@')
        if at and code and tail.isdigit():
            out.append({'code': code.strip().upper(), 'count': int(tail)})
        else:
            out.append({'code': s.upper(), 'count': a.ion_copies})
    return out


_SMILES_LABELS = {}          # smiles -> label, for the manifest only (never emitted to JSON)


def lig_tag(p):
    """Short, filesystem-safe tag for job names."""
    if p.get('label'):
        return sanitize(p['label'])
    if 'ccd' in p:
        return sanitize(p['ccd'])
    if 'file' in p:
        return sanitize(os.path.splitext(os.path.basename(p['file']))[0])
    return 'smi' + hashlib.md5(p['smiles'].encode()).hexdigest()[:6]


# --------------------------------------------------------------------------- #
# neutral entity builders  (dialect-agnostic intermediate representation)
# --------------------------------------------------------------------------- #
def e_protein(seq, count):
    return {'kind': 'protein', 'seq': clean_seq(seq), 'count': int(count)}

def e_nucleic(seq, count, kind):          # kind in {'dna','rna'}
    return {'kind': kind, 'seq': clean_seq(seq), 'count': int(count)}

def e_ligand(parsed):
    body = {k: v for k, v in parsed.items() if k != 'count'}
    return {'kind': 'ligand', 'count': int(parsed['count']), **body}

def e_ion(code, count):
    return {'kind': 'ion', 'code': code, 'count': int(count)}

def make_chain(seq, count, a):
    t = a.type if a.type != 'auto' else guess_type(seq)
    if t == 'protein':
        return e_protein(seq, count)
    return e_nucleic(seq, count, t)

def extras_tags(ligs, ions, a):
    """Short name tags for a set of ligands + ions."""
    tags = []
    for p in ligs:
        t = lig_tag(p)
        tags.append(t if p['count'] <= 1 else f"{t}_{p['count']}")
    for io in ions:
        t = sanitize(io['code'])
        tags.append(t if io['count'] <= 1 else f"{t}_{io['count']}")
    if len(tags) > a.name_max_ligands:          # keep filenames sane
        return [f'{len(tags)}lig']
    return tags


def attach_extras(base, a):
    """Turn (core, chains) into (name, entities), applying the ligand panel.

    ligand_mode 'all'  : every ligand co-present in one job  (N proteins -> N jobs)
    ligand_mode 'each' : one job per ligand, screened separately
                         (N proteins x M ligands -> N*M jobs, + apo with --include-apo)
    Ions are treated as constant background and are added to every variant.
    """
    ions = a._ions
    ligs = a._ligands

    if a.ligand_mode == 'each' and ligs:
        variants = [[p] for p in ligs]
        if a.include_apo:
            variants.insert(0, [])              # ligand-free control
    else:
        variants = [ligs]                       # single variant: all (or none)

    out = []
    for core, chains in base:
        for v in variants:
            ents = list(chains)
            ents += [e_ligand(p) for p in v]
            ents += [e_ion(io['code'], io['count']) for io in ions]
            tags = extras_tags(v, ions, a)
            if a.ligand_mode == 'each' and ligs and not v:
                tags = ['apo'] + tags           # label the control explicitly
            # tag identifying just the screened ligand (for grouping / prefixing)
            ligtag = (lig_tag(v[0]) if v else ('apo' if ligs else ''))
            name = '_'.join(tags + [core]) if a.ligand_first else '_'.join([core] + tags)
            out.append((name, ents, ligtag))
    return out


# --------------------------------------------------------------------------- #
# job assembly
# --------------------------------------------------------------------------- #
def compose_name(core, a, seed=None):
    parts = [core]                              # ligand/ion tags already folded in
    if seed is not None:
        parts.append(f'seed{seed}')
    if a.suffix:
        parts.append(sanitize(a.suffix))
    return '_'.join(parts)


def _clean_id(s):
    # strip whitespace, any stray BOM, and surrounding quotes
    return s.strip().lstrip('\ufeff').strip().strip('"').strip("'")

def read_pairs_table(path):
    """Return a list of (idA, idB) tuples. BOM/CRLF/quote-safe; no header stripping
    here (that needs the FASTA ids and is done by the caller)."""
    with open(path, newline='', encoding='utf-8-sig') as f:   # utf-8-sig drops a BOM; newline='' lets csv handle CRLF
        sniff = f.read(4096); f.seek(0)
        delim = '\t' if sniff.count('\t') >= sniff.count(',') else ','
        rows = []
        for r in csv.reader(f, delimiter=delim):
            if not r or not r[0].strip() or _clean_id(r[0]).startswith('#'):
                continue
            if len(r) < 2:
                sys.exit(f'pairs row needs two columns (got {r!r} with delimiter {delim!r})')
            rows.append((_clean_id(r[0]), _clean_id(r[1])))
    if not rows:
        sys.exit(f'No pairs parsed from {path}')
    return rows


# common header labels seen in pair tables (lower-case); used only for auto-detection
_HEADER_TOKENS = {
    'ida', 'id_a', 'a', 'idb', 'id_b', 'b', 'id1', 'id2', 'col1', 'col2',
    'effector', 'sensor', 'nlr', 'receptor', 'bait', 'prey', 'ligand',
    'query', 'target', 'fasta_a', 'fasta_b', 'name1', 'name2',
    'protein_a', 'protein_b', 'chain_a', 'chain_b',
}

def _pair_err(pid, label, path, D):
    near = difflib.get_close_matches(pid, list(D), n=3, cutoff=0.6)
    hint = f'\n  - closest ids in {label}: {", ".join(near)}' if near else ''
    sample = ', '.join(list(D)[:2])
    return (f'pair id "{pid}" not found in {label} ({path}).{hint}\n'
            f'  - if your table has a header row, pass --header yes (auto-detect may have missed it)\n'
            f'  - example {label} ids: {sample} ...')


def base_jobs(a):
    out = []
    if a.mode == 'monomer':
        for nm, sq in read_fasta(a.fasta).items():
            out.append((sanitize(nm), [make_chain(sq, 1, a)]))

    elif a.mode == 'homomer':
        for nm, sq in read_fasta(a.fasta).items():
            out.append((f'{sanitize(nm)}_{a.copies}', [make_chain(sq, a.copies, a)]))

    elif a.mode == 'combos':
        fastas = [read_fasta(f) for f in a.fastas]
        for combo in product(*[list(d.items()) for d in fastas]):
            ids = [c[0] for c in combo]
            chains = [make_chain(sq, a.copies, a) for (_, sq) in combo]
            out.append(('_'.join(sanitize(i) for i in ids), chains))

    elif a.mode == 'all_pairs':
        A = read_fasta(a.fasta_a)
        if a.fasta_b:
            pairs = list(product(A.items(), read_fasta(a.fasta_b).items()))
        else:
            items = list(A.items()); pairs = []
            for i in range(len(items)):
                lo = i if a.include_self else i + 1
                for j in range(lo, len(items)):
                    pairs.append((items[i], items[j]))
        for (na, sa), (nb, sb) in pairs:
            ents = [make_chain(sa, a.copies_a, a), make_chain(sb, a.copies_b, a)]
            out.append((f'{sanitize(na)}_{sanitize(nb)}', ents))

    elif a.mode == 'pairs':
        A = read_fasta(a.fasta_a)
        B = read_fasta(a.fasta_b) if a.fasta_b else A
        b_label = 'fasta_b' if a.fasta_b else 'fasta_a'
        b_path = a.fasta_b or a.fasta_a
        rows = read_pairs_table(a.pairs)

        # header handling: yes = always skip row 0, no = never, auto = detect
        drop = False
        if a.header == 'yes':
            drop = True
        elif a.header == 'auto':
            ia, ib = rows[0]
            if not (ia in A and ib in B):            # a real data row would resolve in both FASTAs
                if ia.lower() in _HEADER_TOKENS or ib.lower() in _HEADER_TOKENS:
                    drop = True                      # recognised label -> header
                elif len(rows) > 1 and rows[1][0] in A and rows[1][1] in B:
                    drop = True                      # row 0 unresolved but row 1 is clean data -> header
        if drop:
            print(f'note: skipping first row as a header: {rows[0]}', file=sys.stderr)
            rows = rows[1:]

        for ida, idb in rows:
            if ida not in A:
                sys.exit(_pair_err(ida, 'fasta_a', a.fasta_a, A))
            if idb not in B:
                sys.exit(_pair_err(idb, b_label, b_path, B))
            ents = [make_chain(A[ida], a.copies_a, a), make_chain(B[idb], a.copies_b, a)]
            out.append((f'{sanitize(ida)}_{sanitize(idb)}', ents))

    if not out:
        sys.exit('No jobs generated - check your inputs.')
    return out


def finalize(base, a):
    sd = [int(x) for x in a.seeds.split(',') if x.strip()] if a.seeds else []
    jobs = []
    for core, ents, ligtag in base:
        if a.split_seeds and sd:
            for s in sd:
                jobs.append({'name': compose_name(core, a, seed=s), 'seeds': [s],
                             'entities': ents, 'ligtag': ligtag})
        else:
            jobs.append({'name': compose_name(core, a), 'seeds': sd,
                         'entities': ents, 'ligtag': ligtag})
    seen = {}
    for jb in jobs:
        b = jb['name']; n = seen.get(b, 0)
        if n:
            jb['name'] = f'{b}_{n + 1}'
        seen[b] = n + 1
    return jobs


# --------------------------------------------------------------------------- #
# renderers
# --------------------------------------------------------------------------- #
def _idx_to_id(i):                        # 0->A, 25->Z, 26->AA, 27->AB, ...
    s = ''
    i += 1
    while i > 0:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s

class _IdGen:
    def __init__(self):
        self.i = 0
    def take(self, n):
        ids = [_idx_to_id(self.i + k) for k in range(n)]
        self.i += n
        return ids


def render_entity(e, a, ids=None):
    """Render one neutral entity into OpenDDE's AlphaFold-Server-style schema."""
    k = e['kind']
    if k == 'protein':
        chain = {'sequence': e['seq'], 'count': e['count']}
        if ids: chain['id'] = ids
        return {'proteinChain': chain}
    if k in ('dna', 'rna'):
        body = {'sequence': e['seq'], 'count': e['count']}
        if ids: body['id'] = ids
        return {'dnaSequence' if k == 'dna' else 'rnaSequence': body}
    if k == 'ligand':
        if 'ccd' in e:
            val = e['ccd'] if e['ccd'].startswith('CCD_') else f"CCD_{e['ccd']}"
        elif 'file' in e:
            val = f"FILE_{e['file']}"
        else:
            val = e['smiles']
        body = {'ligand': val, 'count': e['count']}
        if ids: body['id'] = ids
        return {'ligand': body}
    if k == 'ion':
        body = {'ion': e['code'], 'count': e['count']}
        if ids: body['id'] = ids
        return {'ion': body}
    raise ValueError(k)


def to_opendde_job(job, a):
    g = _IdGen() if a.ids else None
    seqs = [render_entity(e, a, g.take(e['count']) if g else None) for e in job['entities']]
    out = {'name': job['name'], 'sequences': seqs}
    if job['seeds']:
        out = {'name': job['name'], 'modelSeeds': job['seeds'], 'sequences': seqs}
    return out


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def job_ligands(rendered):
    """Summarise ligand/ion content of a rendered job for the manifest."""
    out = []
    for ent in rendered['sequences']:
        k, v = next(iter(ent.items()))
        if k == 'ligand':
            val = str(v.get('ligand', ''))
            if val.startswith('CCD_'):
                out.append(val[4:])
            elif val.startswith('FILE_'):
                out.append(os.path.basename(val[5:]))
            else:
                out.append(_SMILES_LABELS.get(val, 'smiles'))
        elif k == 'ion':
            out.append(v['ion'])
    return ';'.join(out) if out else 'apo'


def write_outputs(jobs, a):
    os.makedirs(a.out_dir, exist_ok=True)
    manifest = []
    rendered = [to_opendde_job(job, a) for job in jobs]

    if a.per_job:
        subdirs = set()
        for job, rj in zip(jobs, rendered):
            d = a.out_dir
            if a.split_by_ligand and job.get('ligtag'):
                d = os.path.join(a.out_dir, job['ligtag'])
                if d not in subdirs:
                    os.makedirs(d, exist_ok=True); subdirs.add(d)
            fp = os.path.join(d, f"{rj['name']}.json")
            with open(fp, 'w') as f:
                json.dump([rj], f, indent=2)          # top level must be a LIST
            manifest.append((os.path.relpath(fp, a.out_dir), rj['name'],
                             len(rj['sequences']), job_ligands(rj)))
        extra = f' across {len(subdirs)} ligand subdirectorie(s)' if subdirs else ''
        print(f'wrote {len(rendered)} per-job file(s) to {a.out_dir}/{extra}')
    else:
        # optionally emit one bundle per ligand instead of one for everything
        if a.split_by_ligand:
            groups = {}
            for job, rj in zip(jobs, rendered):
                groups.setdefault(job.get('ligtag') or 'all', []).append(rj)
        else:
            groups = {None: rendered}
        for tag, grp in groups.items():
            chunks = [grp] if a.chunk <= 0 else [grp[i:i + a.chunk]
                                                for i in range(0, len(grp), a.chunk)]
            for ci, ch in enumerate(chunks):
                suffix = '' if len(chunks) == 1 else f'_{ci + 1:03d}'
                stem = f'{a.name}_{tag}' if tag else a.name
                fp = os.path.join(a.out_dir, f'{stem}{suffix}.json')
                with open(fp, 'w') as f:
                    json.dump(ch, f, indent=2)
                manifest += [(os.path.basename(fp), rj['name'], len(rj['sequences']),
                              job_ligands(rj)) for rj in ch]
                print(f'wrote {fp}  ({len(ch)} jobs)')

    with open(os.path.join(a.out_dir, f'{a.name}_manifest.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['json_file', 'job_name', 'n_entities', 'ligands'])
        w.writerows(manifest)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=['monomer', 'homomer', 'combos', 'all_pairs', 'pairs'])

    p.add_argument('--ids', action='store_true',
                   help='write explicit chain id lists (A, B, ... AA, AB) on every entity; '
                        'by default OpenDDE assigns ids from `count`')

    # inputs
    p.add_argument('--fasta')
    p.add_argument('--fastas', nargs='+', help='two or more FASTA files (combos mode)')
    p.add_argument('--fasta_a'); p.add_argument('--fasta_b'); p.add_argument('--pairs')
    p.add_argument('--header', choices=['auto', 'yes', 'no'], default='auto',
                   help='pairs table header row: auto-detect (default), yes = always skip row 1, no = never skip')

    # copy numbers
    p.add_argument('--copies', type=int, default=1, help='homomer copy number / copies per chain in combos')
    p.add_argument('--copies_a', type=int, default=1)
    p.add_argument('--copies_b', type=int, default=1)
    p.add_argument('--include_self', action='store_true',
                   help='all_pairs within one FASTA: also pair i with itself')

    # entity typing / templates
    p.add_argument('--type', choices=['auto', 'protein', 'dna', 'rna'], default='auto',
                   help='force entity type (default: auto-detect per sequence)')

    # global add-ons
    p.add_argument('--ligand', nargs='+', default=None,
                   help='one or more ligands added to EVERY job. Space- or comma-separated. '
                        'Forms: CCD_ATP | ccd:ATP | ATP | smiles:<STR>. '
                        'Per-ligand count with @N, e.g. CCD_ATP@6.')
    p.add_argument('--ligand_file', help='text file of ligand specs, one per line '
                                         '(optional count as "SPEC@N" or "SPEC <N>"; # comments allowed)')
    p.add_argument('--ligand_copies', type=int, default=1,
                   help='default copy number for ligands that do not specify @N (default: 1)')
    p.add_argument('--ion', nargs='+', default=None,
                   help='one or more ion CCD codes, e.g. MG ZN CA (also accepts MG@2)')
    p.add_argument('--ion_copies', type=int, default=1,
                   help='default copy number for ions that do not specify @N (default: 1)')
    p.add_argument('--ligand-mode', dest='ligand_mode', choices=['all', 'each'], default='all',
                   help="all = every ligand co-present in one job (default); "
                        "each = one job per ligand (all-vs-all panel: N inputs x M ligands)")
    p.add_argument('--ligand-first', dest='ligand_first', action='store_true',
                   help='put the ligand tag at the START of job names (readable when protein ids are long)')
    p.add_argument('--split-by-ligand', dest='split_by_ligand', action='store_true',
                   help='write each ligand into its own subdirectory of --out_dir')
    p.add_argument('--include-apo', dest='include_apo', action='store_true',
                   help='with --ligand-mode each, also emit a ligand-free (apo) control job per input')
    p.add_argument('--name-max-ligands', dest='name_max_ligands', type=int, default=3,
                   help='if more than N ligand+ion tags, collapse them to "<n>lig" in job names (default: 3)')

    # seeds / naming
    p.add_argument('--seeds', help='comma-separated model seeds embedded in each job. '
                                   'server: empty -> random; alphafold3: empty -> defaults to 1.')
    p.add_argument('--split-seeds', dest='split_seeds', action='store_true',
                   help='write one job per seed (legacy) instead of one job carrying all seeds')
    p.add_argument('--suffix', help='optional suffix appended to every job name')

    # output
    p.add_argument('-o', '--out_dir', default='af3_inputs')
    p.add_argument('--name', default='batch', help='base name for the bundled output file(s)')
    p.add_argument('--chunk', type=int, default=0, help='split the bundle into files of N jobs (keeps runs restartable)')
    p.add_argument('--per-job', dest='per_job', action='store_true',
                   help='write one <job_name>.json per job instead of a single bundle')

    a = p.parse_args()
    a._ligands = collect_ligands(a)
    a._ions = collect_ions(a)

    # minimal arg sanity per mode
    if a.mode in ('monomer', 'homomer') and not a.fasta:
        p.error(f'{a.mode} needs --fasta')
    if a.mode == 'combos' and (not a.fastas or len(a.fastas) < 2):
        p.error('combos needs --fastas with two or more FASTA files')
    if a.mode == 'all_pairs' and not a.fasta_a:
        p.error('all_pairs needs --fasta_a (and optionally --fasta_b)')
    if a.mode == 'pairs' and not (a.pairs and a.fasta_a):
        p.error('pairs needs --pairs and --fasta_a (and --fasta_b if cross-set)')

    # OpenDDE-specific notes
    if a.mode == 'combos' and a.copies != 1:
        print(f'note: --copies {a.copies} applies to EVERY chain in each combo.', file=sys.stderr)

    unlabelled = [p for p in a._ligands if 'smiles' in p and not p.get('label')]
    if unlabelled:
        print(f'note: {len(unlabelled)} SMILES ligand(s) have no label, so job names will use an opaque '
              f'hash (e.g. {lig_tag(unlabelled[0])}). Add a label as the last column in the ligand file '
              f'-- "smiles:<SMILES> [count] <label>" -- or append "|<label>" to the spec.', file=sys.stderr)

    jobs = finalize(attach_extras(base_jobs(a), a), a)
    write_outputs(jobs, a)

    print(f'\nTotal {len(jobs)} job(s) in {a.out_dir}/  (+ manifest).')
    print('Next, per file:')
    print('  export MMSEQS_SERVICE_HOST_URL=https://api.colabfold.com')
    print('  opendde msa  -i <file>.json -o ./msa            # writes <file>-update-msa.json')
    print('  opendde pred -i <file>-update-msa.json -o ./outputs -n opendde_v1 \\')
    print('               --use_msa true --need_atom_confidence true')
    print('Or point cell B1 of OpenDDE_batch.ipynb at <file>.json and run B2.')


if __name__ == '__main__':
    main()
