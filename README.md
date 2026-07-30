# OpenDDE on Colab: All-Atom Biomolecular Structure Prediction on Google Colab

Two Google Colab notebooks and a helper script for running [OpenDDE](https://github.com/aurekaresearch/OpenDDE)
(Aureka Research, AlphaFold3-class all-atom co-folding) end-to-end on Colab, using the
**ColabFold MSA server** so **no local sequence databases** are needed.

Everything runs from forms, predictions and the MSA cache persist to Google Drive, and the outputs
are wired straight into downstream interface scoring (**ipSAE**), PAE export for **ChimeraX**, and
an interactive **MolView** model browser.

* OpenDDE is installed into an **isolated `uv` virtualenv** — it pins its own PyTorch build, so it
  never disturbs Colab's Python. The package is not on PyPI; it installs from the GitHub repo.
* The checkpoint (~2.6 GB) and runtime common files (~0.6 GB) are fetched once
  (`hf_transfer` → `aria2c` → `curl`) and cached to Drive, so later sessions reuse them.
* **Please read [Job size & token limits](#job-size--token-limits) before planning a screen** —
  OpenDDE is considerably more memory-hungry per token than Protenix, and this is the main
  constraint on Colab.

## Contents

| File | What it is |
| --- | --- |
| `OpenDDE_single.ipynb` | One complex at a time, with full per-model inspection (3D view, pLDDT, interface contacts, PAE, ipSAE). |
| `OpenDDE_batch.ipynb`  | Many complexes in one pass, with aggregate tables (confidence, per-job seed means, ipSAE, interface contacts). |
| `make_opendde_batch.py` | Generates OpenDDE batch-input JSON(s) from FASTA files for common screen designs. |

## Open in Colab

<!-- PLACEHOLDER — update the repo path if you name the repository differently -->

[![Single](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amiralito/OpenDDE_Colab/blob/main/OpenDDE_single.ipynb) &nbsp;**single prediction**

[![Batch](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amiralito/OpenDDE_Colab/blob/main/OpenDDE_batch.ipynb) &nbsp;**batch predictions**

## Requirements

- A Colab GPU runtime. Memory scales steeply with token count (see
  [Job size & token limits](#job-size--token-limits)): a **T4** handles monomers up to ~300 tokens,
  an **A100 (40 GB)** reaches ~500. Larger assemblies are out of reach on a single Colab GPU.
- A Google Drive (optional but strongly recommended — the ~3.2 GB of model files, the MSA cache,
  and all outputs survive a disconnect).
- Nothing to install locally; the venv, OpenDDE, weights, and the MSA server are handled in the
  notebook.

The first run on a fresh Drive downloads the model files automatically (once).

---

## Job size & token limits

**There is no hard-coded token cap in OpenDDE** — the limit is GPU memory, and it binds early.
The numbers below are from the project's own single-GPU baseline
([`docs/foldcp_e2e_baseline.md`](https://github.com/aurekaresearch/OpenDDE/blob/main/docs/foldcp_e2e_baseline.md),
A800 80 GB, `opendde_v1`, **BF16**, sample=1, diffusion steps=2, cycle=1, PyTorch triangle kernels):

| Tokens (N) | Single-GPU peak |
| ---: | ---: |
| 101 | 3.5 GiB |
| 200 | 6.6 GiB |
| 299 | 14.2 GiB |
| 401 | 28.6 GiB |
| 600 | 43.2 GiB |
| 799 | 44.1 GiB |
| 1001 | 66.9 GiB |
| 1399 | 64.2 GiB |
| ≥ 2000 | **exceeds single-card capacity** (80 GB) |

Upstream omits single-GPU runs at N ≥ 2000 because they no longer fit on one 80 GB card; those
sizes require the **4-GPU Fold-CP** path, which needs `torchrun` across four GPUs and is therefore
**not available on Colab**.

**What that means per Colab GPU** (leaving headroom):

| Colab GPU | VRAM | Practical ceiling |
| --- | ---: | --- |
| T4 | 16 GB | ~300 tokens — monomers, small dimers |
| L4 | 22.5 GB | ~330 tokens |
| A100 | 40 GB | ~500 tokens |
| A100 | 80 GB (rare on Colab) | ~1400 tokens |

**How tokens are counted.** One token per polymer residue (protein / DNA / RNA), plus **one token
per heavy atom** for ligands and modified residues. So a 300-residue protein with ATP (31 heavy
atoms) is ~331 tokens, and a hexamer of a 250-residue NLR is ~1500 tokens — beyond a single
Colab GPU.

**Three caveats that move these numbers:**

- **dtype.** The baseline is BF16; the notebooks default to **fp32**, which needs substantially
  more memory. Switch `dtype` to `bf16` in *Run settings* for anything above ~300 tokens.
- **Kernels.** The baseline uses the PyTorch triangle kernels. cuEquivariance (selected by `auto`
  on supported GPUs) is leaner; on Blackwell it is unavailable and the notebooks fall back to the
  PyTorch path, so expect the table's numbers there.
- **Samples and steps** raise the diffusion-stage cost but not the dominant N² trunk term; reducing
  `samples_per_seed` is the first thing to try when a job is marginal.

If a job OOMs: lower `dtype` to `bf16`, cut `samples_per_seed`, split the complex, or move the
run to a multi-GPU machine and use Fold-CP.

---

## Single notebook — `OpenDDE_single.ipynb`

Predict and inspect one complex.

1. **1–3c** Setup: GPU check, install (auto-installs the Blackwell nightly torch if needed),
   helpers, Drive mount, OpenDDE weights + common files.
2. **Input** Pick one form: monomer, multimer (≤4 distinct chains), homomer (N copies), or
   protein + ligand (+ ion).
3. **Run settings** Model, checkpoint, seeds, samples per seed, steps, cycles, dtype, MSA,
   triangle kernels.
4. **6** MSA (ColabFold server) + inference. The MSA-annotated JSON is cached, so reruns skip it.
5. **7** Rank predictions by confidence.
6. **MolView viewer** — dropdowns to pick any model, colored by pLDDT, with its scores.
7. **9 / 9b** Per-residue pLDDT; inter-chain interface contacts with cross-seed recurrence.
8. **B5 / B6** PAE export (ChimeraX/ipSAE `.npz` + heatmap) and ipSAE interface scores.
9. **10 / 10b** Zip & download / archive the run.

## Batch notebook — `OpenDDE_batch.ipynb`

Predict and analyze many complexes; unique sequences are MSA-searched once and reused across jobs.

1. **1–3c** Setup (same as above).
2. **Run settings**.
3. **B1** Load a batch list (built with `make_opendde_batch.py`).
4. **B2** MSA + inference over the whole batch. The annotated `-update-msa.json` is cached on
   Drive, so reruns skip the MSA server entirely.
5. **B3** Per-job summary (best sample, ranked by interface ipTM).
6. **B4** Master confidence table — every model in the batch (CSV).
7. **B5** PAE export — best per seed (ChimeraX/ipSAE `.npz` + heatmap).
8. **B6** ipSAE interface scores (all chain pairs, CSV).
9. **B7** Per-job means across seeds — pTM / ipTM / interface-ipTM / ipSAE (mean ± std).
10. **MolView viewer** across all jobs.
11. **9b** Interface contacts — one table per job.
12. **10 / 10b** Zip & download / archive.

> Unlike the Protenix notebooks, no PAE-capture cell is needed: OpenDDE writes the PAE matrix
> directly into the full-data JSON (`token_pair_pae`) whenever `need_atom_confidence` is on.

---

## `make_opendde_batch.py`

An OpenDDE input file is a top-level **list** of job dicts; `opendde pred` / `opendde msa` iterate
over every entry. This script builds that list from FASTA files, then **B1** in the batch notebook
loads it. The schema is shared with Protenix, so the same files work with either tool.

```bash
# one job per sequence (monomers)
python make_opendde_batch.py monomer   --fasta seqs.fasta -o opendde_inputs

# homo-oligomer: each sequence as an N-mer (e.g. a hexameric resistosome cap)
python make_opendde_batch.py homomer   --fasta nlrs.fasta --copies 6 -o opendde_inputs

# N-way combinatorial screen across two or more FASTAs
python make_opendde_batch.py combos    --fastas effectors.fasta nlrs.fasta -o opendde_inputs

# every effector × NLR pair (cartesian product of two FASTAs)
python make_opendde_batch.py all_pairs --fasta_a effectors.fasta --fasta_b nlrs.fasta -o opendde_inputs

# explicit pairs from a TSV/CSV (columns: idA, idB; sequences pulled from the FASTAs)
python make_opendde_batch.py pairs     --pairs pairs.tsv --fasta_a effectors.fasta --fasta_b nlrs.fasta -o opendde_inputs
```

**Modes**

| Mode | Builds | Needs |
| --- | --- | --- |
| `monomer`   | one job per sequence | `--fasta` |
| `homomer`   | each sequence as a `--copies` N-mer | `--fasta --copies` |
| `combos`    | N-way cartesian product across two or more FASTAs | `--fastas a.fasta b.fasta …` |
| `all_pairs` | cartesian product of two FASTAs (or all within-set pairs of one) | `--fasta_a [--fasta_b]` |
| `pairs`     | explicit `idA,idB` rows from a TSV/CSV | `--pairs --fasta_a [--fasta_b]` |

**Ligands and ions** (added to every job)

- `--ligand CCD_ATP` — a CCD code (`CCD_ATP`, `ccd:ATP`, `ATP`), a SMILES string
  (`smiles:CC(=O)O`), a ligand file (`file:/path/lig.sdf`), or underscore-joined CCD codes
  (`CCD_NAG_BMA_BGC`). Per-ligand counts with `@N` (e.g. `CCD_ATP@6`), otherwise `--ligand_copies`.
- `--ligand_file cofactors.txt` — one spec per line, optional count and label
  (`smiles:OC(=O)… 1 OG7`); labels keep job names readable instead of a SMILES hash.
- `--ion MG ZN` — bare CCD ion codes (also `MG@2`); `--ion_copies N`.

**Ligand panel layout**

- `--ligand-mode all` *(default)* — every ligand co-present in one job.
- `--ligand-mode each` — one job per ligand (N inputs × M ligands); add `--include-apo` for a
  ligand-free control, `--split-by-ligand` to write each ligand into its own file/folder, and
  `--ligand-first` to put the ligand tag at the start of job names.

**Other options**

- `--seeds 101,102,103` — model seeds embedded in each job (`--split-seeds` for one job per seed)
- `--copies_a` / `--copies_b` — copy number per partner in `all_pairs` / `pairs`
- `--include_self` — in single-FASTA `all_pairs`, also pair each sequence with itself
- `--type auto|protein|dna|rna` — entity type (auto-detected per sequence by default)
- `--ids` — write explicit chain id lists (`A`, `B`, … `AA`, `AB`) instead of relying on `count`
- `--chunk 50` — split a large screen into runnable files (`batch_001.json`, …)
- `--per-job` — write one `<job_name>.json` per job instead of a single bundle
- `--header auto|yes|no` — pairs-table header handling (auto-detected, with close-match hints)
- `-o` / `--out_dir`, `--name`, `--suffix`

Each run also writes a `*_manifest.csv` (json file × job name × entity count × ligands). Job names
are sanitized and de-duplicated automatically.

---

## Outputs

Per run (under `OUT_DIR/run_<date>_<name>/`):

- `<job>/seed_<n>/predictions/` — OpenDDE `.cif` models, `*_summary_confidence_sample_*.json`, and
  `*_full_data_sample_*.json` (the latter when `need_atom_confidence` is on)
- `<job>/seed_<n>/pae/` — `pae_<model>.npz` (per-token, ChimeraX ≥1.10 + ipSAE), a ColabFold-style
  `.json`, and a heatmap `.png`
- `batch_confidence_all_models.csv`, `batch_confidence_best_per_job.csv` *(batch)*
- `batch_perjob_seed_means.csv` *(batch — pTM/ipTM/ipSAE mean ± std across seeds)*
- `ipsae_all_chain_pairs.csv`, `ipsae_best_per_job.csv`
- `interface_contacts_all_jobs.csv` (+ per-job `interface_contacts.csv`)

**Viewing PAE in ChimeraX** (≥1.10, ligand-safe):
```
open <model>.cif
open pae_<model>.npz structure #1
```
For older ChimeraX / protein-only complexes, open the `*_pae.json` with `format pae` instead.

---

## Notes & tips

- **Blackwell GPUs (sm_120).** Stock cu126 wheels lack sm_120 kernels, and cuEquivariance ships no
  sm_120 triangle kernels yet. The install cell detects sm_120, installs the nightly cu128 torch
  **into the venv automatically** (no restart — no torch is imported in that cell), and forces the
  documented PyTorch kernel path (`--trimul_kernel torch --triatt_kernel torch`,
  `LAYERNORM_TYPE=torch`). Cell 2c is a manual fallback. An **A100** needs none of this.
- **Weights.** The checkpoint (~2.6 GB) plus common runtime files (~0.6 GB: `components.cif` and
  the RDKit pickle) download once via `hf_transfer`/`aria2c` and are cached under
  `OPENDDE_ROOT_DIR` on Drive. Prefetching the common files avoids a stall mid-inference.
  `opendde_abag.pt` is available in the same cell for antibody–antigen work.
- **MSA.** OpenDDE hosts no MSA server of its own; the notebooks set
  `MMSEQS_SERVICE_HOST_URL=https://api.colabfold.com`. `opendde msa` writes
  `<input>-update-msa.json` next to the input, which is cached on Drive and reused on reruns.
- **Templates** are off. They need `hmmsearch` plus a seqres database (and a `templatesPath` block
  in the JSON), neither of which exists on a Colab runtime.
- **PAE is per-token.** For protein–protein complexes token = residue, so ChimeraX/ipSAE line up.
  Complexes with ligands have extra atom-tokens — the Boltz-style `.npz` route handles this in both
  ChimeraX (≥1.10) and ipSAE; the per-residue `.json` route does not.
- **ipSAE** runs in Boltz mode (`.npz` + `.cif`), default cutoffs 10/10 (adjustable in the cell).
- **Interface ipTM** is read from OpenDDE's `chain_pair_iptm_global` matrix.
- **Preview software.** OpenDDE is a recent preview release; CLI flags, JSON fields, and
  checkpoints may change between versions, and predictions are not guaranteed reproducible across
  releases.

## Built on

- **OpenDDE** — Aureka Research · <https://github.com/aurekaresearch/OpenDDE>
- **OpenDDE weights** — <https://huggingface.co/aurekaresearch/OpenDDE>
- **ColabFold MSA server** — Mirdita et al., *Nat. Methods* (2022)
- **ipSAE** — Dunbrack, *bioRxiv* (2025) · <https://github.com/DunbrackLab/IPSAE>
- **MolView** — Steven Yu · <https://github.com/54yyyu/molview>
- **ChimeraX** — Pettersen et al., *Protein Sci.* (2021)

## References

1. **OpenDDE.** Aureka Research. *OpenDDE: Open-Source Drug Design Engine.* Software (2026).
   https://github.com/aurekaresearch/OpenDDE
2. **OpenDDE Technical Report.** Aureka Research (2026).
   https://huggingface.co/aurekaresearch/OpenDDE/blob/main/docs/OpenDDE_Technical_reports.pdf <!-- verify title/authors before release -->
3. **AlphaFold 3.** Abramson J, Adler J, Dunger J, *et al.* Accurate structure prediction of
   biomolecular interactions with AlphaFold 3. *Nature* **630**, 493–500 (2024).
   https://doi.org/10.1038/s41586-024-07487-w
4. **Protenix.** ByteDance AML AI4Science Team. *Protenix: Advancing Structure Prediction Through a
   Comprehensive AlphaFold3 Reproduction.* bioRxiv (2025).
   https://doi.org/10.1101/2025.01.08.631967 <!-- verify DOI before release -->
5. **ColabFold.** Mirdita M, Schütze K, Moriwaki Y, Heo L, Ovchinnikov S, Steinegger M. ColabFold:
   making protein folding accessible to all. *Nature Methods* **19**, 679–682 (2022).
   https://doi.org/10.1038/s41592-022-01488-1
6. **ipSAE.** Dunbrack RL Jr. *Increasing the accuracy of protein–protein interface confidence
   estimation with ipSAE.* bioRxiv (2025). https://doi.org/10.1101/2025.02.10.637595 ·
   https://github.com/DunbrackLab/IPSAE <!-- verify title/DOI before release -->
7. **MolView.** Yu S. *MolView: a Mol\*-based molecular viewer for Jupyter/Colab.* Software.
   https://github.com/54yyyu/molview
8. **UCSF ChimeraX.** Pettersen EF, Goddard TD, Huang CC, *et al.* UCSF ChimeraX: Structure
   visualization for researchers, educators, and developers. *Protein Science* **30**, 70–82 (2021).
   https://doi.org/10.1002/pro.3943

## Citation

*Under Development!*
If you use these notebooks in your work, please cite this repository:

<!-- PLACEHOLDER — fill in after creating the Zenodo release -->
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

> Toghani, A. (2026). *OpenDDE on Colab: All-Atom Biomolecular Structure Prediction on Google Colab.*
> Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX

```bibtex
@software{toghani_opendde_colab_2026,
  author    = {Toghani, AmirAli},
  title     = {{OpenDDE on Colab: All-Atom Biomolecular Structure Prediction on Google Colab}},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v1.0.0},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

Please also cite **OpenDDE** [1, 2], which these notebooks wrap, and the tools you make use of:
**ColabFold** [5] for MSAs, **ipSAE** [6] for interface scoring, **MolView** [7] for visualization,
and **ChimeraX** [8] for PAE/structure inspection. OpenDDE builds on the **AlphaFold 3** [3]
ecosystem, including **Protenix** [4], OpenFold, and ColabFold.

## License

MIT for the notebooks and helper script. **OpenDDE itself is Apache-2.0**; the other bundled tools
(ipSAE, MolView, ColabFold) keep their own licenses — please cite them if you use this in
published work.
