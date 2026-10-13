# Buffer-Enhanced HTM Columns for Episodic-to-Semantic Knowledge Consolidation

Code, data and raw results for the manuscript *Buffer-Enhanced HTM Columns for Episodic-to-Semantic Knowledge
Consolidation: Mechanism, Ablations, and Comparative Analysis* (Journal of Computer Science, under review).

* **Stage 1** consolidates a stream of episodic sentences into semantic facts: similarity routing, an ART-style
  vigilance test, running-mean column templates, periodic column merging and closed-set decoding.
* **Stage 2** learns transitions between consolidated facts with a buffer-enhanced cell-level Temporal Memory and a
  transition detector; a transformer generator verbalizes them as compound sequential facts.

The synthetic benchmark is also available on Kaggle:
https://www.kaggle.com/datasets/rahulshrivastava06/episodic-to-semantic-consolidation-benchmark

## Setup

```bash
pip install -r requirements.txt
bash get_data.sh          # GloVe-100 vectors and LoCoMo (only needed for the experiments marked *)
```

All scripts are run **from the repository root**, e.g. `python run_improved.py`. Experiments use Python
multiprocessing (2 workers) and run on a CPU. Each script writes its raw per-run results to the JSON file named below;
the JSON files in this repository are the results reported in the paper.

## Code

| File | Content |
|---|---|
| `htm_consolidation.py` | Stage 1 base configuration (random-projection shortlist, gates, Eqs. 1–6), TF-IDF+SVD encoder, contrastive projection |
| `htm_v2.py` | Revised Stage 1: similarity routing, column merging, optional gates and update rules; GloVe-SIF encoder |
| `htm_v3.py` | Replay-based offline consolidation with column splitting (tested, not adopted) |
| `new_baselines.py` | Fuzzy ART, DenStream, CluStream, online nearest-prototype memory |
| `run_clinc.py` | CLINC150 protocol (90/30/30 intent split, out-of-scope negatives), metrics, classical baselines |
| `stage2/tm_buffer.py` | Cell-level Temporal Memory with the decaying predictive buffer (Algorithm 2) |
| `stage2/seqgen*.py` | Sentence generators (LSTM, attention LSTM, column attention, relation-aware template, transformer) |

## Reproducing the results

| Manuscript | Script | Output |
|---|---|---|
| Table 1 (benchmark composition) | `synthetic/make_synthetic.py` | `synthetic/episodic_semantic_dataset_v2.json` |
| Table 3, synthetic (original pipeline, A–E, baselines) | `synthetic/run_synthetic.py` | `synthetic/synthetic_results.json` |
| Table 3, CLINC150 (A–E, baselines) | `run_v2_clinc.py` | `v2_results.json` |
| Table 3, added online baselines; Tables 4–5 (settings, runtime, factorial ablation) | `run_c120.py` | `c120_results_final.json` |
| Table 3, final configuration F (validation-selected) | `run_improved.py` | `improved_results.json` |
| Tables 6–7 (100-run results, extended measures) | `synthetic/run_extended.py`, `run_final_ext.py` | `synthetic/extended_results.json`, `final_ext_results.json` |
| Section 5.2/5.3 checks (leakage, closed set, unsupervised) | `run_leakage.py`, `synthetic/run_unsup_compare.py`, `synthetic/run_lfto.py` | `leakage_results.json`, `synthetic/unsup_compare_results.json`, `synthetic/lfto_results.json` |
| Table 8 (forgetting, gradual drift) | `run_forgetting.py`, `run_drift.py` | `forgetting_results.json`, `drift_results.json` |
| Table 9 (scaling); replay consolidation | `run_scaling.py`, `run_v3_main.py` | `scaling_results.json`, `v3_results.json` |
| Table 10 (skewed and long streams, abrupt drift, transfer) | `run_cl2.py` (`python run_cl2.py skew` for the skewed test) | `cl2_results.json`, `cl2_skew.json` |
| Transformer pooling (Section 5.3) | `run_tfpool.py` | `tfpool_results.json` |
| LoCoMo (Section 5.4) * | `run_locomo.py`, `locomo_sweep.py` | `locomo_results.json`, `locomo_sweep.json` |
| Tables 13–14 (sentence generation under disjoint splits) * | `stage2/seqgen.py`, `stage2/seqgen2.py`, `stage2/seqgen3.py`, `stage2/seqgen4.py` | `stage2/seqgen*_results.json` |
| Tables 15–16 (Temporal Memory, stress tests) | `stage2/run_stage2.py` | `stage2/stage2_results.json` |
| Table 17 (end-to-end) * | `e2e/run_e2e_v1.py` (true facts, contrastive projection), `e2e/run_e2e.py` (fact-anchored projection) | `e2e/e2e_results.json`, `e2e/e2e2_results.json` |

Some scripts read the settings selected in earlier runs (for example `improved_results.json`), so run them in the
order of the table. `run_clinc.py`, `run_improve.py`, `sens.py` and `sweep.py` contain earlier exploratory runs of
the CLINC150 protocol (`clinc_results.json`, `improve_results.json`, `sens_results.json`, `sweep_results.json`).

The development experiments of Section 5.1 and the Supplementary Material, and the generation results of Tables 11–12, were run with the original implementation
on an earlier dataset version and are not reproduced by this repository.

## Data

* `synthetic/` — the synthetic benchmark (280 samples, 979 episodes, 70 entities in 5 entity-disjoint folds,
  three disjoint template sets) and its generator; see `synthetic/README_dataset.txt`.
* `data_full.json` — CLINC150 (Larson et al., 2019), https://github.com/clinc/oos-eval.
* `glove_clinc.npy`, `synthetic/glove_synthetic.npy` — GloVe-100 vectors (Pennington et al., 2014) for the
  vocabularies of CLINC150 and the benchmark.

## Citation

Please cite the accompanying paper (full reference to be added on publication).
