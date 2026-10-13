Episodic-to-semantic consolidation benchmark v2 (synthetic)
- make_synthetic.py generates episodic_semantic_dataset_v2.json deterministically (seed 2026).
- 280 samples / 979 episodes: 240 positive samples (40 per fact type: category generalization (food, exercise),
  reading preference, transport mode, temporal habit, location preference, causal-conditional) and 40 negative-control
  samples (no_generalization). 68 positive samples contain one exception episode (confidence label 0.75 instead of 1.0).
- Leakage control: 70 entities split into 5 entity-disjoint folds; every episode is rendered with three disjoint
  template sets (train / val / test). Negative-control episodes never echo one of the same entity's real facts.
- run_synthetic.py: 5-fold CV (test fold f, validation fold f+1, three training folds), 5 seeds per fold; old pipeline
  vs revised mechanism, ablation A-E, baselines, transductive vs training-only fitting, closed vs enlarged catalogue,
  weakly supervised vs fully unsupervised. Results: synthetic_results.json, synthetic_summary.csv.
