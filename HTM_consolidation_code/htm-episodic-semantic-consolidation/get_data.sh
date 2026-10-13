#!/usr/bin/env bash
# Downloads the external resources that are too large or not ours to redistribute.
set -e
mkdir -p data
# 1. GloVe 100-d vectors (Wikipedia + Gigaword), used by the end-to-end, generator and LoCoMo experiments.
#    (glove_clinc.npy and synthetic/glove_synthetic.npy, the vectors for the CLINC150 and benchmark vocabularies, are included.)
if [ ! -f glove100.gz ]; then
  curl -L -o data/glove.6B.zip https://nlp.stanford.edu/data/glove.6B.zip
  unzip -p data/glove.6B.zip glove.6B.100d.txt | gzip > glove100.gz
fi
# 2. LoCoMo (Maharana et al., 2024)
curl -L -o data/locomo10.json https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
# 3. CLINC150 (data_full.json) is included; source: https://github.com/clinc/oos-eval
echo "done"
