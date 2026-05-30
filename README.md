# Mining Contrast Patterns over Retrieved Evidence Sets for Generation-Failure Discovery in RAG

Code and configuration accompanying the ICDM 2026 submission of the same title.

## Quick start

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Pipeline (primary run: BM25 + Qwen-2.5-1.5B-Instruct)

```bash
python scripts/01_retrieve.py              --config config/default.yaml
python scripts/02_featurize.py             --config config/default.yaml
python scripts/02b_generate_and_label.py   --config config/default.yaml
python scripts/02c_label_generations.py    --config config/default.yaml
python scripts/03_signatures.py            --config config/default.yaml
python scripts/04b_mine_patterns.py        --config config/default.yaml --labels generation
python scripts/05b_pattern_eval.py         --config config/default.yaml
python scripts/06b_pattern_report.py       --config config/default.yaml
```

## Robustness variants

* **Dense retrieval (BGE-small):** run `scripts/01b_retrieve_dense.py` instead of `01_retrieve.py`, then use `config/dense.yaml` for the remaining steps. `cache_dir` is set to `cache_dense` to isolate from the primary featurization cache.
* **Cross-family generator (Phi-3.5-mini-instruct):** copy primary `retrievals.pkl`, `passage_attrs.pkl`, and `histograms.csv` into `outputs_phi/`, then run the pipeline from 02b onwards with `config/phi.yaml`.

## Running the Phi-3.5-mini variant on a free GPU

The Phi-3.5-mini generation step (~6-9 hours on CPU, exceeds 16 GB of RAM in fp32) can be offloaded to a free Google Colab T4 GPU using `notebooks/colab_phi.ipynb`. The notebook bundles the necessary scripts, runs `02b_generate_and_label.py` on GPU at fp16 in ~20 minutes, and saves the resulting `generations.jsonl` to Google Drive. The downstream steps (02c onward) are fast enough to run locally on a CPU.

## Aggregating the three runs

```bash
python scripts/08_robustness_compare.py
```

Writes `outputs_robustness/robustness_comparison.csv`.

## Expected headline results

| Condition                          |   n | Pattern AUC | Baseline AUC |
| ---------------------------------- | --: | ----------: | -----------: |
| BM25 + Qwen-2.5-1.5B-Instruct      | 536 |       0.612 |        0.498 |
| BGE-small + Qwen-2.5-1.5B-Instruct | 599 |       0.738 |        0.676 |
| BM25 + Phi-3.5-mini-instruct       | 536 |       0.693 |        0.478 |

The primary result is produced by the standard pipeline above. The dense-retrieval and Phi-3.5-mini variants are produced using the robustness settings described above. The three runs can be summarized with:

```bash
python scripts/08_robustness_compare.py
```

This writes:

```bash
outputs_robustness/robustness_comparison.csv
```

## Hardware

Experiments executed on a commodity laptop CPU (Intel Core i7 family, 16 GB RAM). The Phi-3.5-mini generation pass used a single NVIDIA T4 GPU via Google Colab.
