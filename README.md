# Composer Classification from Symbolic Music

UE24CS352A Machine Learning mini-project, PES University, Section C, Team 10.

- Vivian Sobers E (PES1UG24CS901)
- C Nitish (PES1UG25CS809)

## Problem

Problem statement #67: classify pieces of Western classical music by composer from their MIDI encoding alone, and find which musical properties separate composers best. The project reimplements and re-evaluates [Jain, Smit and Yngesjö, "Analysis and Classification of Symbolic Western Classical Music by Composer" (Stanford CS229, 2019)](https://cs229.stanford.edu/proj2019aut/data/assignment_308832_raw/26583519.pdf). The original dataset was never released, so this project uses public data.

## Data

- [midi-classical-music](https://huggingface.co/datasets/drengskapur/midi-classical-music): 4,796 MIDI files with piano, chamber and orchestral works. Used for the feature-based classifiers.
- [MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro): 1,276 piano performances with a split that keeps each composition in one subset. Used for the piano-roll CNN and a like-for-like comparison against the feature-based models.

Only composers with at least 50 pieces are kept.

## Method

- Features: [jSymbolic 2.2](https://sourceforge.net/projects/jmir/files/jSymbolic/) extracts 1,495 values per piece (pitch, melodic and vertical intervals, rhythm, texture, instrumentation, dynamics). music21's feature extractors are the fallback.
- Classifiers: LightGBM, softmax regression, SVM, LDA, k-NN and an MLP, each given the same tuning budget.
- Evaluation: train/dev/test splits grouped by musical work so that movements or copies of one piece never appear on both sides.
- CNN: a piano-roll image classifier following the original paper's architecture.

## Setup

You need Python 3.13 and a Java runtime for jSymbolic. We tested with OpenJDK 25 on our laptop and OpenJDK 11 on the GPU server.

```bash
python3.13 -m venv .venv
.venv/bin/pip install -e ".[dev,app]"      # add ,cnn for the piano-roll CNN
```

The CPU build of PyTorch is enough for the tests. For training the CNN on a GPU, install the default CUDA build instead (`pip install -e ".[cnn]"`).

Download the two datasets and jSymbolic (about 200 MB in total). The Hugging Face files come down four at a time because the server rate-limits faster downloads, so this step takes around 15 minutes:

```bash
.venv/bin/python -m composer_clf.data.download all
```

Then build the catalogs. This removes duplicate files, gives every file a piece ID (all movements and copies of one work share it) and keeps composers with at least 50 files and 5 distinct pieces:

```bash
.venv/bin/python -m composer_clf.data.catalog
```

Run the tests with `.venv/bin/pytest`. The tests that start jSymbolic are marked slow and skipped by default; run them with `.venv/bin/pytest -m slow`.

## Running the experiments

Extract jSymbolic features for both datasets. Each run splits the files into shards and runs several Java processes at once. On a 16-core laptop with 14 GB of RAM, four workers was the most we could run; the Hugging Face set then takes roughly two hours. Files that jSymbolic can't read are listed in `data/processed/<dataset>_jsymbolic_failed.txt`.

```bash
.venv/bin/python -m composer_clf.features.extract --dataset hf
.venv/bin/python -m composer_clf.features.extract --dataset maestro --shards 48 --workers 8
```

Every experiment writes one folder per model and seed under `results/<experiment>/`, holding the full config (including the git commit and library versions), the metrics, the dev-set tuning table and the test predictions. Each model gets the same eight hyperparameter settings, the best one on the dev set is kept, and the test set is scored once. Seeds 0 to 4 are used unless you pass `--seeds`.

```bash
# E1: six classifiers, splits grouped by piece
.venv/bin/python -m composer_clf.run_tabular --exp e1_composers --split grouped

# E2: the same models with a random file-level split, to measure leakage
.venv/bin/python -m composer_clf.run_tabular --exp e2_random_split --split random --models lightgbm softmax svm

# E3: LightGBM on one feature group at a time, for composers and for eras
for g in rhythm melodic vertical texture instrumentation no_instrumentation; do
  .venv/bin/python -m composer_clf.run_tabular --exp e3_composer_$g --features $g --models lightgbm
done
for g in all rhythm melodic vertical texture; do
  .venv/bin/python -m composer_clf.run_tabular --exp e3_era_$g --task era --features $g --models lightgbm
done

# E4: the six classifiers on MAESTRO's official split
.venv/bin/python -m composer_clf.run_tabular --exp e4_maestro_tabular --dataset maestro --split official

# E5: the piano-roll CNN on MAESTRO (run on a GPU)
.venv/bin/python -m composer_clf.cnn.run --exp e5_cnn_noaug --repeats 1
.venv/bin/python -m composer_clf.cnn.run --exp e5_cnn_aug10 --augment --repeats 10
```

`python -m composer_clf.report` collects every `summary.csv` into `results/tables.md`.

## Demo

The demo is a Streamlit app. You upload a MIDI file, and it shows the predicted composer, the probability of each composer, the features that pushed the prediction most and the piano roll. It uses the LightGBM model selected in E1 for seed 0:

```bash
.venv/bin/python -m composer_clf.train_final
.venv/bin/streamlit run app/streamlit_app.py
```

## Differences from the original paper

- Data. The paper used about 10,000 MIDI files donated by Classical Archives, which were never released. We use the two public datasets above, so our composer set and our numbers are not directly comparable with theirs.
- Splits. The paper split files at random after removing duplicate titles, so different movements or versions of one work could land in both train and test. We group every split by piece, and E2 measures what a random split would have reported instead.
- Scaling. The paper's code fitted a separate `StandardScaler` on the dev set. Ours fits the scaler on the training set only, inside each model's pipeline.
- Tuning. The paper tuned LightGBM by hand and the SVM by grid search. Every model here gets the same budget of eight configurations, chosen on dev macro-F1.
- CNN evaluation. The paper's code set `model.training = False` before scoring, which leaves BatchNorm in training mode, and its released code never scores the test set. We call `model.eval()`, choose the epoch on the dev set and score the test set once.
- MLP. The paper used PyTorch; we use scikit-learn's `MLPClassifier` with the same architecture and optimizer settings.
