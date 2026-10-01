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

Requires Python 3.13 and a Java runtime for jSymbolic (tested with OpenJDK 25).

Setup and run instructions will be added as the pipeline is built.
