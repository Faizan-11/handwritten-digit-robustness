# Handwritten Digit Classification: Baseline vs Neural Network

## Overview

This project compares logistic regression with a small multilayer perceptron (MLP) on the handwritten-digits dataset distributed by scikit-learn.

The goal is to practise reproducible experimentation, compare two classification approaches, and analyse their errors. This is a preliminary learning experiment, not a claim of novel computer-vision research.

## Dataset

- Source: `sklearn.datasets.load_digits`
- Samples: 1,797
- Image dimensions: 8 × 8 grayscale pixels
- Classes: digits 0–9
- Pixel values: normalised from the original range [0, 16] to [0, 1]

## Experimental design

A stratified split with random seed 42 was used:

- Training: 1,149 samples
- Validation: 288 samples
- Test: 360 samples

The models were:

1. **Baseline:** standardised pixel features with logistic regression.
2. **MLP:** a fully connected neural network with hidden layers of 128 and 64 units, ReLU activations, and dropout of 0.2.

The MLP was trained for 50 epochs using Adam with a learning rate of 0.001 and batch size 64. The best checkpoint was selected by validation accuracy. The test set was evaluated after model selection.

## Results

| Model | Test accuracy | Test macro-F1 | Training time |
|---|---:|---:|---:|
| Logistic regression | 96.11% | 96.05% | 0.015 s |
| MLP | 97.22% | 97.19% | 2.002 s |

On this split, the MLP correctly classified 350 of 360 test samples, compared with 346 of 360 for logistic regression. This is a difference of four samples and should not be interpreted as proof of general superiority.

The timing values are specific to this run and small dataset; they are not general hardware benchmarks.



## Robustness experiment

The trained models were evaluated on the same held-out test set after adding Gaussian noise to normalised pixel values. Noise standard deviations were 0.00, 0.05, 0.10, 0.20, and 0.30. At each repetition, the same standard-normal noise array was scaled across the severity levels and pixel values were clipped to [0, 1]. Five independent noise patterns were evaluated; the trained models and test set remained fixed.

| Noise standard deviation | Logistic regression accuracy (mean ± SD) | MLP accuracy (mean ± SD) |
|---:|---:|---:|
| 0.00 | 96.11% ± 0.00 pp | 97.22% ± 0.00 pp |
| 0.05 | 93.78% ± 0.75 pp | 96.50% ± 0.58 pp |
| 0.10 | 84.94% ± 0.30 pp | 95.33% ± 0.84 pp |
| 0.20 | 61.83% ± 2.87 pp | 92.00% ± 1.07 pp |
| 0.30 | 41.44% ± 1.34 pp | 84.22% ± 1.68 pp |

On this test split and under this synthetic corruption, the MLP had higher mean accuracy at every tested noise level. These repetitions measure variation across noise patterns, not across independently trained models or datasets. They do not establish general robustness or performance on real-world image corruptions.

The per-run measurements and summary statistics are saved in `results/robustness.json`.

## Reproducibility

Activate the project environment and run:

```bash
python src/experiment.py

The script uses a fixed seed and writes:

- `results/metrics.json`
- `results/baseline_confusion_matrix.csv`
- `results/mlp_confusion_matrix.csv`
- `results/test_predictions.csv`
- `results/error_analysis.json`

Dependencies are listed in `requirements.txt`. The complete installed environment is recorded in `requirements-lock.txt`.

## Limitations

- The dataset is small and consists of 8 × 8 handwritten digit images.
- Only one train/validation/test split and one random seed were used.
- Only one configuration per model was evaluated.
- The MLP has more trainable parameters than logistic regression.
- The small test set means the four-example difference should be interpreted cautiously.
- This experiment does not establish novelty or generalisation to more complex computer-vision datasets.

## Next steps

- Inspect misclassified examples and compare errors between models.
- Repeat experiments across multiple seeds and report variability.
- Consider a more challenging real-world image dataset when it can be obtained reliably.

