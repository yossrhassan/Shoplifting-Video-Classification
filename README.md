# Shoplifting Video Classification

## Project Overview

This project explores video classification for distinguishing between normal shopping behavior and shoplifting. It compares a 3D Convolutional Neural Network (3D CNN) trained from scratch with a pretrained VideoMAE model fine-tuned for binary classification.

The main objective is to evaluate whether transfer learning improves classification performance and to understand the trade-offs in model size and inference speed.

## Dataset and Data Leakage Checks

The original dataset contained 855 video files:

- Normal shopping: 531 files
- Shoplifting: 324 files

An exact-duplicate check using SHA-256 hashes identified 637 unique video contents and 218 extra copies. Some identical videos had been assigned to different data splits.

To address this issue, one representative of each exact video duplicate was retained before creating a new stratified split.

| Split | Normal shopping | Shoplifting | Total |
|---|---:|---:|---:|
| Training | 219 | 226 | 445 |
| Validation | 47 | 49 | 96 |
| Test | 47 | 49 | 96 |
| **Total** | **313** | **324** | **637** |

The clean split is saved in `video_splits_clean.csv`. Exact duplicate video contents do not cross the training, validation, and test splits. Near-duplicate scenes and videos that have been re-encoded were not exhaustively checked.

## Models

### 1. Scratch CNN

A 3D CNN trained from random initialization, using the architecture from the Week 5 task.

- Trainable parameters: 292,578
- Frames per video: 32
- Frame dimensions: 104 × 128
- Input color format: RGB
- Pixel values scaled to the range [0, 1]

### 2. VideoMAE

A pretrained model loaded from Hugging Face:

`MCG-NJU/videomae-base-finetuned-kinetics`

The original classifier was replaced with a two-class head:

- Class 0: Normal shopping
- Class 1: Shoplifting

The model was fine-tuned on the clean training split, with validation F1-score used to select the best checkpoint.

- Trainable parameters: 86,228,738
- Frames per video: 16
- Input dimensions: 224 × 224
- Frame preprocessing: VideoMAE image processor

## Clean Test Results

Both models were evaluated on the same 96-video test set. The reported precision, recall, and F1-score refer to shoplifting as the positive class.

| Metric | Scratch CNN | VideoMAE |
|---|---:|---:|
| Accuracy | 87.50% | **91.67%** |
| Precision | 95.12% | **100.00%** |
| Recall | 79.59% | **83.67%** |
| **F1-score** | 86.67% | **91.11%** |

VideoMAE improved the test F1-score by 4.44 percentage points over the scratch baseline.

### Confusion Matrices

Rows represent the actual class and columns represent the predicted class, in the order Normal shopping and Shoplifting.

**Scratch CNN**

| | Predicted Normal | Predicted Shoplifting |
|---|---:|---:|
| Actual Normal | 45 | 2 |
| Actual Shoplifting | 10 | 39 |

**VideoMAE**

| | Predicted Normal | Predicted Shoplifting |
|---|---:|---:|
| Actual Normal | 47 | 0 |
| Actual Shoplifting | 8 | 41 |

VideoMAE produced no false positives on this test set, but it missed 8 of the 49 shoplifting videos. These results are specific to this dataset and do not guarantee performance on unseen footage.

## Model Size and Inference Speed

The forward-pass benchmark was performed on a Tesla T4 GPU using FP32 precision, batch size 1, five warm-up runs, and 30 timed runs. Video decoding and preprocessing were excluded from the timing.

| Measurement | Scratch CNN | VideoMAE |
|---|---:|---:|
| Parameters | 292,578 | 86,228,738 |
| Checkpoint size | 1.13 MB | 329.01 MB |
| Median forward-pass time | **6.92 ms** | 108.33 ms |
| Approximate throughput | **144.52 videos/s** | 9.23 videos/s |

VideoMAE achieved higher test classification scores, but it requires a much larger checkpoint and has higher forward-pass latency. The scratch CNN is substantially smaller and faster, making it an attractive option when resource constraints are important.

These timings measure model inference only; full video processing will take additional time.

## Repository Files

- `Shoplifting_Detection.ipynb` — original Week 5 notebook.
- `Shoplifting_Pretrained_Finetuning.ipynb` — pretrained fine-tuning, clean evaluation, and model comparison workflow.
- `MODEL_COMPARISON.md` — model comparison summary.
- `model_comparison_clean.csv` — comparison metrics in CSV format.
- `model_comparison_clean.json` — structured comparison results.
- `videomae_clean_test_report.json` — VideoMAE clean test report.
- `scratch_clean_test_report.json` — scratch CNN clean test report.
- `videomae_training_history.csv` — VideoMAE training history.
- `scratch_clean_training_history.csv` — scratch CNN training history.
- `videomae_training_summary.json` — VideoMAE training summary.
- `scratch_clean_training_summary.json` — scratch CNN training summary.
- `video_splits_clean.csv` — clean dataset split.
- `inference_onnx.py` and `shoplifting_3dcnn.onnx` — inference script and ONNX model from the earlier Week 5 experiment.

The fine-tuned VideoMAE checkpoint is not included in this repository because of its size. It was saved separately during training. The notebook can be used to reproduce the fine-tuning workflow, subject to access to the dataset and sufficient compute resources.

## Reproducibility

The comparison uses a fixed random seed of 42 for the clean split and records training histories and evaluation reports. Both models were evaluated on the same held-out test videos.

The earlier Week 5 test result was obtained before the exact duplicate issue was discovered. It is retained as historical information, but it should not be treated as the clean baseline or directly compared with the new test results.

## Limitations and Responsible Use

The dataset is limited in size and may not represent real-world shop environments. Exact hash matching does not detect every near-duplicate video or similar scene, and results may vary on different cameras, environments, and behaviors.

This model is a research experiment. Its predictions are not proof of wrongdoing and must not be used alone to accuse, discipline, or take action against anyone. Any real-world use would require additional validation and human review.
