# Shoplifting Video Classification

## Project Overview

This project uses a 3D Convolutional Neural Network (3D CNN) to classify shopping videos into two categories:
- `Normal shopping` (class 0)
- `Shoplifting` (class 1)

The model was trained from scratch using video data and exported to ONNX for inference.

## Dataset

- Total videos: 855
- Normal shopping videos: 531
- Shoplifting videos: 324
- Training videos: 598
- Validation videos: 128
- Test videos: 129

The dataset was split using stratified sampling with a fixed random seed.

## Model and Preprocessing

- Architecture: 3D CNN
- Trainable parameters: 292,338
- Frames sampled per video: 32
- Frame dimensions: 104 × 128
- Color format: RGB
- Pixel normalization: values scaled to [0, 1]
- Decision threshold: 0.525
- Deployment format: ONNX

Frames are sampled uniformly throughout each video. The decision threshold was selected using the validation set.

## Test Results

Evaluation on the held-out test set using the selected threshold:

| Metric | Result |
|---|---:|
| Accuracy | 79.84% |
| Precision | 70.18% |
| Recall | 81.63% |
| F1-score | 75.47% |

Confusion matrix counts:

- True negatives: 63
- False positives: 17
- False negatives: 9
- True positives: 40

These results describe performance on this dataset and do not guarantee performance on unseen real-world footage.

## Files

- `shoplifting_3dcnn.onnx` — exported inference model
- `inference_onnx.py` — script for classifying a video
- `final_evaluation_report.json` — evaluation summary
- `training_history.csv` — training metrics by epoch
- `final_shoplifting_3dcnn.pth` — final PyTorch checkpoint
- `best_shoplifting_3dcnn.pth` — best validation checkpoint
- `video_splits.csv` — saved dataset split information

## Run Inference

Install the inference dependencies:

```bash
pip install numpy opencv-python-headless onnxruntime
```

Place `shoplifting_3dcnn.onnx` and `inference_onnx.py` in the same directory, then run:

```bash
python inference_onnx.py --video "path/to/video.mp4"
```

The script prints the predicted class, shoplifting probability, decision threshold, and number of sampled frames.

## Limitations and Responsible Use

This model can produce false positives and false negatives. Its predictions must not be treated as proof that a person has committed shoplifting. Do not use predictions alone to accuse, discipline, or take action against anyone. Any practical use would require human review and additional validation on representative footage.

## Reproducibility

The saved training history, evaluation report, model checkpoints, and dataset split information support inspection of the experiment. The original dataset and training environment may be required to reproduce training.
