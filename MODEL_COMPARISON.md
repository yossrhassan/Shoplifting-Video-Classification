# Pretrained Video Model Comparison

## Evaluation protocol

Both models were evaluated on the same clean test split of 96 videos.
The dataset contains 637 unique video files after exact duplicates
were grouped by SHA-256 and one representative retained.

| Model | Accuracy | Precision | Recall | F1-score | Parameters | Checkpoint (MB) | Median forward (ms) | Videos/sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Scratch CNN | 0.8750 | 0.9512 | 0.7959 | 0.8667 | 292,578 | 1.13 | 6.92 | 144.52 |
| VideoMAE | 0.9167 | 1.0000 | 0.8367 | 0.9111 | 86,228,738 | 329.01 | 108.33 | 9.23 |

## Findings

VideoMAE achieved an F1-score of 0.9111,
compared with 0.8667 for the Scratch CNN.
The improvement is 4.44
percentage points on the same clean test set.

VideoMAE has more parameters and a larger checkpoint, and its
forward-pass latency is higher. The timing benchmark uses FP32,
batch size 1, a Tesla T4, five warm-up runs and 30 timed runs.
Video decoding and preprocessing are excluded.

## Limitations

Exact duplicate detection does not rule out near-duplicate videos or
similar scenes. The results are specific to this dataset and split.
Shoplifting predictions should not be treated as proof of wrongdoing.
The earlier Week 5 result is not directly comparable because its
original split contained exact duplicate videos across splits.
