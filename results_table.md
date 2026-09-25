| Method                                     |   Macro-F1 |   Accuracy | Training time   | Inference cost   |
|:-------------------------------------------|-----------:|-----------:|:----------------|:-----------------|
| TF-IDF + LogReg (45K rows)                 |      0.748 |      0.842 | 1.6 min (CPU)   | ~$0              |
| DistilBERT fine-tuned (15K rows, 2 ep)     |      0.721 |      0.839 | 3 min (T4 GPU)  | ~$0              |
| DistilBERT fine-tuned (44K rows, 2 ep)     |      0.752 |      0.857 | 9 min (T4 GPU)  | ~$0              |
| Claude Haiku 4.5 v1_zero_shot (500 rows)   |      0.685 |      0.814 | none            | $0.33 per 1K     |
| Claude Haiku 4.5 v2_definitions (500 rows) |      0.676 |      0.822 | none            | $0.53 per 1K     |
| Claude Haiku 4.5 v3_few_shot (500 rows)    |      0.682 |      0.822 | none            | $0.98 per 1K     |