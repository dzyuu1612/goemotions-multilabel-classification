# Interrupted run — excluded from C3 results

The first full seed 42 attempt stopped during epoch 3 because gradient clipping raised on AMP overflow before GradScaler could skip the optimizer update. This folder preserves the actual log and original protocol. The incomplete checkpoint/data are retained in data/processed/c3_distilbert/interrupted_fp16_gradcheck_seed_42.

This attempt is not a completed seed and must not enter the final mean/std or enhancement tables. The training loop was corrected to let GradScaler skip an overflow update, reduce scale, clear gradients and advance the LR scheduler only after a successful optimizer update. The corrected code is verified on a real training batch in reports/c3_distilbert/amp_regression_check.json. All three reported seeds will use the corrected source hash and be run from the pinned pretrained checkpoint.
