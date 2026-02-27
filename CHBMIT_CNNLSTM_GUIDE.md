# CHB-MIT Seizure Detection (CNN-LSTM) Practical Guide

This guide is focused on your issue: **high non-seizure accuracy but 0 seizure recall**.

## Why this happens

When seizure windows are only 1–5%, a model can minimize loss by predicting almost everything as non-seizure. Typical failure causes:

1. Random window split causes leakage or skewed label distribution.
2. Inadequate positive sampling (too few seizure windows per batch).
3. Loss not weighted for minority class.
4. Threshold fixed at 0.5 instead of tuned for sensitivity.
5. Weak labeling policy (seizure event touches only tiny part of a long window).

---

## What successful CHB-MIT papers usually do

Common patterns across CHB-MIT seizure detection literature (including CNN/LSTM hybrids):

1. **Patient-wise split** (train/val/test by patient, not random windows).
2. **Short windows** (2–8 s), often with overlap (50%+).
3. **Bandpass filtering** (roughly 0.5–40 or 0.5–70 Hz).
4. **Class-imbalance mitigation**:
   - class-weighted loss (`pos_weight`),
   - oversampling positives / balanced batches,
   - focal loss,
   - threshold tuning for high recall.
5. **Post-processing** at event level (smoothing, minimum duration) to reduce false alarms.

---

## Recommended training recipe

1. Download CHB-MIT from `s3://physionet-open/chbmit/`.
2. Parse `chbXX-summary.txt` for seizure start/end times.
3. Segment each EDF into windows:
   - `window_sec=4`, `stride_sec=2` (good baseline),
   - label positive if seizure overlap >= 30% of window.
4. Use subject-level train/val/test split.
5. In training:
   - `WeightedRandomSampler` so each batch has more positives,
   - `BCEWithLogitsLoss(pos_weight=neg/pos)` or focal loss,
   - tune threshold on validation PR curve for desired recall (e.g., >=0.8).
6. Evaluate with sensitivity/recall, specificity, AUPRC, and event-level metrics.

---

## How to run the provided script

```bash
python chbmit_cnn_lstm_pipeline.py \
  --download \
  --download_dir ./data/chbmit \
  --window_sec 4 --stride_sec 2 \
  --target_fs 128 \
  --epochs 20 --batch_size 64 \
  --use_focal
```

For quick debug on small subset:

```bash
python chbmit_cnn_lstm_pipeline.py \
  --download_dir ./data/chbmit \
  --max_edf_files 8 \
  --epochs 3
```

---

## If seizure recall is still near 0

Try in this exact order:

1. Print training label ratio after split (`train_pos_ratio`).
2. Verify each training batch contains positives.
3. Increase `pos_weight` multiplier (e.g. `1.5 * neg/pos`).
4. Decrease window size to 2 sec if events are short.
5. Raise overlap (stride 1 sec).
6. Lower decision threshold based on val PR curve.
7. Add simple event smoothing over consecutive windows.

---

## Notes on reported high sensitivity in papers

Many reported high sensitivity values come with one or more of:

1. Highly tuned post-processing at event level.
2. Different split protocol (sometimes not fully patient-independent).
3. Performance reported on selected patients/seizure types.
4. Optimization toward sensitivity at cost of false positives.

So, compare only when protocol is matched: same split, same windowing, same metric definition.
