#!/usr/bin/env python3
"""
End-to-end CHB-MIT seizure detection pipeline (download -> preprocess -> train/eval).

Highlights for class imbalance:
1) Subject-level split to avoid leakage.
2) Window-level labeling with overlap threshold.
3) Weighted sampling + class-weighted/focal loss.
4) Threshold tuning on validation set for sensitivity/recall.

Dataset source:
- AWS Open Data mirror of PhysioNet CHB-MIT at s3://physionet-open/chbmit/

Example:
    python chbmit_cnn_lstm_pipeline.py \
      --download_dir ./data/chbmit \
      --window_sec 4 --stride_sec 2 \
      --target_fs 128 --epochs 20 --batch_size 64
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import boto3
import mne
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from botocore import UNSIGNED
from botocore.client import Config
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# ----------------------------- Reproducibility -----------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------- S3 Download ---------------------------------

def s3_client_unsigned():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def download_chbmit_from_s3(download_dir: Path, max_files: int | None = None) -> None:
    """Download CHB-MIT from AWS Open Data bucket into `download_dir`.

    Bucket/key structure:
        s3://physionet-open/chbmit/*
    """
    bucket = "physionet-open"
    prefix = "chbmit/"

    download_dir.mkdir(parents=True, exist_ok=True)
    s3 = s3_client_unsigned()

    paginator = s3.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = key[len(prefix):]
            local_path = download_dir / rel
            local_path.parent.mkdir(parents=True, exist_ok=True)
            if not local_path.exists():
                print(f"Downloading s3://{bucket}/{key} -> {local_path}")
                s3.download_file(bucket, key, str(local_path))
            count += 1
            if max_files is not None and count >= max_files:
                print(f"Stopped at max_files={max_files}")
                return


# ------------------------ CHB-MIT Annotation Parsing -----------------------

SUMMARY_FILE_RE = re.compile(r"chb\d{2}-summary\.txt", re.IGNORECASE)


@dataclass
class FileSeizureInfo:
    edf_file: str
    seizure_intervals_sec: List[Tuple[float, float]]


def parse_summary_file(summary_path: Path) -> Dict[str, FileSeizureInfo]:
    """Parse `chbXX-summary.txt`.

    Returns mapping edf filename -> seizure intervals in seconds.
    """
    text = summary_path.read_text(errors="ignore")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    file_infos: Dict[str, FileSeizureInfo] = {}
    current_file = None
    seizure_starts: List[float] = []

    for line in lines:
        if line.lower().startswith("file name"):
            m = re.search(r"File Name:\s*(\S+)", line, flags=re.IGNORECASE)
            if m:
                if current_file is not None:
                    intervals = list(zip(seizure_starts[::2], seizure_starts[1::2]))
                    file_infos[current_file] = FileSeizureInfo(
                        edf_file=current_file,
                        seizure_intervals_sec=intervals,
                    )
                current_file = m.group(1)
                seizure_starts = []
        elif "Seizure Start Time" in line:
            m = re.search(r"(\d+)\s*seconds", line, flags=re.IGNORECASE)
            if m:
                seizure_starts.append(float(m.group(1)))
        elif "Seizure End Time" in line:
            m = re.search(r"(\d+)\s*seconds", line, flags=re.IGNORECASE)
            if m:
                seizure_starts.append(float(m.group(1)))

    if current_file is not None:
        intervals = list(zip(seizure_starts[::2], seizure_starts[1::2]))
        file_infos[current_file] = FileSeizureInfo(
            edf_file=current_file,
            seizure_intervals_sec=intervals,
        )

    return file_infos


def collect_annotations(root: Path) -> Dict[str, List[Tuple[float, float]]]:
    """Collect seizure intervals for every EDF file under CHB-MIT root."""
    out: Dict[str, List[Tuple[float, float]]] = {}
    for txt in root.rglob("*.txt"):
        if SUMMARY_FILE_RE.fullmatch(txt.name):
            parsed = parse_summary_file(txt)
            for k, v in parsed.items():
                out[k] = v.seizure_intervals_sec
    return out


# ----------------------------- Preprocessing -------------------------------

def resample_and_select_channels(raw: mne.io.BaseRaw, target_fs: int) -> mne.io.BaseRaw:
    drop = [ch for ch in raw.ch_names if "ECG" in ch.upper() or "VNS" in ch.upper()]
    if drop:
        raw = raw.copy().drop_channels(drop)

    if int(raw.info["sfreq"]) != target_fs:
        raw = raw.copy().resample(target_fs)

    raw = raw.copy().filter(l_freq=0.5, h_freq=40.0, verbose=False)
    return raw


def window_label(
    start_sec: float,
    end_sec: float,
    seizure_intervals: Sequence[Tuple[float, float]],
    min_overlap_ratio: float,
) -> int:
    wlen = end_sec - start_sec
    for s0, s1 in seizure_intervals:
        overlap = max(0.0, min(end_sec, s1) - max(start_sec, s0))
        if overlap / wlen >= min_overlap_ratio:
            return 1
    return 0


def segment_file(
    edf_path: Path,
    seizure_intervals: Sequence[Tuple[float, float]],
    window_sec: float,
    stride_sec: float,
    target_fs: int,
    min_overlap_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    raw = resample_and_select_channels(raw, target_fs)
    sig = raw.get_data().astype(np.float32)  # [C, T]

    w = int(window_sec * target_fs)
    s = int(stride_sec * target_fs)
    xs, ys = [], []

    for i in range(0, sig.shape[1] - w + 1, s):
        j = i + w
        start_sec = i / target_fs
        end_sec = j / target_fs
        x = sig[:, i:j]
        y = window_label(start_sec, end_sec, seizure_intervals, min_overlap_ratio)
        xs.append(x)
        ys.append(y)

    return np.stack(xs), np.array(ys, dtype=np.int64)


# ------------------------------- Dataset -----------------------------------

class EEGWindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.x[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y


# ------------------------------- Model -------------------------------------

class CNNLSTM(nn.Module):
    """Input shape: [B, C, T]."""

    def __init__(self, in_ch: int, lstm_hidden: int = 128, lstm_layers: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )
        self.cls = nn.Sequential(
            nn.Linear(2 * lstm_hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x)                # [B, 128, T']
        z = z.transpose(1, 2)           # [B, T', 128]
        z, _ = self.lstm(z)             # [B, T', 2H]
        z = z[:, -1, :]                 # final timestep
        return self.cls(z).squeeze(1)   # logits


class FocalBCEWithLogits(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, pos_weight: float | None = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer(
            "pos_weight",
            torch.tensor([pos_weight], dtype=torch.float32) if pos_weight is not None else torch.tensor([1.0]),
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=self.pos_weight)
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


# ---------------------------- Train / Evaluate -----------------------------

def build_sampler(y: np.ndarray) -> WeightedRandomSampler:
    class_count = np.bincount(y.astype(int), minlength=2)
    class_weights = np.array([1.0 / max(class_count[0], 1), 1.0 / max(class_count[1], 1)], dtype=np.float32)
    sample_weights = class_weights[y.astype(int)]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.5) -> Dict[str, float]:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        prob = torch.sigmoid(logits).cpu().numpy()
        ys.append(y.numpy())
        ps.append(prob)

    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)

    out = {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auprc": average_precision_score(y_true, y_prob),
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
    }
    return out


@torch.no_grad()
def find_best_threshold(model: nn.Module, loader: DataLoader, device: torch.device, min_recall: float = 0.8) -> float:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        prob = torch.sigmoid(model(x)).cpu().numpy()
        ys.append(y.numpy())
        ps.append(prob)
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    best_t, best_f1 = 0.5, -1.0
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if r < min_recall:
            continue
        f1 = 2 * p * r / max(p + r, 1e-8)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    in_ch: int,
    epochs: int,
    lr: float,
    use_focal: bool,
    pos_weight: float,
    device: torch.device,
) -> Tuple[nn.Module, float]:
    model = CNNLSTM(in_ch=in_ch).to(device)
    if use_focal:
        criterion = FocalBCEWithLogits(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_state = None
    best_score = -math.inf

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        val_metrics = evaluate(model, val_loader, device, threshold=0.5)
        score = 0.7 * val_metrics["sensitivity"] + 0.3 * val_metrics["f1"]
        print(f"Epoch {epoch:03d} | train_loss={np.mean(losses):.4f} | val={val_metrics}")

        if score > best_score:
            best_score = score
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    thr = find_best_threshold(model, val_loader, device, min_recall=0.8)
    return model, thr


# ------------------------------- Orchestration -----------------------------

def subject_from_file(fname: str) -> str:
    m = re.match(r"(chb\d{2})_", fname.lower())
    return m.group(1) if m else "unknown"


def build_window_dataset(
    root: Path,
    window_sec: float,
    stride_sec: float,
    target_fs: int,
    min_overlap_ratio: float,
    max_edf_files: int | None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    annotations = collect_annotations(root)

    edf_files = sorted(root.rglob("*.edf"))
    if max_edf_files is not None:
        edf_files = edf_files[:max_edf_files]

    xs, ys, subjects = [], [], []
    for edf in edf_files:
        intervals = annotations.get(edf.name, [])
        x_file, y_file = segment_file(
            edf, intervals, window_sec, stride_sec, target_fs, min_overlap_ratio
        )
        xs.append(x_file)
        ys.append(y_file)
        sub = subject_from_file(edf.name)
        subjects.extend([sub] * len(y_file))
        print(f"Processed {edf.name}: windows={len(y_file)}, seizure_windows={int(y_file.sum())}")

    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return x, y, subjects


def subject_level_split(subjects: Sequence[str], y: np.ndarray, seed: int = 42):
    unique_subjects = sorted(set(subjects))
    train_sub, test_sub = train_test_split(unique_subjects, test_size=0.2, random_state=seed)
    train_sub, val_sub = train_test_split(train_sub, test_size=0.2, random_state=seed)

    subj_arr = np.array(subjects)
    idx_train = np.where(np.isin(subj_arr, train_sub))[0]
    idx_val = np.where(np.isin(subj_arr, val_sub))[0]
    idx_test = np.where(np.isin(subj_arr, test_sub))[0]

    return idx_train, idx_val, idx_test


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true", help="Download CHB-MIT from AWS S3 first")
    p.add_argument("--download_dir", type=Path, default=Path("./data/chbmit"))
    p.add_argument("--max_download_files", type=int, default=None)
    p.add_argument("--max_edf_files", type=int, default=None)
    p.add_argument("--window_sec", type=float, default=4.0)
    p.add_argument("--stride_sec", type=float, default=2.0)
    p.add_argument("--target_fs", type=int, default=128)
    p.add_argument("--min_overlap_ratio", type=float, default=0.3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--use_focal", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)

    if args.download:
        download_chbmit_from_s3(args.download_dir, max_files=args.max_download_files)

    x, y, subjects = build_window_dataset(
        root=args.download_dir,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        target_fs=args.target_fs,
        min_overlap_ratio=args.min_overlap_ratio,
        max_edf_files=args.max_edf_files,
    )

    print(f"Total windows: {len(y)} | seizure ratio: {y.mean():.4f}")

    idx_train, idx_val, idx_test = subject_level_split(subjects, y, seed=args.seed)

    x_train, y_train = x[idx_train], y[idx_train]
    x_val, y_val = x[idx_val], y[idx_val]
    x_test, y_test = x[idx_test], y[idx_test]

    train_ds = EEGWindowDataset(x_train, y_train)
    val_ds = EEGWindowDataset(x_val, y_val)
    test_ds = EEGWindowDataset(x_test, y_test)

    sampler = build_sampler(y_train)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    pos_weight = float(neg / max(pos, 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, threshold = train_model(
        train_loader,
        val_loader,
        in_ch=x.shape[1],
        epochs=args.epochs,
        lr=args.lr,
        use_focal=args.use_focal,
        pos_weight=pos_weight,
        device=device,
    )

    test_metrics = evaluate(model, test_loader, device, threshold=threshold)
    print(f"Best threshold from val={threshold:.4f}")
    print(f"Test metrics={test_metrics}")

    out = {
        "threshold": threshold,
        "test_metrics": test_metrics,
        "train_pos_ratio": float(y_train.mean()),
        "val_pos_ratio": float(y_val.mean()),
        "test_pos_ratio": float(y_test.mean()),
    }
    Path("results.json").write_text(json.dumps(out, indent=2))
    torch.save(model.state_dict(), "cnn_lstm_chbmit.pt")


if __name__ == "__main__":
    main()
