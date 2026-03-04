"""
Evaluate a saved model on the eval set and print detailed metrics.
"""
import os, sys
import numpy as np
import torch
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, roc_auc_score)
import matplotlib
matplotlib.use("Agg")           # headless backend
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from models.detector_model import get_model
from data.dataset import ASVspoofDataset
from torch.utils.data import DataLoader


def evaluate(checkpoint_path=None, split="dev"):
    device = Config.DEVICE
    model = get_model(Config.MODEL_TYPE).to(device)

    ckpt_path = checkpoint_path or os.path.join(
        Config.CHECKPOINT_DIR, "best_model.pth")
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"Loaded {ckpt_path}")

    proto = Config.ASVSPOOF_DEV_PROTOCOL if split == "dev" \
            else Config.ASVSPOOF_EVAL_PROTOCOL
    audio = Config.ASVSPOOF_DEV if split == "dev" \
            else Config.ASVSPOOF_EVAL

    ds = ASVspoofDataset(proto, audio, "mel_spectrogram")
    loader = DataLoader(ds, batch_size=64, num_workers=0)

    ys, ps, probs = [], [], []
    with torch.no_grad():
        for feats, labels in tqdm(loader, desc="Evaluating"):
            feats = feats.to(device)
            out = model(feats)
            prob = torch.softmax(out, 1)[:, 1]
            pred = out.argmax(1)
            ys.extend(labels.numpy())
            ps.extend(pred.cpu().numpy())
            probs.extend(prob.cpu().numpy())

    ys, ps, probs = np.array(ys), np.array(ps), np.array(probs)

    print("\n" + classification_report(
        ys, ps, target_names=["REAL", "FAKE"]))
    print("Confusion matrix:")
    print(confusion_matrix(ys, ps))

    # EER
    fpr, tpr, _ = roc_curve(ys, probs)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    auc = roc_auc_score(ys, probs)
    print(f"\nEER  = {eer:.4f}")
    print(f"AUC  = {auc:.4f}")

    # Save ROC curve
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC  (AUC={auc:.3f}, EER={eer:.3f})")
    plt.savefig(os.path.join(Config.LOG_DIR, "roc_curve.png"), dpi=150)
    print(f"ROC curve saved → {Config.LOG_DIR}/roc_curve.png")


if __name__ == "__main__":
    evaluate()