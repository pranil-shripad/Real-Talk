"""
Diagnose WHY the model is classifying everything as fake.
Run this FIRST before retraining.
"""
import os, sys, random
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config
from features.feature_extractor import FeatureExtractor
from models.detector_model import get_model


def load_model():
    device = Config.DEVICE
    model = get_model(Config.MODEL_TYPE).to(device)
    
    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "final_model.pth")
    
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, device


def load_protocol(protocol_path):
    samples = []
    with open(protocol_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                samples.append({
                    "name": parts[1],
                    "label": parts[4],  # "bonafide" or "spoof"
                })
    return samples


def main():
    print("=" * 60)
    print("  MODEL DIAGNOSIS")
    print("=" * 60)

    model, device = load_model()
    extractor = FeatureExtractor("mel_spectrogram")

    # ── Check 1: Dataset balance ──────────────────────────
    print("\n📊  CHECK 1: Dataset balance")
    train_samples = load_protocol(Config.ASVSPOOF_TRAIN_PROTOCOL)
    real_count = sum(1 for s in train_samples if s["label"] == "bonafide")
    fake_count = sum(1 for s in train_samples if s["label"] == "spoof")
    ratio = fake_count / max(real_count, 1)
    
    print(f"  Training set:")
    print(f"    Real (bonafide): {real_count}")
    print(f"    Fake (spoof):    {fake_count}")
    print(f"    Ratio fake:real: {ratio:.1f}:1")
    
    if ratio > 3:
        print(f"  ⚠️  PROBLEM: Dataset is heavily imbalanced ({ratio:.0f}x more fakes)")
        print(f"     The model learned to just predict 'FAKE' for everything!")

    # ── Check 2: Raw output distribution ──────────────────
    print(f"\n📊  CHECK 2: Model output distribution")
    
    dev_samples = load_protocol(Config.ASVSPOOF_DEV_PROTOCOL)
    
    # Get some real and fake samples
    real_files = [s for s in dev_samples if s["label"] == "bonafide"]
    fake_files = [s for s in dev_samples if s["label"] == "spoof"]
    
    random.seed(42)
    test_real = random.sample(real_files, min(30, len(real_files)))
    test_fake = random.sample(fake_files, min(30, len(fake_files)))

    real_probs = []
    fake_probs = []

    print("\n  Testing on REAL (bonafide) samples:")
    for s in tqdm(test_real, desc="  Real samples"):
        path = os.path.join(Config.ASVSPOOF_DEV, s["name"] + ".flac")
        if not os.path.exists(path):
            continue
        try:
            feat = extractor.from_file(path).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(feat)
                probs = torch.softmax(logits, 1).cpu().numpy()[0]
            real_probs.append(probs)
        except:
            pass

    print("\n  Testing on FAKE (spoof) samples:")
    for s in tqdm(test_fake, desc="  Fake samples"):
        path = os.path.join(Config.ASVSPOOF_DEV, s["name"] + ".flac")
        if not os.path.exists(path):
            continue
        try:
            feat = extractor.from_file(path).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(feat)
                probs = torch.softmax(logits, 1).cpu().numpy()[0]
            fake_probs.append(probs)
        except:
            pass

    real_probs = np.array(real_probs)
    fake_probs = np.array(fake_probs)

    print(f"\n  Results on REAL audio (should predict class 0 = REAL):")
    print(f"    Avg P(real): {real_probs[:, 0].mean():.3f}")
    print(f"    Avg P(fake): {real_probs[:, 1].mean():.3f}")
    print(f"    Predicted REAL: {(real_probs[:, 0] > real_probs[:, 1]).sum()}/{len(real_probs)}")
    print(f"    Predicted FAKE: {(real_probs[:, 1] > real_probs[:, 0]).sum()}/{len(real_probs)}")

    print(f"\n  Results on FAKE audio (should predict class 1 = FAKE):")
    print(f"    Avg P(real): {fake_probs[:, 0].mean():.3f}")
    print(f"    Avg P(fake): {fake_probs[:, 1].mean():.3f}")
    print(f"    Predicted REAL: {(fake_probs[:, 0] > fake_probs[:, 1]).sum()}/{len(fake_probs)}")
    print(f"    Predicted FAKE: {(fake_probs[:, 1] > fake_probs[:, 0]).sum()}/{len(fake_probs)}")

    # ── Check 3: Is model just predicting one class? ─────
    print(f"\n📊  CHECK 3: Is model collapsed?")
    all_probs = np.concatenate([real_probs, fake_probs])
    all_preds = all_probs.argmax(axis=1)
    pred_fake_pct = (all_preds == 1).mean() * 100
    
    print(f"    Predicts FAKE: {pred_fake_pct:.1f}% of the time")
    print(f"    Predicts REAL: {100 - pred_fake_pct:.1f}% of the time")
    
    if pred_fake_pct > 80:
        print(f"  ⚠️  PROBLEM: Model is biased — predicts FAKE {pred_fake_pct:.0f}% of the time")
        print(f"     This is caused by class imbalance + wrong loss weights")
    elif pred_fake_pct < 20:
        print(f"  ⚠️  PROBLEM: Model is biased — predicts REAL {100-pred_fake_pct:.0f}% of the time")

    # ── Check 4: Find optimal threshold ──────────────────
    print(f"\n📊  CHECK 4: Optimal threshold search")
    
    true_labels = ([0] * len(real_probs)) + ([1] * len(fake_probs))
    fake_scores = np.concatenate([real_probs[:, 1], fake_probs[:, 1]])
    
    best_acc = 0
    best_thresh = 0.5
    
    for thresh in np.arange(0.05, 0.95, 0.05):
        preds = (fake_scores > thresh).astype(int)
        acc = (preds == np.array(true_labels)).mean()
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    
    print(f"    Default threshold (0.5):  acc = {((fake_scores > 0.5).astype(int) == np.array(true_labels)).mean()*100:.1f}%")
    print(f"    Optimal threshold ({best_thresh:.2f}): acc = {best_acc*100:.1f}%")
    
    if abs(best_thresh - 0.5) > 0.15:
        print(f"  ⚠️  Threshold is far from 0.5 — model outputs are miscalibrated")

    # ── Summary ──────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  DIAGNOSIS SUMMARY")
    print(f"{'=' * 60}")
    print(f"""
  Most likely causes (in order):

  1. CLASS IMBALANCE: ASVspoof has {ratio:.0f}x more fake than real samples.
     The model learned it's "safer" to always predict FAKE.

  2. WRONG CLASS WEIGHTS: The loss weights [1.0, 0.1] in training
     were not aggressive enough to counter the imbalance.

  3. INSUFFICIENT TRAINING: Model may not have converged properly.

  ✅  FIX: Run the retrain script (retrain_balanced.py) provided next.
  It uses:
     • Balanced sampling (50% real, 50% fake per batch)
     • Correct loss weights
     • Better learning rate schedule
     • Focal loss to handle hard examples
    """)


if __name__ == "__main__":
    main()