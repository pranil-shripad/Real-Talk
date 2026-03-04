"""
Detailed accuracy audit — shows EXACTLY where and why the model fails.
Run this first to understand the problem.
"""
import os, sys, random
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config
from features.feature_extractor import FeatureExtractor
from models.detector_model import get_model


def load_model():
    device = Config.DEVICE
    model = get_model(Config.MODEL_TYPE).to(device)

    for name in ["best_model_balanced.pth", "best_model.pth", "final_model.pth"]:
        p = os.path.join(Config.CHECKPOINT_DIR, name)
        if os.path.exists(p):
            ck = torch.load(p, map_location=device)
            key = "model" if "model" in ck else "model_state_dict"
            model.load_state_dict(ck[key])
            thresh = ck.get("threshold", 0.5)
            print(f"✓ Loaded: {name}  (threshold={thresh:.3f})")
            break

    model.eval()
    return model, device, thresh


def main():
    print("=" * 60)
    print("  ACCURACY AUDIT")
    print("=" * 60)

    model, device, threshold = load_model()
    ext = FeatureExtractor("mel_spectrogram")

    # Load ALL dev samples
    with open(Config.ASVSPOOF_DEV_PROTOCOL) as f:
        lines = f.readlines()

    real_names = [l.split()[1] for l in lines if "bonafide" in l]
    fake_names = [l.split()[1] for l in lines if "spoof" in l]

    random.seed(42)
    test_real = random.sample(real_names, min(200, len(real_names)))
    test_fake = random.sample(fake_names, min(200, len(fake_names)))

    # ── Collect all predictions ──
    real_scores = []  # P(fake) for real samples
    fake_scores = []  # P(fake) for fake samples

    print(f"\nScanning {len(test_real)} real + {len(test_fake)} fake ...")

    for name in tqdm(test_real, desc="Real samples"):
        path = os.path.join(Config.ASVSPOOF_DEV, name + ".flac")
        if not os.path.exists(path):
            continue
        try:
            feat = ext.from_file(path).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(feat), 1).cpu().numpy()[0]
            real_scores.append(probs[1])  # P(fake)
        except:
            pass

    for name in tqdm(test_fake, desc="Fake samples"):
        path = os.path.join(Config.ASVSPOOF_DEV, name + ".flac")
        if not os.path.exists(path):
            continue
        try:
            feat = ext.from_file(path).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(feat), 1).cpu().numpy()[0]
            fake_scores.append(probs[1])  # P(fake)
        except:
            pass

    real_scores = np.array(real_scores)
    fake_scores = np.array(fake_scores)

    # ── Analysis ──
    print(f"\n{'=' * 60}")
    print(f"  SCORE DISTRIBUTIONS")
    print(f"{'=' * 60}")

    print(f"\n  REAL voices — P(fake) scores:")
    print(f"    Min:    {real_scores.min():.3f}")
    print(f"    Max:    {real_scores.max():.3f}")
    print(f"    Mean:   {real_scores.mean():.3f}")
    print(f"    Median: {np.median(real_scores):.3f}")
    print(f"    Std:    {real_scores.std():.3f}")

    print(f"\n  FAKE voices — P(fake) scores:")
    print(f"    Min:    {fake_scores.min():.3f}")
    print(f"    Max:    {fake_scores.max():.3f}")
    print(f"    Mean:   {fake_scores.mean():.3f}")
    print(f"    Median: {np.median(fake_scores):.3f}")
    print(f"    Std:    {fake_scores.std():.3f}")

    # Score separation
    separation = fake_scores.mean() - real_scores.mean()
    print(f"\n  Score separation (higher = better): {separation:.3f}")

    if separation < 0.1:
        print("  ⚠️  VERY LOW — model barely distinguishes real from fake")
    elif separation < 0.3:
        print("  ⚠️  LOW — model has weak discrimination ability")
    elif separation < 0.5:
        print("  ⚠  MODERATE — room for improvement")
    else:
        print("  ✓  GOOD separation")

    # ── Accuracy at different thresholds ──
    print(f"\n{'=' * 60}")
    print(f"  ACCURACY AT DIFFERENT THRESHOLDS")
    print(f"{'=' * 60}")
    print(f"  {'Thresh':>8}  {'Real Acc':>10}  {'Fake Acc':>10}  "
          f"{'Overall':>10}  {'Balanced':>10}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")

    best_bal_acc = 0
    best_thresh = 0.5

    for t in np.arange(0.1, 0.95, 0.05):
        real_acc = (real_scores < t).mean() * 100
        fake_acc = (fake_scores >= t).mean() * 100
        overall = ((real_scores < t).sum() + (fake_scores >= t).sum()) / \
                  (len(real_scores) + len(fake_scores)) * 100
        balanced = (real_acc + fake_acc) / 2

        marker = "  ◀── current" if abs(t - threshold) < 0.03 else ""
        if balanced > best_bal_acc:
            best_bal_acc = balanced
            best_thresh = t

        print(f"  {t:>8.2f}  {real_acc:>9.1f}%  {fake_acc:>9.1f}%  "
              f"{overall:>9.1f}%  {balanced:>9.1f}%{marker}")

    print(f"\n  ★ Best threshold: {best_thresh:.2f} → "
          f"balanced acc = {best_bal_acc:.1f}%")

    # ── Histogram ──
    print(f"\n{'=' * 60}")
    print(f"  SCORE HISTOGRAM (visual)")
    print(f"{'=' * 60}")

    bins = np.arange(0, 1.05, 0.1)
    print(f"\n  REAL voices P(fake):")
    hist_r, _ = np.histogram(real_scores, bins=bins)
    for i, count in enumerate(hist_r):
        bar = "█" * (count // max(1, max(hist_r) // 30))
        print(f"    {bins[i]:.1f}-{bins[i+1]:.1f}: {bar} ({count})")

    print(f"\n  FAKE voices P(fake):")
    hist_f, _ = np.histogram(fake_scores, bins=bins)
    for i, count in enumerate(hist_f):
        bar = "█" * (count // max(1, max(hist_f) // 30))
        print(f"    {bins[i]:.1f}-{bins[i+1]:.1f}: {bar} ({count})")

    # ── Recommendations ──
    print(f"\n{'=' * 60}")
    print(f"  RECOMMENDATIONS")
    print(f"{'=' * 60}")

    if separation < 0.3:
        print("""
  Your model has WEAK discrimination. Main fixes needed:

  1. USE FULL DATASET — You only used 2580+2580 = 5160 samples.
     ASVspoof has 25,380 training files. Use ALL of them with
     balanced sampling (not subsampling).

  2. STRONGER AUGMENTATION — Add noise, pitch shift, speed change,
     room reverb simulation.

  3. BETTER FEATURES — Use raw waveform + Mel spectrogram together.

  4. LONGER TRAINING — Train for 50-80 epochs with cosine LR.

  Run: python improve_model.py
        """)
    else:
        print("""
  Your model has decent discrimination. Fine-tuning needed:

  1. CALIBRATE THRESHOLD — Use the optimal threshold above.
  2. ENSEMBLE — Train 2-3 models and average predictions.
  3. MORE AUGMENTATION during training.

  Run: python improve_model.py
        """)


if __name__ == "__main__":
    main()