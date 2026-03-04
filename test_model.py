"""
Universal test script for your balanced-trained model.
Auto-detects checkpoint and provides 4 test modes:

  python test_model.py --mode file --path /path/to/audio.wav
  python test_model.py --mode mic
  python test_model.py --mode batch
  python test_model.py --mode live
"""
import os
import sys
import time
import random
import argparse
import numpy as np
import torch
import sounddevice as sd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config
from features.feature_extractor import FeatureExtractor
from models.detector_model import get_model


class Detector:
    """Load trained model and run predictions."""

    def __init__(self):
        self.device = Config.DEVICE
        print(f"Device: {self.device}")

        # Build model
        self.model = get_model(Config.MODEL_TYPE).to(self.device)
        self.extractor = FeatureExtractor("mel_spectrogram")

        # Auto-find checkpoint
        ckpt_path = self._find_checkpoint()
        if ckpt_path is None:
            print("✗ No checkpoint found in checkpoints/")
            print("  Make sure training completed successfully.")
            sys.exit(1)

        # Load checkpoint
        ck = torch.load(ckpt_path, map_location=self.device)

        # Handle different checkpoint formats
        if "model" in ck:
            self.model.load_state_dict(ck["model"])
        elif "model_state_dict" in ck:
            self.model.load_state_dict(ck["model_state_dict"])
        else:
            # Maybe the checkpoint IS the state dict directly
            self.model.load_state_dict(ck)

        self.model.eval()

        # Load threshold if saved, otherwise default 0.5
        self.threshold = ck.get("threshold", 0.5)

        print(f"✓ Loaded: {ckpt_path}")
        print(f"  Threshold: {self.threshold:.3f}")

        # Print any saved metrics
        for key in ["val_loss", "val_acc", "best_f1", "best_acc"]:
            if key in ck:
                print(f"  {key}: {ck[key]:.4f}")

    def _find_checkpoint(self):
        """Auto-detect the best checkpoint file."""
        candidates = [
            "best_model_balanced.pth",
            "final_model_balanced.pth",
            "best_model.pth",
            "final_model.pth",
        ]
        for name in candidates:
            path = os.path.join(Config.CHECKPOINT_DIR, name)
            if os.path.exists(path):
                return path

        # Try any .pth file
        if os.path.exists(Config.CHECKPOINT_DIR):
            for f in os.listdir(Config.CHECKPOINT_DIR):
                if f.endswith(".pth"):
                    return os.path.join(Config.CHECKPOINT_DIR, f)
        return None

    @torch.no_grad()
    def predict_file(self, filepath):
        """Predict on an audio file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Not found: {filepath}")

        feat = self.extractor.from_file(filepath)
        feat = feat.unsqueeze(0).to(self.device)

        logits = self.model(feat)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        real_prob = float(probs[0])
        fake_prob = float(probs[1])
        is_fake = fake_prob > self.threshold

        return {
            "label": "FAKE" if is_fake else "REAL",
            "real_prob": real_prob * 100,
            "fake_prob": fake_prob * 100,
            "confidence": max(real_prob, fake_prob) * 100,
        }

    @torch.no_grad()
    def predict_numpy(self, audio_np, sr=16000):
        """Predict on a numpy array of audio."""
        feat = self.extractor.from_numpy(audio_np, sr)
        feat = feat.unsqueeze(0).to(self.device)

        logits = self.model(feat)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        real_prob = float(probs[0])
        fake_prob = float(probs[1])
        is_fake = fake_prob > self.threshold

        return {
            "label": "FAKE" if is_fake else "REAL",
            "real_prob": real_prob * 100,
            "fake_prob": fake_prob * 100,
            "confidence": max(real_prob, fake_prob) * 100,
        }


# ══════════════════════════════════════════════════════════
#  MODE 1: Test a single audio file
# ══════════════════════════════════════════════════════════
def test_file(detector, filepath):
    print(f"\n  Analysing: {filepath}")
    print("─" * 50)

    result = detector.predict_file(filepath)

    icon = "✅" if result["label"] == "REAL" else "⚠️"
    print(f"  Result:     {icon}  {result['label']}")
    print(f"  P(Real):    {result['real_prob']:.1f}%")
    print(f"  P(Fake):    {result['fake_prob']:.1f}%")
    print(f"  Confidence: {result['confidence']:.1f}%")
    print("─" * 50)


# ══════════════════════════════════════════════════════════
#  MODE 2: Record your voice from microphone and test
# ══════════════════════════════════════════════════════════
def test_mic(detector, duration=4, rounds=5):
    sr = Config.SAMPLE_RATE

    print(f"\n  Will record {rounds} rounds of {duration}s each")
    print(f"  Using microphone: {sd.query_devices(sd.default.device[0])['name']}")
    print()

    results = []

    for r in range(1, rounds + 1):
        print(f"── Round {r}/{rounds} ──")
        input("  Press ENTER to start recording ...")

        print(f"  🎤  Speak now! Recording {duration}s ", end="", flush=True)
        audio = sd.rec(
            int(duration * sr),
            samplerate=sr,
            channels=1,
            dtype="float32"
        )
        for i in range(duration):
            time.sleep(1)
            print(f"█", end="", flush=True)
        sd.wait()
        print("  Done!")

        audio = audio.flatten()

        # Check audio energy
        energy = float(np.abs(audio).mean())
        print(f"  Audio energy: {energy:.5f}", end="")

        if energy < 0.002:
            print("  ⚠️  Very quiet! Speak louder or check mic.")
            continue
        print("  ✓")

        # Predict
        result = detector.predict_numpy(audio, sr)
        results.append(result)

        icon = "✅" if result["label"] == "REAL" else "⚠️"
        print(f"\n  {icon}  {result['label']}  "
              f"(Real: {result['real_prob']:.1f}%  "
              f"Fake: {result['fake_prob']:.1f}%)\n")

    # Summary
    if results:
        print("═" * 50)
        print("  SUMMARY")
        print("═" * 50)
        for i, r in enumerate(results, 1):
            icon = "✅" if r["label"] == "REAL" else "⚠️"
            print(f"  Round {i}: {icon} {r['label']:4s}  "
                  f"Real={r['real_prob']:.1f}%  Fake={r['fake_prob']:.1f}%")

        real_count = sum(1 for r in results if r["label"] == "REAL")
        total = len(results)
        print(f"\n  Detected as REAL: {real_count}/{total}")

        if real_count == total:
            print("  ✅  Your real voice is correctly identified!")
        elif real_count >= total * 0.6:
            print("  ⚠️  Mostly correct, some misclassifications")
        else:
            print("  ❌  Model is still struggling — try threshold adjustment")


# ══════════════════════════════════════════════════════════
#  MODE 3: Batch test on ASVspoof dev set
# ══════════════════════════════════════════════════════════
def test_batch(detector, num_per_class=50):
    print(f"\n  Testing {num_per_class} real + {num_per_class} fake from dev set")

    protocol = Config.ASVSPOOF_DEV_PROTOCOL
    audio_dir = Config.ASVSPOOF_DEV

    if not os.path.exists(protocol):
        print(f"  ✗ Protocol not found: {protocol}")
        print("  Skipping batch test.")
        return

    # Parse protocol
    real_files = []
    fake_files = []
    with open(protocol) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                name = parts[1]
                label = parts[4]
                if label == "bonafide":
                    real_files.append(name)
                else:
                    fake_files.append(name)

    print(f"  Dev set: {len(real_files)} real, {len(fake_files)} fake")

    # Sample
    random.seed(42)
    test_real = random.sample(real_files, min(num_per_class, len(real_files)))
    test_fake = random.sample(fake_files, min(num_per_class, len(fake_files)))

    # Test REAL files
    print(f"\n  Testing REAL samples ...")
    real_correct = 0
    real_total = 0
    real_probs = []

    for name in test_real:
        path = os.path.join(audio_dir, name + ".flac")
        if not os.path.exists(path):
            path = os.path.join(audio_dir, name + ".wav")
        if not os.path.exists(path):
            continue
        try:
            result = detector.predict_file(path)
            real_probs.append(result["real_prob"])
            if result["label"] == "REAL":
                real_correct += 1
            real_total += 1
        except Exception as e:
            print(f"    Error: {name}: {e}")

    # Test FAKE files
    print(f"  Testing FAKE samples ...")
    fake_correct = 0
    fake_total = 0
    fake_probs = []

    for name in test_fake:
        path = os.path.join(audio_dir, name + ".flac")
        if not os.path.exists(path):
            path = os.path.join(audio_dir, name + ".wav")
        if not os.path.exists(path):
            continue
        try:
            result = detector.predict_file(path)
            fake_probs.append(result["fake_prob"])
            if result["label"] == "FAKE":
                fake_correct += 1
            fake_total += 1
        except Exception as e:
            print(f"    Error: {name}: {e}")

    # Results
    total_correct = real_correct + fake_correct
    total_tested = real_total + fake_total

    print(f"\n{'═' * 55}")
    print(f"  BATCH TEST RESULTS")
    print(f"{'═' * 55}")
    print(f"  REAL voices detected correctly: "
          f"{real_correct}/{real_total}  "
          f"({100*real_correct/max(real_total,1):.1f}%)")
    print(f"  FAKE voices detected correctly: "
          f"{fake_correct}/{fake_total}  "
          f"({100*fake_correct/max(fake_total,1):.1f}%)")
    print(f"  Overall accuracy:              "
          f"{total_correct}/{total_tested}  "
          f"({100*total_correct/max(total_tested,1):.1f}%)")

    if real_probs:
        print(f"\n  REAL samples — avg P(real): {np.mean(real_probs):.1f}%")
    if fake_probs:
        print(f"  FAKE samples — avg P(fake): {np.mean(fake_probs):.1f}%")

    print(f"{'═' * 55}")


# ══════════════════════════════════════════════════════════
#  MODE 4: Live continuous monitoring from mic
# ══════════════════════════════════════════════════════════
def test_live(detector, chunk_duration=3):
    sr = Config.SAMPLE_RATE
    chunk_samples = int(sr * chunk_duration)

    print(f"\n  🎤  Live monitoring — speak into your microphone")
    print(f"  Analysing every {chunk_duration} seconds")
    print(f"  Press Ctrl+C to stop\n")
    print("─" * 55)

    history = []

    try:
        while True:
            audio = sd.rec(chunk_samples, samplerate=sr,
                           channels=1, dtype="float32")
            sd.wait()
            audio = audio.flatten()

            energy = float(np.abs(audio).mean())

            if energy < 0.003:
                print(f"  {'░' * 30}  Silence")
                history.clear()
                continue

            result = detector.predict_numpy(audio, sr)

            # Smooth over last 3 predictions
            history.append(result["fake_prob"])
            if len(history) > 3:
                history.pop(0)
            avg_fake = np.mean(history)

            smoothed = "FAKE" if avg_fake > 50 else "REAL"

            # Visual bar
            bar_len = 30
            real_bars = int((100 - avg_fake) / 100 * bar_len)
            bar = f"{'█' * real_bars}{'░' * (bar_len - real_bars)}"

            icon = "✅" if smoothed == "REAL" else "⚠️"

            print(f"  {icon} {smoothed:4s}  [{bar}]  "
                  f"Real={100-avg_fake:.0f}%  Fake={avg_fake:.0f}%  "
                  f"energy={energy:.4f}")

    except KeyboardInterrupt:
        print(f"\n{'─' * 55}")
        print("  Stopped.")


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Test your trained deepfake detection model",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python test_model.py --mode file --path audio.wav
  python test_model.py --mode mic --rounds 5
  python test_model.py --mode batch --num 100
  python test_model.py --mode live
        """
    )
    parser.add_argument(
        "--mode", required=True,
        choices=["file", "mic", "batch", "live"],
        help="file  = test a single audio file\n"
             "mic   = record your voice and test\n"
             "batch = test on ASVspoof dev set\n"
             "live  = continuous microphone monitoring"
    )
    parser.add_argument("--path", type=str, default=None,
                        help="Audio file path (for --mode file)")
    parser.add_argument("--duration", type=int, default=4,
                        help="Recording duration in seconds (default: 4)")
    parser.add_argument("--rounds", type=int, default=5,
                        help="Number of mic recordings (default: 5)")
    parser.add_argument("--num", type=int, default=50,
                        help="Samples per class for batch test (default: 50)")

    args = parser.parse_args()

    print("═" * 55)
    print("  DEEPFAKE AUDIO DETECTOR — TEST")
    print("═" * 55)

    detector = Detector()

    if args.mode == "file":
        if args.path is None:
            args.path = input("  Enter audio file path: ").strip()
        test_file(detector, args.path)

    elif args.mode == "mic":
        test_mic(detector, duration=args.duration, rounds=args.rounds)

    elif args.mode == "batch":
        test_batch(detector, num_per_class=args.num)

    elif args.mode == "live":
        test_live(detector, chunk_duration=args.duration)


if __name__ == "__main__":
    main()