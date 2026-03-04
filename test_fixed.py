"""
Test script that uses the BALANCED model + calibrated threshold.
"""
import sys, os, time
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config
from features.feature_extractor import FeatureExtractor
from models.detector_model import get_model


class FixedDetector:
    """Detector that uses the balanced model + calibrated threshold."""

    def __init__(self):
        self.device = Config.DEVICE
        self.model = get_model(Config.MODEL_TYPE).to(self.device)
        self.extractor = FeatureExtractor("mel_spectrogram")

        # Try balanced model first, then fall back
        ckpt_path = os.path.join(Config.CHECKPOINT_DIR,
                                  "best_model_balanced.pth")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(Config.CHECKPOINT_DIR,
                                      "final_model_balanced.pth")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(Config.CHECKPOINT_DIR,
                                      "best_model.pth")

        ck = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()

        # Load calibrated threshold
        self.threshold = ck.get("threshold", 0.5)
        print(f"✓ Loaded: {ckpt_path}")
        print(f"  Threshold: {self.threshold:.3f}")

    @torch.no_grad()
    def predict_file(self, filepath):
        feat = self.extractor.from_file(filepath).unsqueeze(0).to(self.device)
        logits = self.model(feat)
        probs = torch.softmax(logits, 1).cpu().numpy()[0]

        fake_prob = float(probs[1])
        # Use calibrated threshold instead of 0.5
        is_fake = fake_prob > self.threshold

        return {
            "label": "FAKE" if is_fake else "REAL",
            "confidence": float(fake_prob if is_fake else 1 - fake_prob) * 100,
            "real_probability": float(probs[0]) * 100,
            "fake_probability": float(probs[1]) * 100,
            "threshold_used": self.threshold,
        }

    @torch.no_grad()
    def predict_numpy(self, audio_np, sr=16000):
        feat = self.extractor.from_numpy(audio_np, sr).unsqueeze(0).to(self.device)
        logits = self.model(feat)
        probs = torch.softmax(logits, 1).cpu().numpy()[0]

        fake_prob = float(probs[1])
        is_fake = fake_prob > self.threshold

        return {
            "label": "FAKE" if is_fake else "REAL",
            "confidence": float(fake_prob if is_fake else 1 - fake_prob) * 100,
            "real_probability": float(probs[0]) * 100,
            "fake_probability": float(probs[1]) * 100,
        }


def test_single_file():
    """Test on a single file from command line."""
    if len(sys.argv) < 2:
        print("Usage: python test_fixed.py <audio_file>")
        return

    detector = FixedDetector()
    result = detector.predict_file(sys.argv[1])

    icon = "✅" if result["label"] == "REAL" else "⚠️"
    print(f"\n  {icon}  {result['label']}")
    print(f"  Confidence:  {result['confidence']:.1f}%")
    print(f"  P(Real):     {result['real_probability']:.1f}%")
    print(f"  P(Fake):     {result['fake_probability']:.1f}%")
    print(f"  Threshold:   {result['threshold_used']:.3f}")


if __name__ == "__main__":
    test_single_file()