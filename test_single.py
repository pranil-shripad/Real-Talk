"""
Test the trained model on a single audio file.
Usage:
    python test_single.py /path/to/audio.wav
    python test_single.py /path/to/audio.flac
    python test_single.py /path/to/audio.mp3
"""
import sys
import os
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config
from features.feature_extractor import FeatureExtractor
from models.detector_model import get_model


class DeepfakeDetector:
    """Load trained model and run inference on audio."""

    def __init__(self, checkpoint_path=None):
        self.device = Config.DEVICE
        print(f"Device: {self.device}")

        # Load model
        self.model = get_model(Config.MODEL_TYPE).to(self.device)

        ckpt_path = checkpoint_path or os.path.join(
            Config.CHECKPOINT_DIR, "final_model_balanced.pth"
        )

        if not os.path.exists(ckpt_path):
            # Try final_model.pth as fallback
            ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "final_model.pth")

        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            print(f"✓ Loaded checkpoint: {ckpt_path}")

            # Print saved metrics if available
            if "val_loss" in checkpoint:
                print(f"  Val loss: {checkpoint['val_loss']:.4f}")
            if "val_acc" in checkpoint:
                print(f"  Val acc:  {checkpoint['val_acc']:.1f}%")
        else:
            print(f"✗ No checkpoint found at {ckpt_path}")
            print("  Make sure training completed successfully.")
            sys.exit(1)

        self.model.eval()
        self.extractor = FeatureExtractor("mel_spectrogram")

    @torch.no_grad()
    def predict_file(self, filepath):
        """
        Predict whether an audio file is REAL or FAKE.

        Returns:
            dict with label, confidence, probabilities
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        # Extract features
        features = self.extractor.from_file(filepath)
        features = features.unsqueeze(0).to(self.device)  # (1, C, H, W)

        # Run model
        logits = self.model(features)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        real_prob = float(probs[0])
        fake_prob = float(probs[1])
        is_fake = fake_prob > real_prob

        return {
            "label": "FAKE" if is_fake else "REAL",
            "confidence": max(real_prob, fake_prob) * 100,
            "real_probability": real_prob * 100,
            "fake_probability": fake_prob * 100,
        }

    @torch.no_grad()
    def predict_numpy(self, audio_np, sr=16000):
        """Predict from numpy array."""
        features = self.extractor.from_numpy(audio_np, sr)
        features = features.unsqueeze(0).to(self.device)

        logits = self.model(features)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        real_prob = float(probs[0])
        fake_prob = float(probs[1])

        return {
            "label": "FAKE" if fake_prob > real_prob else "REAL",
            "confidence": max(real_prob, fake_prob) * 100,
            "real_probability": real_prob * 100,
            "fake_probability": fake_prob * 100,
        }


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_single.py <audio_file>")
        print("  Supported: .wav  .flac  .mp3  .ogg")
        print("\nExample:")
        print("  python test_single.py ~/Downloads/sample_voice.wav")
        sys.exit(0)

    filepath = sys.argv[1]

    print("=" * 50)
    print("  DEEPFAKE AUDIO DETECTOR — Single File Test")
    print("=" * 50)

    detector = DeepfakeDetector()

    print(f"\nAnalysing: {filepath}")
    print("-" * 50)

    result = detector.predict_file(filepath)

    # Pretty output
    if result["label"] == "REAL":
        icon = "✅"
    else:
        icon = "⚠️"

    print(f"\n  Result:      {icon}  {result['label']}")
    print(f"  Confidence:  {result['confidence']:.1f}%")
    print(f"  P(Real):     {result['real_probability']:.1f}%")
    print(f"  P(Fake):     {result['fake_probability']:.1f}%")
    print("-" * 50)


if __name__ == "__main__":
    main()