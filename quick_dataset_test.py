"""Minimal one-shot test."""
import torch, sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config
from features.feature_extractor import FeatureExtractor
from models.detector_model import get_model

# Load
device = Config.DEVICE
model = get_model(Config.MODEL_TYPE).to(device)
ck = torch.load(os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"),
                map_location=device)
model.load_state_dict(ck["model"])
model.eval()
ext = FeatureExtractor("mel_spectrogram")

# Test on a file
filepath = sys.argv[1] if len(sys.argv) > 1 else input("Audio file path: ").strip()
feat = ext.from_file(filepath).unsqueeze(0).to(device)

with torch.no_grad():
    probs = torch.softmax(model(feat), 1).cpu().numpy()[0]

print(f"\n  REAL: {probs[0]*100:.1f}%")
print(f"  FAKE: {probs[1]*100:.1f}%")
print(f"  → {'✅ REAL' if probs[0] > probs[1] else '⚠️  FAKE'}")