import os, sys, random, torch, numpy as np
sys.path.append(".")
from config import Config
from features.feature_extractor import FeatureExtractor
from models.detector_model import get_model

device = Config.DEVICE
model = get_model(Config.MODEL_TYPE).to(device)

# Find and load checkpoint
for name in ["best_model_balanced.pth", "best_model.pth", "final_model.pth"]:
    p = os.path.join(Config.CHECKPOINT_DIR, name)
    if os.path.exists(p):
        ck = torch.load(p, map_location=device)
        key = "model" if "model" in ck else "model_state_dict"
        model.load_state_dict(ck[key])
        print(f"Loaded: {name}")
        break

model.eval()
ext = FeatureExtractor("mel_spectrogram")

# Read protocol
with open(Config.ASVSPOOF_DEV_PROTOCOL) as f:
    lines = f.readlines()

real = [l.split()[1] for l in lines if "bonafide" in l]
fake = [l.split()[1] for l in lines if "spoof" in l]

random.seed(42)
print("\n── 10 REAL samples ──")
for name in random.sample(real, 10):
    path = os.path.join(Config.ASVSPOOF_DEV, name + ".flac")
    if not os.path.exists(path): continue
    feat = ext.from_file(path).unsqueeze(0).to(device)
    with torch.no_grad():
        p = torch.softmax(model(feat), 1).cpu().numpy()[0]
    tag = "✅ REAL" if p[0] > p[1] else "❌ FAKE"
    print(f"  {tag}  P(real)={p[0]:.3f}  P(fake)={p[1]:.3f}  {name}")

print("\n── 10 FAKE samples ──")
for name in random.sample(fake, 10):
    path = os.path.join(Config.ASVSPOOF_DEV, name + ".flac")
    if not os.path.exists(path): continue
    feat = ext.from_file(path).unsqueeze(0).to(device)
    with torch.no_grad():
        p = torch.softmax(model(feat), 1).cpu().numpy()[0]
    tag = "✅ FAKE" if p[1] > p[0] else "❌ REAL"
    print(f"  {tag}  P(real)={p[0]:.3f}  P(fake)={p[1]:.3f}  {name}")
