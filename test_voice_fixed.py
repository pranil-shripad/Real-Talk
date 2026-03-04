"""
Record your voice and test with the FIXED balanced model.
"""
import sys, os, time, argparse
import numpy as np
import sounddevice as sd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config
from test_fixed import FixedDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    print("=" * 55)
    print("  FIXED MODEL — Voice Test")
    print("=" * 55)

    detector = FixedDetector()
    sr = Config.SAMPLE_RATE
    results = []

    for r in range(1, args.rounds + 1):
        print(f"\n── Round {r}/{args.rounds} ──")
        input("  Press ENTER, then speak for "
              f"{args.duration} seconds ...")

        print(f"  🎤  Recording ...", end="", flush=True)
        audio = sd.rec(int(args.duration * sr),
                       samplerate=sr, channels=1,
                       dtype="float32")
        for i in range(args.duration):
            time.sleep(1)
            print(f" {i+1}", end="", flush=True)
        sd.wait()
        audio = audio.flatten()
        print("  Done!")

        energy = np.abs(audio).mean()
        if energy < 0.002:
            print("  ⚠️  Too quiet! Speak louder or check your mic.")
            continue

        result = detector.predict_numpy(audio, sr)
        results.append(result)

        icon = "✅" if result["label"] == "REAL" else "⚠️"
        print(f"\n  {icon}  {result['label']}  "
              f"(Real={result['real_probability']:.1f}%  "
              f"Fake={result['fake_probability']:.1f}%)")

    # Summary
    if results:
        print(f"\n{'=' * 55}")
        print("  SUMMARY")
        print(f"{'=' * 55}")
        for i, r in enumerate(results, 1):
            icon = "✅" if r["label"] == "REAL" else "⚠️"
            print(f"  Round {i}: {icon} {r['label']}  "
                  f"Real={r['real_probability']:.1f}%  "
                  f"Fake={r['fake_probability']:.1f}%")

        real_count = sum(1 for r in results if r["label"] == "REAL")
        print(f"\n  Detected REAL: {real_count}/{len(results)}")

        if real_count >= len(results) * 0.8:
            print("  ✅  Model correctly identifies your real voice!")
        elif real_count >= len(results) * 0.5:
            print("  ⚠️  Mixed — model still needs improvement")
        else:
            print("  ❌  Still detecting as fake — see troubleshooting below")
            print_troubleshooting()


def print_troubleshooting():
    print(f"""
  ── TROUBLESHOOTING ──────────────────────────────

  If STILL failing after balanced retrain:

  1. RECORDING QUALITY
     • Make sure room is quiet
     • Speak naturally, not too close to mic
     • Avoid background noise

  2. MANUAL THRESHOLD OVERRIDE
     • Edit test_fixed.py → set self.threshold = 0.7
       (higher = less likely to flag as fake)

  3. DOMAIN MISMATCH
     • ASVspoof was recorded differently than your mic
     • Try fine-tuning on YOUR OWN voice recordings

  4. QUICK FIX — run this:
     python adjust_threshold.py
    """)


if __name__ == "__main__":
    main()