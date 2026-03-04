"""
Continuous real-time detection from your microphone.
Shows live updates in the terminal.
Press Ctrl+C to stop.
"""
import sys, os, time
import numpy as np
import sounddevice as sd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config
from test_single import DeepfakeDetector


def main():
    print("=" * 55)
    print("  REAL-TIME DEEPFAKE DETECTOR — Microphone")
    print("=" * 55)

    detector = DeepfakeDetector()

    sr = Config.SAMPLE_RATE
    chunk_sec = Config.CHUNK_DURATION   # 3 seconds
    chunk_samples = int(sr * chunk_sec)

    print(f"\n  Sample rate:    {sr} Hz")
    print(f"  Chunk duration: {chunk_sec} sec")
    print(f"  Analysing every {chunk_sec} seconds ...")
    print(f"\n  🎤  Speak into your microphone!")
    print(f"  Press Ctrl+C to stop.\n")
    print(f"{'─' * 55}")

    # Smoothing history
    history = []
    history_size = 3

    try:
        while True:
            # Record a chunk
            audio = sd.rec(chunk_samples, samplerate=sr,
                           channels=1, dtype="float32")
            sd.wait()
            audio = audio.flatten()

            # Check energy
            energy = float(np.abs(audio).mean())

            if energy < 0.003:
                # Silence
                bar = "░" * 20
                print(f"  {bar}  Silence (energy={energy:.5f})")
                history.clear()
                continue

            # Run detection
            result = detector.predict_numpy(audio, sr)

            # Smoothing
            history.append(result["fake_probability"])
            if len(history) > history_size:
                history.pop(0)
            avg_fake = np.mean(history)

            smoothed_label = "FAKE" if avg_fake > 50 else "REAL"

            # Visual bar
            bar_len = 30
            real_bars = int((100 - avg_fake) / 100 * bar_len)
            fake_bars = bar_len - real_bars

            bar = f"{'█' * real_bars}{'░' * fake_bars}"

            if smoothed_label == "REAL":
                icon = "✅"
            else:
                icon = "⚠️"

            print(
                f"  {icon}  {smoothed_label:4s}  "
                f"[{bar}]  "
                f"Real={100 - avg_fake:.0f}%  "
                f"Fake={avg_fake:.0f}%  "
                f"(energy={energy:.4f})"
            )

    except KeyboardInterrupt:
        print(f"\n{'─' * 55}")
        print("  Stopped.")


if __name__ == "__main__":
    main()