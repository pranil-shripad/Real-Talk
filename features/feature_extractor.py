"""
Audio feature extraction — Mel spectrograms, MFCCs, deltas.
"""
import torch
import torchaudio
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class FeatureExtractor:
    def __init__(self, feature_type="mel_spectrogram"):
        self.feature_type = feature_type
        self.sr = Config.SAMPLE_RATE

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            win_length=Config.WIN_LENGTH,
            n_mels=Config.N_MELS,
            power=2.0,
        )
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sr,
            n_mfcc=Config.N_MFCC,
            melkwargs=dict(
                n_fft=Config.N_FFT,
                hop_length=Config.HOP_LENGTH,
                n_mels=Config.N_MELS,
            ),
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    # ── loaders ───────────────────────────────────────────
    def load_audio(self, filepath: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(filepath)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sr:
            waveform = torchaudio.transforms.Resample(sr, self.sr)(waveform)
        return self._fix_length(waveform, Config.AUDIO_LENGTH)

    def load_audio_numpy(self, audio_np, sr=None) -> torch.Tensor:
        sr = sr or self.sr
        waveform = torch.FloatTensor(audio_np)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if sr != self.sr:
            waveform = torchaudio.transforms.Resample(sr, self.sr)(waveform)
        return self._fix_length(waveform, Config.AUDIO_LENGTH)

    @staticmethod
    def _fix_length(waveform, target):
        cur = waveform.shape[1]
        if cur < target:
            waveform = torch.nn.functional.pad(waveform, (0, target - cur))
        elif cur > target:
            start = (cur - target) // 2
            waveform = waveform[:, start:start + target]
        return waveform

    # ── extractors ────────────────────────────────────────
    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.feature_type == "mel_spectrogram":
            return self._mel(waveform)
        elif self.feature_type == "mfcc":
            return self._mfcc(waveform)
        elif self.feature_type == "raw":
            return waveform
        raise ValueError(f"Unknown feature type: {self.feature_type}")

    def _mel(self, waveform):
        spec = self.amp_to_db(self.mel_transform(waveform))
        return (spec - spec.mean()) / (spec.std() + 1e-8)

    def _mfcc(self, waveform):
        mfcc = self.mfcc_transform(waveform)
        d1 = torchaudio.functional.compute_deltas(mfcc)
        d2 = torchaudio.functional.compute_deltas(d1)
        feat = torch.cat([mfcc, d1, d2], dim=0)
        return (feat - feat.mean()) / (feat.std() + 1e-8)

    # ── convenience ───────────────────────────────────────
    def from_file(self, filepath):
        return self.extract(self.load_audio(filepath))

    def from_numpy(self, audio_np, sr=None):
        return self.extract(self.load_audio_numpy(audio_np, sr))


if __name__ == "__main__":
    ext = FeatureExtractor("mel_spectrogram")
    dummy = torch.randn(1, Config.AUDIO_LENGTH)
    print("Mel shape:", ext.extract(dummy).shape)

    ext2 = FeatureExtractor("mfcc")
    print("MFCC shape:", ext2.extract(dummy).shape)