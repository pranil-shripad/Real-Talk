"""
Central configuration — Apple Silicon (M2) aware.
"""
import os
import torch


class Config:
    # ── PATHS ──────────────────────────────────────────────
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "raw")
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    LOG_DIR       = os.path.join(PROJECT_ROOT, "logs")
    SPEAKER_DB_DIR = os.path.join(PROJECT_ROOT, "speaker_db")

    # ASVspoof 2019 LA (update after downloading)
    ASVSPOOF_ROOT = os.path.join(DATA_DIR, "LA")
    ASVSPOOF_TRAIN = os.path.join(ASVSPOOF_ROOT,
                                   "ASVspoof2019_LA_train", "flac")
    ASVSPOOF_DEV   = os.path.join(ASVSPOOF_ROOT,
                                   "ASVspoof2019_LA_dev", "flac")
    ASVSPOOF_EVAL  = os.path.join(ASVSPOOF_ROOT,
                                   "ASVspoof2019_LA_eval", "flac")
    ASVSPOOF_TRAIN_PROTOCOL = os.path.join(
        ASVSPOOF_ROOT, "ASVspoof2019_LA_cm_protocols",
        "ASVspoof2019.LA.cm.train.trn.txt")
    ASVSPOOF_DEV_PROTOCOL = os.path.join(
        ASVSPOOF_ROOT, "ASVspoof2019_LA_cm_protocols",
        "ASVspoof2019.LA.cm.dev.trl.txt")
    ASVSPOOF_EVAL_PROTOCOL = os.path.join(
        ASVSPOOF_ROOT, "ASVspoof2019_LA_cm_protocols",
        "ASVspoof2019.LA.cm.eval.trl.txt")

    # ── AUDIO ──────────────────────────────────────────────
    SAMPLE_RATE   = 16_000
    AUDIO_DURATION = 4                       # seconds
    AUDIO_LENGTH  = SAMPLE_RATE * AUDIO_DURATION  # 64 000 samples
    N_MFCC       = 40
    N_MELS       = 128
    N_FFT        = 1024
    HOP_LENGTH   = 256
    WIN_LENGTH   = 1024

    # ── MODEL ──────────────────────────────────────────────
    NUM_CLASSES = 2        # bonafide / spoof
    MODEL_TYPE  = "resnet" # "light_cnn", "resnet", "se_resnet"

    # ── TRAINING ───────────────────────────────────────────
    BATCH_SIZE     = 32
    NUM_EPOCHS     = 50
    LEARNING_RATE  = 1e-4
    WEIGHT_DECAY   = 1e-4
    SCHEDULER_STEP = 10
    SCHEDULER_GAMMA = 0.5
    EARLY_STOP_PATIENCE = 7

    # macOS fork-safety: keep NUM_WORKERS = 0 on Mac
    # (multiprocessing + MPS can cause issues)
    NUM_WORKERS = 0

    # ── DEVICE (Apple Silicon) ─────────────────────────────
    # Priority: MPS (Metal GPU) → CPU
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")

    # ── REAL-TIME ──────────────────────────────────────────
    CHUNK_DURATION = 3       # seconds per analysis window
    CHUNK_OVERLAP  = 1       # seconds overlap
    CONFIDENCE_THRESHOLD = 0.5
    BUFFER_SIZE = SAMPLE_RATE * CHUNK_DURATION

    # BlackHole virtual device name (what macOS shows)
    LOOPBACK_DEVICE_NAME = "BlackHole 2ch"

    # ── GUI ────────────────────────────────────────────────
    WINDOW_WIDTH     = 520
    WINDOW_HEIGHT    = 450
    UPDATE_INTERVAL_MS = 100

    @classmethod
    def create_dirs(cls):
        for d in [cls.DATA_DIR, cls.PROCESSED_DIR, cls.CHECKPOINT_DIR,
                  cls.LOG_DIR, cls.SPEAKER_DB_DIR]:
            os.makedirs(d, exist_ok=True)


Config.create_dirs()