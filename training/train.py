"""
Full training pipeline — MPS (Apple M2) accelerated.
"""
import os, sys, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, roc_curve)
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from models.detector_model import get_model
from data.dataset import get_dataloaders


class Trainer:
    def __init__(self, model_type="resnet",
                 feature_type="mel_spectrogram",
                 resume=None, max_samples=None):
        self.device = Config.DEVICE
        print(f"Device: {self.device}")

        self.model = get_model(model_type).to(self.device)

        # Weighted loss  (dataset is imbalanced ≈ 9:1 fake:real)
        weights = torch.FloatTensor([1.0, 0.1]).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights)

        self.optimizer = optim.Adam(self.model.parameters(),
                                     lr=Config.LEARNING_RATE,
                                     weight_decay=Config.WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=Config.SCHEDULER_STEP,
            gamma=Config.SCHEDULER_GAMMA)

        self.train_loader, self.val_loader = get_dataloaders(
            feature_type, max_samples=max_samples)

        self.writer = SummaryWriter(Config.LOG_DIR)
        self.start_epoch = 0
        self.best_loss = float("inf")
        self.best_acc = 0.0
        self.patience = 0
        self.best_state = None

        if resume and os.path.exists(resume):
            self._load_ckpt(resume)

    # ── public ────────────────────────────────────────────
    def run(self):
        print(f"\n{'='*60}\nTraining for up to {Config.NUM_EPOCHS} epochs\n{'='*60}")

        for epoch in range(self.start_epoch, Config.NUM_EPOCHS):
            t0 = time.time()

            tr_loss, tr_acc = self._train_one(epoch)
            vl_loss, vl_acc, metrics = self._validate(epoch)
            self.scheduler.step()

            self.writer.add_scalars("Loss",
                {"train": tr_loss, "val": vl_loss}, epoch)
            self.writer.add_scalars("Acc",
                {"train": tr_acc, "val": vl_acc}, epoch)

            dt = time.time() - t0
            print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS} ({dt:.0f}s)")
            print(f"  Train  loss={tr_loss:.4f}  acc={tr_acc:.1f}%")
            print(f"  Val    loss={vl_loss:.4f}  acc={vl_acc:.1f}%")
            print(f"  F1={metrics['f1']:.4f}  P={metrics['prec']:.4f}"
                  f"  R={metrics['rec']:.4f}")

            if vl_loss < self.best_loss:
                self.best_loss = vl_loss
                self.best_acc  = vl_acc
                self.best_state = copy.deepcopy(self.model.state_dict())
                self.patience = 0
                self._save_ckpt(epoch, best=True)
                print(f"  ★ New best model saved")
            else:
                self.patience += 1
                print(f"  Patience {self.patience}/{Config.EARLY_STOP_PATIENCE}")

            if (epoch + 1) % 5 == 0:
                self._save_ckpt(epoch)

            if self.patience >= Config.EARLY_STOP_PATIENCE:
                print("Early stopping.")
                break

        self._save_final()
        self.writer.close()
        print(f"\nDone — best val loss {self.best_loss:.4f}, "
              f"acc {self.best_acc:.1f}%")

    # ── private ───────────────────────────────────────────
    def _train_one(self, epoch):
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        for feats, labels in tqdm(self.train_loader,
                                   desc=f"Epoch {epoch+1} [train]",
                                   leave=False):
            feats  = feats.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(feats)
            loss = self.criterion(out, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += out.argmax(1).eq(labels).sum().item()
            total += labels.size(0)

        return total_loss / total, 100.0 * correct / total

    @torch.no_grad()
    def _validate(self, epoch):
        self.model.eval()
        total_loss, ys, ps, probs = 0, [], [], []

        for feats, labels in tqdm(self.val_loader,
                                   desc=f"Epoch {epoch+1} [val]",
                                   leave=False):
            feats  = feats.to(self.device)
            labels = labels.to(self.device)

            out  = self.model(feats)
            loss = self.criterion(out, labels)
            total_loss += loss.item() * labels.size(0)

            prob = torch.softmax(out, 1)[:, 1]
            pred = out.argmax(1)

            ys.extend(labels.cpu().numpy())
            ps.extend(pred.cpu().numpy())
            probs.extend(prob.cpu().numpy())

        n    = len(ys)
        vloss = total_loss / n
        vacc  = 100.0 * accuracy_score(ys, ps)

        metrics = {
            "prec": precision_score(ys, ps, zero_division=0),
            "rec":  recall_score(ys, ps, zero_division=0),
            "f1":   f1_score(ys, ps, zero_division=0),
        }

        try:
            fpr, tpr, _ = roc_curve(ys, probs)
            fnr = 1 - tpr
            eer = (fpr[np.nanargmin(np.abs(fnr - fpr))]
                   + (1 - tpr)[np.nanargmin(np.abs(fnr - fpr))]) / 2
            metrics["eer"] = eer
            self.writer.add_scalar("EER", eer, epoch)
            print(f"  EER={eer:.4f}")
        except Exception:
            pass

        return vloss, vacc, metrics

    # ── checkpoint helpers ────────────────────────────────
    def _save_ckpt(self, epoch, best=False):
        d = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optim": self.optimizer.state_dict(),
            "sched": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
        }
        name = "best_model.pth" if best else f"ckpt_ep{epoch+1}.pth"
        torch.save(d, os.path.join(Config.CHECKPOINT_DIR, name))

    def _load_ckpt(self, path):
        print(f"Resuming from {path}")
        ck = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ck["model"])
        self.optimizer.load_state_dict(ck["optim"])
        self.scheduler.load_state_dict(ck["sched"])
        self.start_epoch = ck["epoch"] + 1
        self.best_loss = ck["best_loss"]

    def _save_final(self):
        if self.best_state is None:
            return
        path = os.path.join(Config.CHECKPOINT_DIR, "final_model.pth")
        torch.save({"model": self.best_state,
                     "val_loss": self.best_loss,
                     "val_acc": self.best_acc}, path)
        print(f"Final model → {path}")

        # Export ONNX
        try:
            self.model.load_state_dict(self.best_state)
            self.model.eval()
            dummy = torch.randn(1, 1, Config.N_MELS, 251).to(self.device)
            # MPS tensors must move to CPU for ONNX export
            self.model.cpu()
            dummy_cpu = dummy.cpu()
            onnx_path = os.path.join(Config.CHECKPOINT_DIR, "model.onnx")
            torch.onnx.export(self.model, dummy_cpu, onnx_path,
                              opset_version=14,
                              input_names=["features"],
                              output_names=["logits"],
                              dynamic_axes={"features": {0: "batch"}})
            print(f"ONNX model → {onnx_path}")
            self.model.to(self.device)
        except Exception as e:
            print(f"ONNX export failed: {e}")


if __name__ == "__main__":
    trainer = Trainer(
        model_type=Config.MODEL_TYPE,        # "resnet"
        feature_type="mel_spectrogram",
        max_samples=None,                    # set e.g. 2000 for quick test
    )
    trainer.run()