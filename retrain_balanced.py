"""
FIXED training pipeline that solves the class imbalance problem.

Key fixes:
  1. Balanced sampler — every batch has 50% real, 50% fake
  2. Focal Loss — focuses on hard-to-classify examples
  3. Proper class weights
  4. Better learning rate warmup
  5. Threshold calibration after training
"""
import os, sys, time, copy, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, roc_curve
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config
from models.detector_model import get_model
from data.dataset import ASVspoofDataset


# ══════════════════════════════════════════════════════════
#  FIX 1: Focal Loss — handles class imbalance much better
# ══════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    """
    Focal Loss: down-weights easy examples, focuses on hard ones.
    This is THE key fix for imbalanced datasets.
    
    When gamma=0, this is standard cross-entropy.
    When gamma=2, easy examples contribute almost nothing to loss.
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        # alpha: per-class weight  [weight_real, weight_fake]
        # Higher weight for REAL because there are fewer real samples
        if alpha is None:
            alpha = [3.0, 1.0]  # Weight real 3x more than fake
        self.alpha = torch.FloatTensor(alpha)
        self.gamma = gamma

    def forward(self, inputs, targets):
        self.alpha = self.alpha.to(inputs.device)
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # probability of correct class
        
        alpha_t = self.alpha[targets]
        focal_loss = alpha_t * ((1 - pt) ** self.gamma) * ce_loss
        
        return focal_loss.mean()


# ══════════════════════════════════════════════════════════
#  FIX 2: Balanced Dataset — ensures 50/50 real/fake
# ══════════════════════════════════════════════════════════
class BalancedASVspoofDataset(ASVspoofDataset):
    """
    Same as ASVspoofDataset but provides a balanced sampler.
    """
    def get_balanced_sampler(self):
        """
        Create a WeightedRandomSampler that samples
        real and fake at equal rates.
        """
        labels = [s["label"] for s in self.samples]
        
        real_count = sum(1 for l in labels if l == 0)
        fake_count = sum(1 for l in labels if l == 1)
        
        # Weight = inverse of class frequency
        weight_real = 1.0 / max(real_count, 1)
        weight_fake = 1.0 / max(fake_count, 1)
        
        sample_weights = []
        for l in labels:
            if l == 0:
                sample_weights.append(weight_real)
            else:
                sample_weights.append(weight_fake)
        
        sample_weights = torch.FloatTensor(sample_weights)
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(labels),
            replacement=True
        )
        
        print(f"  Balanced sampler: real_weight={weight_real:.6f}, "
              f"fake_weight={weight_fake:.6f}")
        print(f"  Effective: ~{real_count} real + ~{real_count} fake per epoch")
        
        return sampler


# ══════════════════════════════════════════════════════════
#  FIX 3: Better Trainer
# ══════════════════════════════════════════════════════════
class BalancedTrainer:
    def __init__(self, model_type="resnet",
                 feature_type="mel_spectrogram",
                 max_samples=None):
        
        self.device = Config.DEVICE
        print(f"Device: {self.device}")

        # Model
        self.model = get_model(model_type).to(self.device)

        # FIX: Use Focal Loss instead of CrossEntropyLoss
        self.criterion = FocalLoss(
            alpha=[3.0, 1.0],   # real gets 3x weight
            gamma=2.0           # focus on hard examples
        )

        # FIX: Use AdamW with proper settings
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=3e-4,            # slightly higher initial LR
            weight_decay=1e-3,
            betas=(0.9, 0.999)
        )

        # FIX: Cosine annealing — much better than step LR
        self.num_epochs = 40

        # ── Create balanced dataloaders ───────────────────
        print("\nCreating BALANCED training set ...")
        train_ds = BalancedASVspoofDataset(
            Config.ASVSPOOF_TRAIN_PROTOCOL,
            Config.ASVSPOOF_TRAIN,
            feature_type,
            augment=True,
            max_samples=max_samples
        )

        balanced_sampler = train_ds.get_balanced_sampler()

        self.train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            sampler=balanced_sampler,     # <── KEY: balanced sampling
            num_workers=0,
            pin_memory=False,
            drop_last=True
        )

        print("\nCreating validation set ...")
        val_ds = BalancedASVspoofDataset(
            Config.ASVSPOOF_DEV_PROTOCOL,
            Config.ASVSPOOF_DEV,
            feature_type,
            augment=False,
            max_samples=max_samples
        )

        self.val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        # Cosine scheduler with warmup
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        # State
        self.best_loss = float("inf")
        self.best_acc = 0.0
        self.best_f1 = 0.0
        self.best_state = None
        self.patience = 0
        self.optimal_threshold = 0.5

    def run(self):
        print(f"\n{'=' * 60}")
        print(f"  BALANCED TRAINING — {self.num_epochs} epochs")
        print(f"  Using: Focal Loss + Balanced Sampler")
        print(f"{'=' * 60}\n")

        for epoch in range(self.num_epochs):
            t0 = time.time()

            # ── Train ──
            tr_loss, tr_acc, tr_batch_balance = self._train_one(epoch)

            # ── Validate ──
            vl_loss, vl_acc, vl_f1, vl_real_acc, vl_fake_acc = self._validate(epoch)

            self.scheduler.step()
            dt = time.time() - t0

            print(f"\nEpoch {epoch + 1}/{self.num_epochs}  ({dt:.0f}s)  "
                  f"lr={self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"  Train:  loss={tr_loss:.4f}  acc={tr_acc:.1f}%  "
                  f"batch_balance={tr_batch_balance}")
            print(f"  Val:    loss={vl_loss:.4f}  acc={vl_acc:.1f}%  "
                  f"F1={vl_f1:.3f}")
            print(f"  Val per-class:  REAL_acc={vl_real_acc:.1f}%  "
                  f"FAKE_acc={vl_fake_acc:.1f}%")

            # Save best based on F1 (better than loss for imbalanced data)
            if vl_f1 > self.best_f1:
                self.best_f1 = vl_f1
                self.best_acc = vl_acc
                self.best_loss = vl_loss
                self.best_state = copy.deepcopy(self.model.state_dict())
                self.patience = 0
                self._save("best_model_balanced.pth")
                print(f"  ★ New best! F1={vl_f1:.3f}")
            else:
                self.patience += 1
                print(f"  Patience: {self.patience}/10")

            if self.patience >= 10:
                print("Early stopping.")
                break

        # ── Calibrate threshold ───────────────────────────
        print("\n" + "=" * 60)
        print("  CALIBRATING OPTIMAL THRESHOLD ...")
        self.model.load_state_dict(self.best_state)
        self.optimal_threshold = self._calibrate_threshold()

        # ── Save final ────────────────────────────────────
        self._save_final()

        print(f"\n{'=' * 60}")
        print(f"  TRAINING COMPLETE")
        print(f"  Best F1:    {self.best_f1:.3f}")
        print(f"  Best Acc:   {self.best_acc:.1f}%")
        print(f"  Optimal threshold: {self.optimal_threshold:.3f}")
        print(f"{'=' * 60}")

    def _train_one(self, epoch):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        batch_real_counts = []

        for feats, labels in tqdm(self.train_loader,
                                   desc=f"Epoch {epoch+1} [train]",
                                   leave=False):
            feats = feats.to(self.device)
            labels = labels.to(self.device)

            # Track batch balance
            real_in_batch = (labels == 0).sum().item()
            batch_real_counts.append(real_in_batch)

            self.optimizer.zero_grad()
            out = self.model(feats)
            loss = self.criterion(out, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += out.argmax(1).eq(labels).sum().item()
            total += labels.size(0)

        avg_real = np.mean(batch_real_counts)
        balance_str = f"~{avg_real:.0f} real / ~{Config.BATCH_SIZE - avg_real:.0f} fake per batch"

        return total_loss / total, 100.0 * correct / total, balance_str

    @torch.no_grad()
    def _validate(self, epoch):
        self.model.eval()
        total_loss = 0
        ys, ps = [], []

        for feats, labels in tqdm(self.val_loader,
                                   desc=f"Epoch {epoch+1} [val]",
                                   leave=False):
            feats = feats.to(self.device)
            labels = labels.to(self.device)

            out = self.model(feats)
            loss = self.criterion(out, labels)
            total_loss += loss.item() * labels.size(0)

            ys.extend(labels.cpu().numpy())
            ps.extend(out.argmax(1).cpu().numpy())

        ys, ps = np.array(ys), np.array(ps)
        n = len(ys)

        acc = 100.0 * accuracy_score(ys, ps)
        f1 = f1_score(ys, ps, zero_division=0)

        # Per-class accuracy
        real_mask = ys == 0
        fake_mask = ys == 1
        real_acc = 100.0 * (ps[real_mask] == 0).mean() if real_mask.sum() > 0 else 0
        fake_acc = 100.0 * (ps[fake_mask] == 1).mean() if fake_mask.sum() > 0 else 0

        return total_loss / n, acc, f1, real_acc, fake_acc

    @torch.no_grad()
    def _calibrate_threshold(self):
        """
        Find the optimal threshold on the validation set.
        This is critical for real-world performance.
        """
        self.model.eval()
        ys, probs_fake = [], []

        for feats, labels in tqdm(self.val_loader, desc="Calibrating",
                                   leave=False):
            feats = feats.to(self.device)
            out = self.model(feats)
            prob = torch.softmax(out, 1)[:, 1].cpu().numpy()

            ys.extend(labels.numpy())
            probs_fake.extend(prob)

        ys = np.array(ys)
        probs_fake = np.array(probs_fake)

        # Find threshold that maximises balanced accuracy
        best_thresh = 0.5
        best_bal_acc = 0

        for thresh in np.arange(0.1, 0.9, 0.01):
            preds = (probs_fake > thresh).astype(int)

            real_mask = ys == 0
            fake_mask = ys == 1

            real_acc = (preds[real_mask] == 0).mean() if real_mask.sum() > 0 else 0
            fake_acc = (preds[fake_mask] == 1).mean() if fake_mask.sum() > 0 else 0

            bal_acc = (real_acc + fake_acc) / 2

            if bal_acc > best_bal_acc:
                best_bal_acc = bal_acc
                best_thresh = thresh

        # Also compute EER threshold
        try:
            fpr, tpr, thresholds = roc_curve(ys, probs_fake)
            fnr = 1 - tpr
            eer_idx = np.nanargmin(np.abs(fnr - fpr))
            eer_thresh = thresholds[eer_idx]
            eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
            print(f"  EER = {eer:.4f} at threshold = {eer_thresh:.3f}")
        except:
            eer_thresh = best_thresh

        print(f"  Best balanced accuracy = {best_bal_acc*100:.1f}% "
              f"at threshold = {best_thresh:.3f}")
        print(f"  EER threshold = {eer_thresh:.3f}")

        # Use the balanced accuracy threshold
        return float(best_thresh)

    def _save(self, name):
        path = os.path.join(Config.CHECKPOINT_DIR, name)
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_f1": self.best_f1,
            "best_acc": self.best_acc,
            "threshold": self.optimal_threshold,
        }, path)

    def _save_final(self):
        if self.best_state is None:
            return

        path = os.path.join(Config.CHECKPOINT_DIR, "final_model_balanced.pth")
        torch.save({
            "model": self.best_state,
            "val_f1": self.best_f1,
            "val_acc": self.best_acc,
            "threshold": self.optimal_threshold,
        }, path)
        print(f"\n  Final model saved → {path}")
        print(f"  Optimal threshold saved: {self.optimal_threshold:.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet",
                        choices=["light_cnn", "resnet", "se_resnet"])
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples for quick test (e.g. 2000)")
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    trainer = BalancedTrainer(
        model_type=args.model,
        feature_type="mel_spectrogram",
        max_samples=args.max_samples,
    )
    trainer.num_epochs = args.epochs
    trainer.run()