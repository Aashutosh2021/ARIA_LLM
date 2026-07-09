"""
AIRA-LLM
Trainer

Orchestrates the training / validation loop: forward, loss, backward,
gradient clipping, optimizer + (per-step) scheduler, logging, periodic
and best-model checkpointing.
"""

# pyrefly: ignore [missing-import]
import torch
from tqdm import tqdm

from training.checkpoint import save_checkpoint
from utils.metrics import AverageMeter, perplexity


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        scheduler=None,
        logger=None,
        config=None,
        grad_clip: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        checkpoint_every: int = 0,
        log_every: int = 10,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.logger = logger
        self.config = config or {}
        self.grad_clip = grad_clip
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every
        self.log_every = max(1, log_every)

        self.step = 0
        self.best_val_loss = float("inf")

    # ------------------------------------------------------------------
    def _log(self, message: str):
        if self.logger is not None:
            self.logger.info(message)
        else:
            print(message, flush=True)

    def _current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    # ------------------------------------------------------------------
    def train(self, train_loader, val_loader=None, epochs: int = 1):

        self._log(
            f"Starting training: {epochs} epochs, "
            f"{len(train_loader)} batches/epoch, device={self.device}"
        )

        for epoch in range(1, epochs + 1):

            train_loss = self._train_epoch(train_loader, epoch, epochs)

            message = (
                f"[epoch {epoch}/{epochs}] "
                f"train_loss={train_loss:.4f} "
                f"train_ppl={perplexity(train_loss):.2f}"
            )

            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                message += (
                    f" | val_loss={val_loss:.4f} "
                    f"val_ppl={perplexity(val_loss):.2f}"
                )

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save("best.pt", epoch)
                    message += "  <- new best"

            self._log(message)

            if self.logger is not None:
                self.logger.log(
                    epoch=epoch,
                    step=self.step,
                    train_loss=train_loss,
                    lr=self._current_lr(),
                )

        # Always keep a final checkpoint.
        self._save("last.pt", epochs)

    # ------------------------------------------------------------------
    def _train_epoch(self, train_loader, epoch, epochs) -> float:

        self.model.train()
        meter = AverageMeter()

        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch}/{epochs}",
            leave=False,
        )

        for inputs, targets in progress:

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            logits, _ = self.model(inputs)

            loss = self.criterion(logits, targets)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if self.grad_clip and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip,
                )

            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            self.step += 1
            meter.update(loss, n=inputs.size(0))

            if self.step % self.log_every == 0:
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{self._current_lr():.2e}",
                )

            if (
                self.checkpoint_every
                and self.step % self.checkpoint_every == 0
            ):
                self._save(f"step_{self.step}.pt", epoch)

        return meter.average

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, val_loader) -> float:

        self.model.eval()
        meter = AverageMeter()

        for inputs, targets in val_loader:

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            logits, _ = self.model(inputs)
            loss = self.criterion(logits, targets)

            meter.update(loss, n=inputs.size(0))

        return meter.average

    # ------------------------------------------------------------------
    def _save(self, filename: str, epoch: int):

        path = f"{self.checkpoint_dir}/{filename}"

        # Under DataParallel the real model lives at model.module; save that so
        # the checkpoint keys have no "module." prefix and load cleanly for
        # single-GPU / CPU inference.
        model = getattr(self.model, "module", self.model)

        save_checkpoint(
            path=path,
            model=model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.step,
            epoch=epoch,
            config=self.config,
        )
