import torch
import torch.distributed as dist


class CoCaTrainer:

    def __init__(
        self,
        model,
        optimizer,
        max_epochs,
        pad_token_id=0,
        scheduler=None,
        freeze_language=True,
        unfreeze_language_layers=0,
        grad_clip_norm=1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.pad_token_id = pad_token_id
        self.scheduler = scheduler
        self.grad_clip_norm = grad_clip_norm

        model_ref = self._get_model_ref()

        # Freeze language encoder, optionally unfreeze top N layers
        if freeze_language:
            lang_enc = model_ref.language_enc
            for p in lang_enc.parameters():
                p.requires_grad = False

            if unfreeze_language_layers > 0:
                encoder_layers = getattr(lang_enc.model, "encoder", None)
                if encoder_layers is not None:
                    layers = encoder_layers.layer
                    for layer in layers[-unfreeze_language_layers:]:
                        for p in layer.parameters():
                            p.requires_grad = True

            lang_enc.eval()

    def _get_model_ref(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _get_device(self):
        return next(self._get_model_ref().parameters()).device

    def _reduce_scalar(self, value, count):
        """Average a scalar across all distributed ranks."""
        if not (dist.is_available() and dist.is_initialized()):
            return value / count if count > 0 else 0.0

        device = self._get_device()
        stats = torch.tensor([value, float(count)], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total, n = stats[0].item(), stats[1].item()
        return total / n if n > 0 else 0.0

    def _prepare_batch(self, batch, device):
        if isinstance(batch, dict):
            x_ts = batch["ecg"]
            input_ids = batch["input_ids"]
            attn_mask = batch["attention_mask"]
            decoder_input_ids = batch.get("decoder_input_ids")
            decoder_attn_mask = batch.get("decoder_attention_mask")
        else:
            if len(batch) < 3:
                raise ValueError("Batch must contain at least (x_ts, input_ids, attention_mask)")
            x_ts, input_ids, attn_mask = batch[:3]
            decoder_input_ids = None
            decoder_attn_mask = None

        x_ts = x_ts.to(device)
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)

        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids.to(device)
        if decoder_attn_mask is not None:
            decoder_attn_mask = decoder_attn_mask.to(device)

        class_labels = batch.get("labels") if isinstance(batch, dict) else None
        if class_labels is not None:
            class_labels = class_labels.to(device)

        caption_ids = decoder_input_ids if decoder_input_ids is not None else input_ids
        caption_mask = decoder_attn_mask if decoder_attn_mask is not None else attn_mask

        labels = caption_ids.clone()
        labels = labels.masked_fill(caption_mask == 0, -100)

        return x_ts, input_ids, attn_mask, labels, decoder_input_ids, decoder_attn_mask, class_labels

    def train_one_epoch(self, data_loader, epoch):
        self.model.train()
        model_ref = self._get_model_ref()

        if hasattr(model_ref, "language_enc"):
            model_ref.language_enc.eval()

        device = self._get_device()

        sums = {
            "loss": 0.0, "caption": 0.0,
            "contrastive": 0.0, "dirichlet": 0.0,
        }
        total_uncertainty = 0.0
        n_samples = 0

        for batch_idx, batch in enumerate(data_loader):
            x_ts, input_ids, attn_mask, labels, dec_ids, dec_mask, class_labels = \
                self._prepare_batch(batch, device)

            self.optimizer.zero_grad()

            output = self.model(
                x_ts, input_ids, attn_mask,
                labels=labels,
                class_labels=class_labels,
                decoder_input_ids=dec_ids,
                decoder_attention_mask=dec_mask,
                return_loss=True,
                epoch=epoch,
            )

            if torch.isnan(output.loss) or torch.isinf(output.loss):
                raise RuntimeError(
                    f"Loss is {output.loss.item()} at epoch {epoch}, batch {batch_idx}. "
                    "Check for NaN in inputs or logit scale explosion."
                )

            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            sums["loss"] += output.loss.item()
            sums["caption"] += output.caption_loss.item()
            sums["contrastive"] += output.contrastive_loss.item()
            if output.dirichlet_loss is not None:
                sums["dirichlet"] += output.dirichlet_loss.item()
            if output.uncertainty is not None:
                total_uncertainty += output.uncertainty.mean(dim=-1).sum().item()
                n_samples += output.uncertainty.size(0)

        n = len(data_loader)
        if n == 0:
            return {"loss": float("inf"), "caption_loss": 0, "contrastive_loss": 0,
                    "dirichlet_loss": 0, "mean_uncertainty": 0}

        metrics = {
            "loss": self._reduce_scalar(sums["loss"], n),
            "caption_loss": self._reduce_scalar(sums["caption"], n),
            "contrastive_loss": self._reduce_scalar(sums["contrastive"], n),
            "dirichlet_loss": self._reduce_scalar(sums["dirichlet"], n),
        }
        if n_samples > 0:
            metrics["mean_uncertainty"] = total_uncertainty / n_samples
        return metrics

    def evaluate(self, data_loader):
        self.model.eval()
        model_ref = self._get_model_ref()
        if hasattr(model_ref, "language_enc"):
            model_ref.language_enc.eval()

        device = self._get_device()
        sums = {
            "loss": 0.0, "caption": 0.0,
            "contrastive": 0.0, "dirichlet": 0.0,
        }
        all_ts_proj = []
        all_text_proj = []
        all_probs = []
        all_labels = []
        all_uncertainty = []

        with torch.no_grad():
            for batch in data_loader:
                x_ts, input_ids, attn_mask, labels, dec_ids, dec_mask, class_labels = \
                    self._prepare_batch(batch, device)

                output = self.model(
                    x_ts, input_ids, attn_mask,
                    labels=labels,
                    class_labels=class_labels,
                    decoder_input_ids=dec_ids,
                    decoder_attention_mask=dec_mask,
                    return_loss=True,
                )

                sums["loss"] += output.loss.item()
                sums["caption"] += output.caption_loss.item()
                sums["contrastive"] += output.contrastive_loss.item()
                if output.dirichlet_loss is not None:
                    sums["dirichlet"] += output.dirichlet_loss.item()

                all_ts_proj.append(output.ts_proj)
                all_text_proj.append(output.text_proj)
                if output.disease_probs is not None:
                    all_probs.append(output.disease_probs)
                if output.uncertainty is not None:
                    all_uncertainty.append(output.uncertainty)
                if class_labels is not None:
                    all_labels.append(class_labels)

        n = len(data_loader)
        if n == 0:
            return {"loss": float("inf")}

        metrics = {
            "loss": self._reduce_scalar(sums["loss"], n),
            "caption_loss": self._reduce_scalar(sums["caption"], n),
            "contrastive_loss": self._reduce_scalar(sums["contrastive"], n),
            "dirichlet_loss": self._reduce_scalar(sums["dirichlet"], n),
        }

        # Contrastive retrieval R@K
        all_ts_proj = torch.cat(all_ts_proj, dim=0)
        all_text_proj = torch.cat(all_text_proj, dim=0)
        metrics.update(self._retrieval_at_k(all_ts_proj, all_text_proj))

        # Uncertainty
        if all_uncertainty:
            all_uncertainty = torch.cat(all_uncertainty, dim=0)
            metrics["mean_uncertainty"] = all_uncertainty.mean().item()

        # Classification metrics (multi-label)
        if all_probs and all_labels:
            probs = torch.cat(all_probs, dim=0)
            labels_t = torch.cat(all_labels, dim=0)
            metrics.update(self._classification_metrics(probs, labels_t))

        return metrics

    @staticmethod
    def _retrieval_at_k(ts_proj, text_proj, ks=(1, 5, 10)):
        """Compute ECG->Text and Text->ECG recall@K."""
        sim = torch.mm(ts_proj, text_proj.T)  # (N, N)
        n = sim.size(0)
        targets = torch.arange(n, device=sim.device)

        metrics = {}
        for k in ks:
            if k > n:
                continue
            _, topk = sim.topk(k, dim=1)
            ecg2text = (topk == targets.unsqueeze(1)).any(dim=1).float().mean().item()
            _, topk_t = sim.T.topk(k, dim=1)
            text2ecg = (topk_t == targets.unsqueeze(1)).any(dim=1).float().mean().item()
            metrics[f"ecg2text_R@{k}"] = ecg2text
            metrics[f"text2ecg_R@{k}"] = text2ecg

        return metrics

    @staticmethod
    def _classification_metrics(probs, labels, threshold=0.5):
        """Compute multi-label classification metrics."""
        preds = (probs >= threshold).float()
        correct = (preds == labels).float()
        metrics = {
            "classif_accuracy": correct.mean().item(),
        }
        tp = (preds * labels).sum(dim=0)
        fp = (preds * (1 - labels)).sum(dim=0)
        fn = ((1 - preds) * labels).sum(dim=0)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        valid = (tp + fn) > 0
        if valid.any():
            metrics["macro_f1"] = f1[valid].mean().item()
        else:
            metrics["macro_f1"] = 0.0
        return metrics
