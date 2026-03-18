import os
import torch
import torch.distributed as dist


class CoCaTrainer:

    def __init__(
        self,
        model,
        optimizer,
        accelerator,
        max_epochs,
        pad_token_id=0,
        save_dir=None,
        save_name="coca",
        save_best_only=True,
        scheduler=None,
        freeze_language=True,
    ):

        # Store core training components
        self.model = model                    
        self.optimizer = optimizer            # Optimizer
        self.accelerator = accelerator        # For distributed/multi-device setups
        self.max_epochs = max_epochs
        self.pad_token_id = pad_token_id      # Padding token ID for text
        self.save_dir = save_dir              # Directory to save checkpoints
        self.save_name = save_name            # Filename for checkpoints
        self.save_best_only = save_best_only  
        self.best_loss = float("inf")         
        self.scheduler = scheduler            # Optional learning rate scheduler



        # If model is wrapped with DataParallel or DistributedDataParallel,
        # the real model is inside model.module
        if hasattr(self.model, "module"):
            model_ref = self.model.module
        else:
            model_ref = self.model

        # Optional: Freeze language encoder
        if freeze_language:
            lang_enc = model_ref.language_enc
            lang_enc.eval()

            # Disable gradient updates for language encoder parameters
            for p in lang_enc.parameters():
                p.requires_grad = False

    def _get_model_ref(self):
        if hasattr(self.model, "module"):
            return self.model.module
        return self.model

    def _get_device(self):
        return next(self._get_model_ref().parameters()).device

    def _prepare_batch(self, batch, device):
        if isinstance(batch, dict):
            x_ts = batch["ecg"]
            input_ids = batch["input_ids"]
            attn_mask = batch["attention_mask"]
            decoder_input_ids = batch.get("decoder_input_ids")
            decoder_attn_mask = batch.get("decoder_attention_mask")
            clinical_labels = batch.get("labels")
        else:
            if len(batch) < 3:
                raise ValueError("Batch must contain at least (x_ts, input_ids, attention_mask)")
            x_ts, input_ids, attn_mask = batch[:3]
            decoder_input_ids = None
            decoder_attn_mask = None
            clinical_labels = None

        x_ts = x_ts.to(device)
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)

        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids.to(device)
        if decoder_attn_mask is not None:
            decoder_attn_mask = decoder_attn_mask.to(device)
        if clinical_labels is not None:
            clinical_labels = clinical_labels.to(device)

        caption_ids = decoder_input_ids if decoder_input_ids is not None else input_ids
        caption_mask = decoder_attn_mask if decoder_attn_mask is not None else attn_mask

        labels = caption_ids.clone()
        labels = labels.masked_fill(caption_mask == 0, -100)

        return x_ts, input_ids, attn_mask, labels, decoder_input_ids, decoder_attn_mask, clinical_labels


    def train_one_epoch(self, data_loader, epoch):

        self.model.train()
        model_ref = self._get_model_ref()

        if hasattr(model_ref, "language_enc"):
            model_ref.language_enc.eval()

        device = self._get_device()

        total_loss = 0.0

        for batch in data_loader:
            x_ts, input_ids, attn_mask, labels, decoder_input_ids, decoder_attn_mask, clinical_labels = self._prepare_batch(batch, device)

            # Forward pass
            loss = self.model(
                x_ts,
                input_ids,
                attn_mask,
                labels=labels,
                clinical_labels=clinical_labels,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attn_mask,
                return_loss=True,
            )
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)


        # Model Saving 
        if self.save_dir is not None:

            # In distributed training, only process with rank 0 saves model
            is_rank_zero = True
            if dist.is_available() and dist.is_initialized():
                is_rank_zero = dist.get_rank() == 0

            if is_rank_zero:

                # Create directory if it doesn't exist
                os.makedirs(self.save_dir, exist_ok=True)

                should_save = True

                # If saving only best model, check improvement
                if self.save_best_only:
                    should_save = avg_loss < self.best_loss

                if should_save:
                    # Update best loss
                    self.best_loss = min(self.best_loss, avg_loss)

                    # Extract correct state_dict depending on wrapper
                    if hasattr(self.model, "module"):
                        state = self.model.module.state_dict()
                    else:
                        state = self.model.state_dict()

                    # Define checkpoint path
                    ckpt_path = os.path.join(self.save_dir, f"{self.save_name}.pt")

                    # Save model + optimizer state for full reproducibility
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": state,
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "loss": avg_loss,
                        },
                        ckpt_path,
                    )

        return avg_loss

    def evaluate(self, data_loader):
        self.model.eval()
        model_ref = self._get_model_ref()
        if hasattr(model_ref, "language_enc"):
            model_ref.language_enc.eval()

        device = self._get_device()
        total_loss = 0.0

        with torch.no_grad():
            for batch in data_loader:
                x_ts, input_ids, attn_mask, labels, decoder_input_ids, decoder_attn_mask, clinical_labels = self._prepare_batch(batch, device)
                loss = self.model(
                    x_ts,
                    input_ids,
                    attn_mask,
                    labels=labels,
                    clinical_labels=clinical_labels,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attn_mask,
                    return_loss=True,
                )
                total_loss += loss.item()

        if len(data_loader) == 0:
            return float("inf")

        return total_loss / len(data_loader)
