class ClipTrainer:

    def __init__(self, model, optimizer, contrastive_loss, accelerator, max_epochs, scheduler=None):

        self.model = model
        self.optimizer = optimizer
        self.contrastive_loss = contrastive_loss
        self.accelerator = accelerator
        self.max_epochs = max_epochs
        self.scheduler = scheduler

        # freeze language encoder
        if hasattr(self.model, "module"):
            lang_enc = self.model.module.language_enc
        else:
            lang_enc = self.model.language_enc

        lang_enc.eval()
        for p in lang_enc.parameters():
            p.requires_grad = False


    def train_one_epoch(self, data_loader, epoch):

        self.model.train()
        total_loss = 0

        for x_ts, input_ids, attn_mask in data_loader:

            if hasattr(self.model, "module"):
                device = next(self.model.module.parameters()).device
            else:
                device = next(self.model.parameters()).device

            x_ts = x_ts.to(device)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)

            out_ts, out_lang = self.model(x_ts, input_ids, attn_mask) #compute embeddings

            loss = self.contrastive_loss(out_ts, out_lang) #compute loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(data_loader)
