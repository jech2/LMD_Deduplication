import math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from ..utils.general_utils import set_manual_seed

from ..utils.inference_utils import debug_inference
import random

class Embedding(nn.Module):
    def __init__(self, dim, num_vocab, max_len):
        super().__init__()

        self.token_embed = nn.Embedding(num_vocab, dim)
        self.pos_embed = FreqEmbedding(max_len, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        embed = self.token_embed(x)

        pos = self.pos_embed(x)
        pos = pos.type_as(embed)

        embed = self.norm(embed + pos)

        return embed

class OctupleEmbedding(nn.Module):
    def __init__(self, dim, n_tokens, max_len):
        super().__init__()
        
        self.token_embeds = nn.ModuleList([
            nn.Embedding(n_tokens[i], dim) for i in range(len(n_tokens))
        ])

        self.embed_linear = nn.Linear(len(n_tokens) * dim, dim)
        self.pos_embed = FreqEmbedding(max_len, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        n_batch, n_seq, n_type = x.shape
        for i in range(n_type):                 
            assert torch.max(x[:, :, i]).item() < self.token_embeds[i].num_embeddings, f"in the model {i}th : max: {torch.max(x[:, :, i]).item()}, num_embeddings: {self.token_embeds[i].num_embeddings}"
            assert torch.min(x[:, :, i]).item() >= 0, f"in the model {i}th min: {torch.min(x[:, :, i]).item()}"
            embed = self.token_embeds[i](x[:, :, i])
            if i == 0:
                embeds = embed
            else:
                embeds = torch.cat((embeds, embed), dim=-1)
        
        embed = self.embed_linear(embeds)
        
        pos = self.pos_embed(x)
        pos = pos.type_as(embed)

        embed = self.norm(embed + pos)

        return embed

class FreqEmbedding(nn.Module):
    """
    Refer to https://github.com/dreamgonfly/transformer-pytorch/blob/master/embeddings.py
    """

    def __init__(self, max_len, dim):
        super().__init__()

        # compute the positional encodings once in log space
        pe = torch.zeros(max_len, dim)
        pe.require_grad = False

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = (torch.arange(0, dim, 2) * -(math.log(10000.0) / dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return self.pe[:, : x.shape[1]]


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.mlp(x)


class Attention(nn.Module):
    """
    Refer to https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py
    """

    def __init__(self, dim, heads=12, dim_head=64):
        super().__init__()

        self.heads = heads
        self.scale = dim_head**-0.5
        hidden_dim = dim_head * heads

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

        self.rotary_emb = RotaryEmbedding(dim=32)

    def forward(self, x, mask=None):
        # x -> (batch (b), seq (n), dim (d))
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        # rotary embedding
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        mask = mask.unsqueeze(1)

        if mask is not None:
            fill_value = 1e-9 if dots.dtype == torch.float32 else 1e-4
            dots.masked_fill_(mask, fill_value)

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)


class Transformer(nn.Module):
    """
    Refer to https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, rate):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.layers = nn.ModuleList([])
        self.dropout = nn.Dropout(rate)

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head),
                        FeedForward(dim, mlp_dim),
                    ]
                )
            )

    def forward(self, x, mask=None):
        h_list = []

        for attn, ff in self.layers:
            x = self.norm1(self.dropout(attn(x, mask)) + x)
            x = self.norm2(self.dropout(ff(x)) + x)
            h_list.append(x[:, 0])

        return x, h_list

class MusicBERT(nn.Module):
    def __init__(self, dim, vocab, depth, heads, dim_head, mlp_dim, max_len, rate):
        super().__init__()
        
        self.vocab = vocab
        self.n_tokens = [len(self.vocab._vocab_base[i]) for i in range(len(self.vocab._vocab_base))]
        num_vocab = sum(self.n_tokens)

        self.embedding = OctupleEmbedding(dim, self.n_tokens, max_len)
        
        self.transformer = Transformer(
            dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            rate=rate,
        )

        self.linear_heads = nn.ModuleList([nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, self.n_tokens[i]),
        ) for i in range(len(self.n_tokens))
        ])
        
        

    def get_attn_pad_mask(self, seq):
        batch_size, len_seq, n_type = seq.shape

        pad_idx = self.vocab.special_tokens.index('PAD_None') 
        pad_attn_mask = seq[:, :, 0].eq(pad_idx).unsqueeze(1) # type tokens are samely padded so we only check the first token is pad or not
        pad_attn_mask = pad_attn_mask.expand(batch_size, len_seq, len_seq)

        return pad_attn_mask

    def forward(self, x):
        attn_mask = self.get_attn_pad_mask(x)

        # Transformers
        x = self.embedding(x)
        h, h_list = self.transformer(x, attn_mask)

        # last layer
        logits = []
        for i in range(len(self.n_tokens)):
            logit = self.linear_heads[i](h)
            logits.append(logit)

        return logits, h[:, 0], h_list


class BERT(nn.Module):
    def __init__(self, dim, vocab, depth, heads, dim_head, mlp_dim, max_len, rate):
        super().__init__()

        self.vocab = vocab
        num_vocab = len(vocab)

        self.embedding = Embedding(dim, num_vocab, max_len)

        self.transformer = Transformer(
            dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            rate=rate,
        )

        self.linear_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, num_vocab),
        )

    def get_attn_pad_mask(self, seq):
        batch_size, len_seq = seq.shape

        pad_idx = self.vocab.to_i(PAD_TOKEN)
        pad_attn_mask = seq.eq(pad_idx).unsqueeze(1)
        pad_attn_mask = pad_attn_mask.expand(batch_size, len_seq, len_seq)

        return pad_attn_mask

    def forward(self, x):
        attn_mask = self.get_attn_pad_mask(x)

        # Transformers
        x = self.embedding(x)
        h, h_list = self.transformer(x, attn_mask)

        # last layer
        logits = self.linear_head(h)

        return logits, h[:, 0], h_list


class BERT_Lightning(pl.LightningModule):
    # BERT base: L=12, H=768, A=12
    def __init__(
        self,
        dim,
        vocab,
        depth=12,
        heads=12,
        dim_head=64,
        mlp_dim=2048,
        max_len=512,
        rate=0.1,
        loss_weights=[1, 1],
        lr=1e-3,
        warm_up=5000,
        temp=1,
        mode="BERT",
        total_steps=10000,
        manual_seed=0,
    ):
        super().__init__()

        self.vocab = vocab
        self.manual_seed = manual_seed

        self.model = MusicBERT(
            dim=dim,
            vocab=self.vocab,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            max_len=max_len,
            rate=rate,
        )

        self.lr = lr
        self.warm_up = warm_up
        self.total_steps = total_steps
        self.temp = temp
        self.mode = mode

        self.loss_weights = loss_weights
        self.mask_ce_loss = nn.CrossEntropyLoss(ignore_index=self.vocab.special_tokens.index('PAD_None'))
        self.nce_ce_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        logits, h, h_list = self.model(x)

        return logits, h, h_list

    def configure_optimizers(self):
        """
        Refer to https://gist.github.com/gautierdag/925760d4295080c1860259dba43e4c01
        """

        opt = AdamW(self.parameters(), lr=self.lr)

        def warm_decay(step):
            if step < self.warm_up:
                return step / self.warm_up
            return self.warm_up**0.5 * step**-0.5

        sch = {
            "scheduler": LambdaLR(opt, warm_decay),
            "interval": "step",
            "frequency": 1,
            "name": "learning_rate",
        }

        return [opt], [sch]

    def on_train_epoch_start(self):
        # This method is called at the start of each training epoch
        set_manual_seed(self.current_epoch)

    def on_validation_epoch_start(self):
        # This method is called at the start of each validation 
        set_manual_seed(self.manual_seed) # ensure same augmentation applied for each validation process

    def get_acc(self, y_pred, y_true):
        y_pred = nn.Softmax(dim=-1)(y_pred)
        y_pred = y_pred.argmax(-1)

        nonzero_idx = y_true != self.vocab.special_tokens.index('PAD_None') 

        nom = (y_pred[nonzero_idx] == y_true[nonzero_idx]).sum()
        denom = y_pred[nonzero_idx].numel()

        return nom / denom

    def compute_nce_loss(self, feat1, feat2):
        """
        Refer to https://github.com/sthalles/SimCLR/blob/master/simclr.py
        """

        labels = torch.cat([torch.arange(feat1.shape[0]) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.type_as(feat1)

        feat = torch.cat([feat1, feat2], dim=0)
        feat = F.normalize(feat, dim=1)
        sim = torch.matmul(feat, feat.T)

        batch_size = feat.shape[0]

        mask = torch.eye(batch_size).bool()
        labels = labels[~mask].view(batch_size, -1).bool()

        sim = sim[~mask].view(batch_size, -1)
        pos = sim[labels].view(batch_size, -1)
        neg = sim[~labels].view(batch_size, -1)

        logits = torch.cat([pos, neg], dim=1) / self.temp
        labels = torch.zeros(logits.shape[0]).type_as(logits).to(torch.long)

        loss = self.nce_ce_loss(logits, labels)

        # get nce accuracy
        pred = torch.argmax(logits, dim=-1)
        acc = (pred == labels).sum() / pred.numel()

        return loss, acc

    def mode_change(self, batch, random_shuffle=False):
        if random_shuffle:
            n_batch = batch["x_mask"].shape[0]
            # get random augmented indices for each idx in batch
            random_mode = [random.randint(0, len(self.mode)-1) for _ in range(n_batch)]
            random_mode = [self.mode[i] for i in random_mode]
            new_batch = torch.zeros_like(batch["x_mask"]) # TODO: for now, it is okay since the padding index is 0
            # for each idx, get the corresponding random augmented data
            for i in range(n_batch):
                if random_mode[i] == "BERT-aug":
                    if len(new_batch[i]) > len(batch["x_aug"][i]): # due to note drop augmentation
                        new_batch[i][:len(batch["x_aug"][i])] = batch["x_aug"][i]
                    elif len(new_batch[i]) < len(batch["x_aug"][i]):
                        NotImplementedError("Currently not implemented for this case.")
                    else:
                        new_batch[i] = batch["x_aug"][i]
                elif random_mode[i] == "BERT-neighbor":
                    if len(new_batch[i]) > len(batch["x_neigh"][i]):
                        new_batch[i][:len(batch["x_neigh"][i])] = batch["x_neigh"][i]
                    elif len(new_batch[i]) < len(batch["x_neigh"][i]):
                        new_batch[i] = batch["x_neigh"][i][:len(new_batch[i])]
                    else:
                        new_batch[i] = batch["x_neigh"][i]
                elif random_mode[i] == "BERT-dropout":
                    new_batch[i] = batch["x_mask"][i]
            return new_batch
        else:
            assert len(self.mode) == 1, 'without random shuffle, only one mode is allowed.' 
            if self.mode in "BERT-aug":
                return batch["x_aug"]

            elif self.mode in "BERT-neighbor":
                return batch["x_neigh"]

            elif self.mode in "BERT-dropout":
                return batch["x_mask"]

    def training_step(self, train_batch, batch_idx):
        y_mask_pred, h_mask_pred, _ = self.model(train_batch["x_mask"])

        if self.mode != ["BERT"]:
            x_pair = self.mode_change(train_batch, random_shuffle=True)
            y_pair_pred, h_pair_pred, _ = self.model(x_pair)

        # MLM loss
        mlm_losses = [self.mask_ce_loss(y_mask_pred[i].transpose(1, 2), train_batch["y_mask"][:, :, i])
            for i in range(len(self.vocab._vocab_base))]
        mlm_accs = [
            self.get_acc(y_mask_pred[i], train_batch["y_mask"][:, :, i]) for i in range(len(self.vocab._vocab_base))
        ]

        mlm_loss = sum(mlm_losses) / len(mlm_losses) # mean of losses

        nce_loss = 0
        nce_acc = 0

        # contrastive loss
        if self.mode != ["BERT"]:
            nce_loss, nce_acc = self.compute_nce_loss(h_mask_pred, h_pair_pred)

        # total loss
        loss = (self.loss_weights[0] * mlm_loss) + (self.loss_weights[1] * nce_loss)

        batch_size = h_mask_pred.shape[0]
        self.log("train_loss", loss, prog_bar=True, batch_size=batch_size)
        self.log("train_nce_loss", nce_loss, prog_bar=True, batch_size=batch_size)
        self.log("train_nce_acc", nce_acc, prog_bar=True, batch_size=batch_size)
        
        for i in range(len(self.vocab._vocab_base)):
            self.log(f"train_mlm_loss_{i}", mlm_losses[i], prog_bar=True, batch_size=batch_size)
            self.log(f"train_mlm_acc_{i}", mlm_accs[i], prog_bar=True, batch_size=batch_size)
        
        return loss

    def get_mode_key_name(self):
        ret = []
        for mode in self.mode:
            if mode == "BERT-aug":
                ret.append("x_aug")
            elif mode == "BERT-neighbor":
                ret.append("x_neigh")
            elif mode == "BERT-dropout":
                ret.append("x_mask")
            else:
                raise ValueError(f"Invalid mode: {mode}")
        return ret
    
    def log_pdseries(self, series, name):
        import wandb
        # Convert the DataFrame to WandB Table
        table = wandb.Table(columns=['File', 'Similarity'])
        for index, value in series.items():
            table.add_data(index, value)
        
        # Log the table; ensure this is done only by one process in case of DDP
        if self.trainer.is_global_zero:
            self.logger.experiment.log({name: table})


    def validation_step(self, val_batch, batch_idx):
        # sample_retrieval
        if batch_idx == 0:
            ret_item = debug_inference(self, self.vocab)
            self.log_pdseries(ret_item, "sample_retrieval")
        
        y_mask_pred, h_mask_pred, _ = self.model(val_batch["x_mask"])

        # MLM loss
        mlm_losses = [self.mask_ce_loss(y_mask_pred[i].transpose(1, 2), val_batch["y_mask"][:, :, i])
            for i in range(len(self.vocab._vocab_base))]
        mlm_accs = [
            self.get_acc(y_mask_pred[i], val_batch["y_mask"][:, :, i]) for i in range(len(self.vocab._vocab_base))
        ]

        mlm_loss = sum(mlm_losses) / len(mlm_losses) # mean of losses
        
        nce_losses = []
        nce_accs = []

        # contrastive loss
        if self.mode != ["BERT"]:
            for mode in self.get_mode_key_name():
                x_pair = val_batch[mode]
                y_pair_pred, h_pair_pred, _ = self.model(x_pair)
                nce_loss, nce_acc = self.compute_nce_loss(h_mask_pred, h_pair_pred)
                nce_losses.append(nce_loss)
                nce_accs.append(nce_acc)
                
        nce_loss = sum(nce_losses) / len(nce_losses) if len(nce_losses) > 0 else 0
        nce_acc = sum(nce_accs) / len(nce_accs) if len(nce_accs) > 0 else 0

        # total loss
        loss = (self.loss_weights[0] * mlm_loss) + (self.loss_weights[1] * nce_loss)

        batch_size = h_mask_pred.shape[0]
        self.log("val_loss", loss, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log("val_nce_loss", nce_loss, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log("val_nce_acc", nce_acc, prog_bar=True, batch_size=batch_size, sync_dist=True)
        if self.mode != ["BERT"]:
            for i in range(len(self.mode)):
                self.log(f"val_nce_loss_{self.mode[i]}", nce_losses[i], prog_bar=True, batch_size=batch_size, sync_dist=True)
                self.log(f"val_nce_acc_{self.mode[i]}", nce_accs[i], prog_bar=True, batch_size=batch_size, sync_dist=True)
        
        self.log("val_mlm_loss", mlm_loss, prog_bar=True, batch_size=batch_size, sync_dist=True)
        for i in range(len(self.vocab._vocab_base)):
            self.log(f"val_mlm_loss_{i}", mlm_losses[i], prog_bar=True, batch_size=batch_size, sync_dist=True)
            self.log(f"val_mlm_acc_{i}", mlm_accs[i], prog_bar=True, batch_size=batch_size, sync_dist=True)

        return loss
    
    def get_attention(self, x, layer=-1):
        pass
            
    def get_last_all_hidden(self, x):
        attn_mask = self.model.get_attn_pad_mask(x)
        # Transformers
        x = self.model.embedding(x)
        h, h_list = self.model.transformer(x, attn_mask)
    
        return h
