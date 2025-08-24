
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout, bias=False, max_seq=2048):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3*n_embd, bias=bias)
        self.proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Causal mask as buffer (upper triangular)
        mask = torch.full((max_seq, max_seq), float('-inf'))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("attn_mask", mask)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x)  # [B,T,3C]
        q, k, v = qkv.split(C, dim=2)
        # reshape to heads
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1,2)  # [B, nh, T, hd]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1,2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1,2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,nh,T,T]
        att = att + self.attn_mask[:T,:T]  # causal
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # [B,nh,T,hd]
        y = y.transpose(1,2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, n_embd, dropout, bias=False, mult=4):
        super().__init__()
        self.fc = nn.Linear(n_embd, mult*n_embd, bias=bias)
        self.proj = nn.Linear(mult*n_embd, n_embd, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)
        x = F.gelu(x)
        x = self.drop(self.proj(x))
        return x

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, bias=False, max_seq=2048):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, bias, max_seq)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout, bias)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class PureAutoregressiveTransformer(nn.Module):
    def __init__(self, n_embd=512, n_head=8, n_layer=8, dropout=0.1, max_seq=1024, vocab_size=256, bias=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq = max_seq
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(max_seq, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout, bias, max_seq) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # weight tying helps
        self.head.weight = self.token_emb.weight

        # init
        for pn, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.max_seq:
            raise ValueError(f"Sequence length {T} exceeds max_seq {self.max_seq}")

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=1.0, top_k=None, top_p=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.where(logits < v[:, [-1]], torch.full_like(logits, float('-inf')), logits)

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cum = probs.cumsum(dim=-1)
                mask = cum > top_p
                mask[:, 1:] = mask[:, :-1].clone()
                mask[:, 0] = 0
                idx_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                idx_to_remove.scatter_(1, sorted_idx, mask)
                logits = logits.masked_fill(idx_to_remove, float('-inf'))

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx
