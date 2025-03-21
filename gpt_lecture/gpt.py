from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import time

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')

def get_batch(split):
    match split:
        case 'train': data = train_data
        case 'val': data = val_data
        case _: raise ValueError('Invalid split')

    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,)) # Randomly select starting indices for sequences
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix]) # Extract input sequences (x) from those indices and stack them
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix]) # Extract target sequences (y) with +1 offset (next character prediction) and stack them
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

def split_data(data, train_frac=0.9, val_frac=0.1, device='cpu'):
    n = len(data)
    train_data = data[:int(n*train_frac)]
    val_data = data[int(n*train_frac):int(n*(train_frac+val_frac))]
    return train_data.to(device), val_data.to(device)

@torch.no_grad()
def estimate_loss(model, eval_iters=100):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train(model, optimizer, max_iters=100, eval_iters=10):

    start_time = time.time()

    for iter in range(max_iters):

        # every once in a while, estimate the loss on the training and validation sets
        if iter % eval_iters == 0:
            losses = estimate_loss(model, eval_iters=eval_iters)
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time / (iter + 1) * max_iters
            remaining_time = estimated_total_time - elapsed_time
            print(f'Epoch:\t{iter}, Train loss: {losses["train"]:.3f}, Val loss: {losses["val"]:.3f}')
            print(f'Elapsed time: {elapsed_time:.2f}s, Estimated total time: {estimated_total_time:.2f}s, Remaining time: {remaining_time:.2f}s')

        # sample a batch of data
        Xb, Yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(Xb, Yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return model

class Head(nn.Module):
    
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.head_size = head_size
        self.n_embd = n_embd
        self.block_size = block_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # Input of size (batch, time-step, channels)
        # Output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        # Compute the attention (affinity) scores between the queries and keys
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # Perform the weighted aggregation of the values
        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    
    def __init__(self, n_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(n_heads)])
        self.projection = nn.Linear(head_size * n_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out


class FeedForward(nn.Module):

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, n_head, n_embd, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.self_attention = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout) # i.e. 4 heads of 8-dimensional self-attention
        self.ffwd = FeedForward(n_embd, dropout)
        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.self_attention(self.layernorm_1(x)) # Residual connection
        x = x + self.ffwd(self.layernorm_2(x)) # Residual connection
        return x


class GPTLM(nn.Module):

    def __init__(self, block_size, vocab_size, n_embd, n_head, n_layer, dropout=0.2):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_head, n_embd, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        token_embedding = self.token_embedding_table(idx) # (BATCH_SIZE, BLOCK_SIZE, N_EMBD)
        position_embedding = self.position_embedding_table(torch.arange(idx.shape[1], device=idx.device)) # (BLOCK_SIZE, N_EMBD)
        x = token_embedding + position_embedding # (BATCH_SIZE, BLOCK_SIZE, N_EMBD)
        x = self.blocks(x) # (BATCH_SIZE, BLOCK_SIZE, N_EMBD)
        x = self.ln_f(x) # (BATCH_SIZE, BLOCK_SIZE, N_EMBD)
        logits = self.lm_head(x) # (BATCH_SIZE, BLOCK_SIZE, VOCAB_SIZE)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape # (BATCH_SIZE, BLOCK_SIZE, VOCAB_SIZE)
            logits = logits.view(B * T, C)
            targets = targets.view(B * T) # or view(-1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx: (BATCH_SIZE, BLOCK_SIZE) arrays of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:] # crop idx to the last block_size tokens
            logits, loss = self(idx_cond) # get the predictions
            logits = logits[:, -1, :] # becomes (BATCH_SIZE, VOCAB_SIZE)
            probs = F.softmax(logits, dim=-1) # apply softmax to get probabilities
            idx_next = torch.multinomial(probs, num_samples=1) # sample from the distribution
            idx = torch.cat([idx, idx_next], dim=1) # add the new token to the context (BATCH_SIZE, BLOCK_SIZE+1)
        return idx

if __name__ == '__main__':

    # Hyperparameters
    BATCH_SIZE = 32
    BLOCK_SIZE = 128
    N_EMBD = 16
    MAX_ITERS = 10_000
    EVAL_ITERS = MAX_ITERS // 100
    LEARNING_RATE = 5e-4
    N_HEAD = 4
    N_LAYER = 4
    DROPOUT = 0.2
    print(f'Using device: {DEVICE}')

    torch.manual_seed(42)

    ROOT = Path(__file__).parents[1]
    DATA_PATH = ROOT / 'data' / 'shakespeare.txt'
    with open(DATA_PATH, 'r') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    VOCAB_SIZE = len(chars)
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}
    encode = lambda x: torch.tensor([char_to_int[c] for c in x], dtype=torch.long)
    decode = lambda x: ''.join([int_to_char[int(c)] for c in x])

    data = torch.tensor(encode(text), dtype=torch.long, device=DEVICE)
    train_data, val_data = split_data(data, device=DEVICE)

    model = GPTLM(
        block_size=BLOCK_SIZE,
        vocab_size=VOCAB_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')

    checkpoint = train(model, optimizer, max_iters=MAX_ITERS, eval_iters=EVAL_ITERS)

    # Save the model
    os.makedirs(ROOT / 'models', exist_ok=True)
    torch.save(model.state_dict(), ROOT / 'models' / 'checkpoint.pth')

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    print(f'Generated text: {decode(model.generate(context, max_new_tokens=500)[0].tolist())}')
