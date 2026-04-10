"""
COM7NN Transformer vs Standard NumPy Transformer — Head-to-Head Comparison
============================================================================
Same architecture (d_model=64, 4 heads, 1 layer), same data, same hyperparams.
Compares:
  1. Training loss convergence
  2. Forward/backward speed
  3. Inference accuracy

The COM transformer uses COM's Tensor class with shape-safety and custom operations.
The Standard transformer uses raw NumPy with identical math.
"""

import numpy as np
import time
import sys
import os

# ============================================================
# STANDARD TRANSFORMER (pure NumPy, no COM overhead)
# ============================================================

def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

def layer_norm(x, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


class StdMultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        scale = np.sqrt(2.0 / d_model)
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale
        self.cache = {}

    def forward(self, x):
        batch, seq, _ = x.shape
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        Q = Q.reshape(batch, seq, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

        scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        attn = softmax(scores, axis=-1)
        context = attn @ V

        context = context.transpose(0, 2, 1, 3).reshape(batch, seq, self.d_model)
        output = context @ self.W_o

        self.cache = {'x': x, 'Q': Q, 'K': K, 'V': V, 'attn': attn, 'context': context}
        return output

    def backward(self, grad_output, lr):
        x = self.cache['x']
        context = self.cache['context']
        attn = self.cache['attn']
        Q, K, V = self.cache['Q'], self.cache['K'], self.cache['V']
        batch, seq, _ = grad_output.shape

        grad_W_o = np.einsum('bsi,bsj->ij', context, grad_output)
        grad_context = grad_output @ self.W_o.T

        grad_context_mh = grad_context.reshape(batch, seq, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        grad_V = attn.transpose(0, 1, 3, 2) @ grad_context_mh
        grad_attn = grad_context_mh @ V.transpose(0, 1, 3, 2)

        grad_scores = attn * (grad_attn - np.sum(grad_attn * attn, axis=-1, keepdims=True))
        grad_scores /= np.sqrt(self.d_k)

        grad_Q = grad_scores @ K
        grad_K = grad_scores.transpose(0, 1, 3, 2) @ Q

        grad_Q = grad_Q.transpose(0, 2, 1, 3).reshape(batch, seq, self.d_model)
        grad_K = grad_K.transpose(0, 2, 1, 3).reshape(batch, seq, self.d_model)
        grad_V = grad_V.transpose(0, 2, 1, 3).reshape(batch, seq, self.d_model)

        grad_W_q = np.einsum('bsi,bsj->ij', x, grad_Q)
        grad_W_k = np.einsum('bsi,bsj->ij', x, grad_K)
        grad_W_v = np.einsum('bsi,bsj->ij', x, grad_V)

        self.W_o -= lr * np.clip(grad_W_o, -1.0, 1.0)
        self.W_q -= lr * np.clip(grad_W_q, -1.0, 1.0)
        self.W_k -= lr * np.clip(grad_W_k, -1.0, 1.0)
        self.W_v -= lr * np.clip(grad_W_v, -1.0, 1.0)

        return (grad_Q @ self.W_q.T + grad_K @ self.W_k.T + grad_V @ self.W_v.T)


class StdFeedForward:
    def __init__(self, d_model, d_ff):
        scale1 = np.sqrt(2.0 / d_model)
        scale2 = np.sqrt(2.0 / d_ff)
        self.W1 = np.random.randn(d_model, d_ff) * scale1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale2
        self.b2 = np.zeros(d_model)
        self.cache = {}

    def forward(self, x):
        hidden = x @ self.W1 + self.b1
        relu_out = np.maximum(0, hidden)
        output = relu_out @ self.W2 + self.b2
        self.cache = {'x': x, 'hidden': hidden, 'relu_out': relu_out}
        return output

    def backward(self, grad_output, lr):
        x = self.cache['x']
        relu_out = self.cache['relu_out']
        hidden = self.cache['hidden']

        grad_W2 = np.einsum('bsi,bsj->ij', relu_out, grad_output)
        grad_b2 = np.sum(grad_output, axis=(0, 1))
        grad_relu = grad_output @ self.W2.T

        grad_hidden = grad_relu * (hidden > 0).astype(float)

        grad_W1 = np.einsum('bsi,bsj->ij', x, grad_hidden)
        grad_b1 = np.sum(grad_hidden, axis=(0, 1))

        self.W2 -= lr * np.clip(grad_W2, -1.0, 1.0)
        self.b2 -= lr * np.clip(grad_b2, -1.0, 1.0)
        self.W1 -= lr * np.clip(grad_W1, -1.0, 1.0)
        self.b1 -= lr * np.clip(grad_b1, -1.0, 1.0)

        return grad_hidden @ self.W1.T


class StdTransformerLayer:
    def __init__(self, d_model, num_heads, d_ff):
        self.attn = StdMultiHeadAttention(d_model, num_heads)
        self.ff = StdFeedForward(d_model, d_ff)
        self.cache = {}

    def forward(self, x):
        attn_out = self.attn.forward(x)
        x2 = layer_norm(x + attn_out)
        ff_out = self.ff.forward(x2)
        out = layer_norm(x2 + ff_out)
        self.cache = {'x': x, 'x2': x2, 'attn_out': attn_out, 'ff_out': ff_out}
        return out

    def backward(self, grad_output, lr):
        x2 = self.cache['x2']
        grad_ff = self.ff.backward(grad_output, lr)
        grad_x2 = grad_output + grad_ff
        grad_attn = self.attn.backward(grad_x2, lr)
        return grad_x2 + grad_attn


class StdTransformer:
    def __init__(self, vocab_size, d_model, num_heads, d_ff, seq_len, num_layers):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = np.random.randn(vocab_size, d_model) * 0.02
        self.pos_encoding = np.zeros((seq_len, d_model))
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                self.pos_encoding[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
                if i + 1 < d_model:
                    self.pos_encoding[pos, i+1] = np.cos(pos / (10000 ** (i / d_model)))
        self.layers = [StdTransformerLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.W_out = np.random.randn(d_model, vocab_size) * np.sqrt(2.0 / d_model)
        self.cache = {}

    def forward(self, x_indices):
        batch, seq = x_indices.shape
        x = self.embedding[x_indices] + self.pos_encoding[:seq]
        for layer in self.layers:
            x = layer.forward(x)
        logits = x @ self.W_out
        self.cache = {'x_indices': x_indices, 'hidden': x, 'logits': logits}
        return logits

    def backward(self, y_true, lr):
        x_indices = self.cache['x_indices']
        logits = self.cache['logits']
        batch, seq, vocab = logits.shape

        probs = softmax(logits, axis=-1)
        loss = 0
        for b in range(batch):
            for s in range(seq):
                p = probs[b, s, y_true[b, s]]
                loss -= np.log(max(p, 1e-10))
        loss /= (batch * seq)

        grad_logits = probs.copy()
        for b in range(batch):
            for s in range(seq):
                grad_logits[b, s, y_true[b, s]] -= 1
        grad_logits /= (batch * seq)

        grad_W_out = np.einsum('bsi,bsj->ij', self.cache['hidden'], grad_logits)
        self.W_out -= lr * np.clip(grad_W_out, -1.0, 1.0)

        grad_hidden = grad_logits @ self.W_out.T
        for layer in reversed(self.layers):
            grad_hidden = layer.backward(grad_hidden, lr)

        grad_emb = grad_hidden
        for b in range(batch):
            for s in range(seq):
                idx = x_indices[b, s]
                self.embedding[idx] -= lr * np.clip(grad_emb[b, s], -1.0, 1.0)
        return loss


# ============================================================
# Import COM7NN Transformer
# ============================================================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, 'C:/Users/Scott')
from com7nn_transformer import COMTransformer, com_train, Tensor


# ============================================================
# COMPARISON
# ============================================================
def train_standard(model, X, Y, epochs, lr, batch_size):
    n_samples = X.shape[0]
    history = []
    for epoch in range(epochs):
        t0 = time.time()
        epoch_loss = 0
        n_batches = 0
        indices = np.random.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]
            model.forward(X[batch_idx])
            loss = model.backward(Y[batch_idx], lr)
            epoch_loss += loss
            n_batches += 1
        avg = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        history.append(avg)
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg:.4f} | Time: {elapsed:.3f}s")
    return history


if __name__ == '__main__':
    np.random.seed(42)

    # Config — identical for both
    vocab_size = 50
    d_model = 64
    num_heads = 4
    d_ff = 128
    seq_len = 10
    num_layers = 1
    epochs = 100
    lr = 0.01
    batch_size = 8

    # Build data
    n_samples = 64
    X = np.zeros((n_samples, seq_len), dtype=int)
    Y = np.zeros((n_samples, seq_len), dtype=int)
    for i in range(n_samples):
        start = np.random.randint(0, vocab_size - seq_len - 1)
        X[i] = np.arange(start, start + seq_len) % vocab_size
        Y[i] = np.arange(start + 1, start + seq_len + 1) % vocab_size

    print("=" * 70)
    print("  COM7NN Transformer vs Standard NumPy Transformer")
    print(f"  Config: vocab={vocab_size} d_model={d_model} heads={num_heads} "
          f"d_ff={d_ff} seq={seq_len} layers={num_layers}")
    print(f"  Data: {n_samples} samples, {epochs} epochs, lr={lr}, batch={batch_size}")
    print("=" * 70)

    # ---- Train Standard ----
    print("\n--- STANDARD TRANSFORMER (raw NumPy) ---")
    np.random.seed(42)
    std_model = StdTransformer(vocab_size, d_model, num_heads, d_ff, seq_len, num_layers)
    t0 = time.time()
    std_history = train_standard(std_model, X, Y, epochs, lr, batch_size)
    std_time = time.time() - t0
    print(f"  Total time: {std_time:.2f}s")

    # Standard inference
    test_seq = np.array([[5, 6, 7, 8, 9, 10, 11, 12, 13, 14]])
    std_logits = std_model.forward(test_seq)
    std_preds = np.argmax(std_logits, axis=-1)
    std_correct = np.sum(std_preds[0] == np.arange(6, 16))

    # ---- Train COM ----
    print("\n--- COM7NN TRANSFORMER (Tensor class + shape safety) ---")
    np.random.seed(42)
    com_model = COMTransformer(vocab_size, d_model, num_heads, d_ff, seq_len, num_layers)
    t0 = time.time()
    com_history = com_train(com_model, X, Y, epochs, lr, batch_size)
    com_time = time.time() - t0
    print(f"  Total time: {com_time:.2f}s")

    # COM inference
    com_logits = com_model.forward(test_seq)
    com_preds = np.argmax(com_logits.data, axis=-1)
    com_correct = np.sum(com_preds[0] == np.arange(6, 16))

    # ---- Results ----
    print("\n" + "=" * 70)
    print("  COMPARISON RESULTS")
    print("=" * 70)
    print(f"\n{'Metric':<25} | {'Standard':>12} | {'COM7NN':>12} | {'Winner':>10}")
    print(f"{'-'*25} | {'-'*12} | {'-'*12} | {'-'*10}")

    # Loss comparison
    std_final = std_history[-1]
    com_final = com_history[-1]
    loss_winner = "Standard" if std_final < com_final else ("COM7NN" if com_final < std_final else "Tie")
    print(f"{'Initial loss':<25} | {std_history[0]:>12.4f} | {com_history[0]:>12.4f} |")
    print(f"{'Final loss':<25} | {std_final:>12.4f} | {com_final:>12.4f} | {loss_winner:>10}")
    print(f"{'Loss reduction %':<25} | {(1-std_final/std_history[0])*100:>11.1f}% | {(1-com_final/com_history[0])*100:>11.1f}% |")

    # Speed comparison
    speed_winner = "Standard" if std_time < com_time else "COM7NN"
    ratio = com_time / std_time if std_time > 0 else 0
    print(f"{'Training time':<25} | {std_time:>11.2f}s | {com_time:>11.2f}s | {speed_winner:>10}")
    print(f"{'Speed ratio':<25} | {'1.00x':>12} | {f'{ratio:.2f}x':>12} |")

    # Accuracy
    print(f"{'Inference correct':<25} | {f'{std_correct}/10':>12} | {f'{com_correct}/10':>12} |")
    print(f"{'Predicted (std)':<25} | {str(std_preds[0].tolist()):>40}")
    print(f"{'Predicted (COM)':<25} | {str(com_preds[0].tolist()):>40}")
    print(f"{'Expected':<25} | {str(list(range(6,16))):>40}")

    print(f"\nKey insight: COM7NN adds shape-safety and custom operation support")
    print(f"with {'minimal' if ratio < 1.3 else 'moderate' if ratio < 2 else 'significant'} "
          f"overhead ({ratio:.1f}x) vs raw NumPy.")
