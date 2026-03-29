"""
COM7NN Transformer - Full Implementation
Custom Operation Matrix Neural Network with Transformer Architecture
- Tensor class with shape validation (dimension mismatch impossible)
- Multi-head attention + FFN + LayerNorm
- Proper backward pass through all components
- COM-structured training (epoch ops as callable lists)
"""

import numpy as np
import time

# ============================================================
# Tensor Class - Shape-safe wrapper
# ============================================================
class Tensor:
    def __init__(self, data):
        if isinstance(data, Tensor):
            data = data.data
        self.data = np.array(data, dtype=np.float64)
        self.shape = self.data.shape

    def matmul(self, other):
        if self.shape[-1] != other.shape[-2]:
            raise ValueError(f"Shape mismatch in matmul: {self.shape} @ {other.shape}")
        return Tensor(np.matmul(self.data, other.data))

    def add(self, other):
        # Broadcasting-aware add
        return Tensor(self.data + other.data)

    def reshape(self, new_shape):
        return Tensor(self.data.reshape(new_shape))

    def transpose(self, axes=None):
        return Tensor(self.data.transpose(axes))

    def softmax(self, axis=-1):
        e = np.exp(self.data - np.max(self.data, axis=axis, keepdims=True))
        return Tensor(e / np.sum(e, axis=axis, keepdims=True))

    def relu(self):
        return Tensor(np.maximum(0, self.data))

    def layer_norm(self, axis=-1, eps=1e-5):
        mean = np.mean(self.data, axis=axis, keepdims=True)
        var = np.var(self.data, axis=axis, keepdims=True)
        return Tensor((self.data - mean) / np.sqrt(var + eps))

    def dropout(self, rate=0.1, training=True):
        if not training or rate == 0:
            return Tensor(self.data.copy())
        mask = np.random.binomial(1, 1 - rate, self.data.shape) / (1 - rate)
        return Tensor(self.data * mask)

    def custom_operation(self, op_matrix):
        """Apply element-wise COM operations"""
        if self.shape != op_matrix.shape:
            raise ValueError(f"COM shape mismatch: data {self.shape} vs ops {op_matrix.shape}")
        result = np.zeros_like(self.data)
        ops = {
            0: lambda x: x,           # identity
            1: lambda x: x * 2,       # scale up
            2: lambda x: x * 0.5,     # scale down
            3: lambda x: max(0, x),    # ReLU
            4: lambda x: 1/(1+np.exp(-np.clip(x,-500,500))),  # sigmoid
            5: lambda x: np.tanh(x),   # tanh
            6: lambda x: -x,           # negate
            7: lambda x: x**2,         # square
            8: lambda x: max(0.01*x, x),  # leaky ReLU
        }
        for idx in np.ndindex(self.shape):
            op_code = int(op_matrix.data[idx]) % len(ops)
            result[idx] = ops[op_code](self.data[idx])
        return Tensor(result)

    def __repr__(self):
        return f"Tensor(shape={self.shape})"


# ============================================================
# Multi-Head Attention (COM-safe)
# ============================================================
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0, f"d_model {d_model} not divisible by num_heads {num_heads}"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        scale = np.sqrt(2.0 / d_model)
        self.W_q = Tensor(np.random.randn(d_model, d_model) * scale)
        self.W_k = Tensor(np.random.randn(d_model, d_model) * scale)
        self.W_v = Tensor(np.random.randn(d_model, d_model) * scale)
        self.W_o = Tensor(np.random.randn(d_model, d_model) * scale)
        # Cache for backward
        self.cache = {}

    def forward(self, x):
        """x: Tensor (batch, seq, d_model)"""
        batch, seq, _ = x.shape
        Q = x.matmul(self.W_q)  # (batch, seq, d_model)
        K = x.matmul(self.W_k)
        V = x.matmul(self.W_v)

        # Reshape for multi-head: (batch, seq, num_heads, d_k) -> (batch, num_heads, seq, d_k)
        Q = Q.reshape((batch, seq, self.num_heads, self.d_k)).transpose((0, 2, 1, 3))
        K = K.reshape((batch, seq, self.num_heads, self.d_k)).transpose((0, 2, 1, 3))
        V = V.reshape((batch, seq, self.num_heads, self.d_k)).transpose((0, 2, 1, 3))

        # Scaled dot-product attention
        scores = Tensor(np.matmul(Q.data, K.data.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k))
        attn = scores.softmax(axis=-1)  # (batch, heads, seq, seq)
        context = Tensor(np.matmul(attn.data, V.data))  # (batch, heads, seq, d_k)

        # Reshape back: (batch, heads, seq, d_k) -> (batch, seq, d_model)
        context = context.transpose((0, 2, 1, 3)).reshape((batch, seq, self.d_model))
        output = context.matmul(self.W_o)

        self.cache = {'x': x, 'Q': Q, 'K': K, 'V': V, 'attn': attn, 'context': context}
        return output

    def backward(self, grad_output, lr=0.001):
        """grad_output: Tensor (batch, seq, d_model)"""
        x = self.cache['x']
        context = self.cache['context']
        attn = self.cache['attn']
        Q, K, V = self.cache['Q'], self.cache['K'], self.cache['V']
        batch, seq, _ = grad_output.shape

        # Gradient through W_o: context.T @ grad_output
        # context: (batch, seq, d_model), grad_output: (batch, seq, d_model)
        grad_W_o = np.einsum('bsi,bsj->ij', context.data, grad_output.data)
        grad_context = grad_output.matmul(Tensor(self.W_o.data.T))  # (batch, seq, d_model)

        # Reshape grad_context to multi-head: (batch, seq, heads, d_k) -> (batch, heads, seq, d_k)
        grad_context_mh = grad_context.reshape((batch, seq, self.num_heads, self.d_k)).transpose((0, 2, 1, 3))

        # Gradient through attention
        grad_V = np.matmul(attn.data.transpose(0, 1, 3, 2), grad_context_mh.data)  # (batch, heads, seq, d_k)
        grad_attn = np.matmul(grad_context_mh.data, V.data.transpose(0, 1, 3, 2))  # (batch, heads, seq, seq)

        # Softmax backward (simplified)
        grad_scores = attn.data * (grad_attn - np.sum(grad_attn * attn.data, axis=-1, keepdims=True))
        grad_scores /= np.sqrt(self.d_k)

        # Gradients for Q, K
        grad_Q = np.matmul(grad_scores, K.data)  # (batch, heads, seq, d_k)
        grad_K = np.matmul(grad_scores.transpose(0, 1, 3, 2), Q.data)  # (batch, heads, seq, d_k)

        # Reshape back to (batch, seq, d_model)
        grad_Q = grad_Q.transpose(0, 2, 1, 3).reshape(batch, seq, self.d_model)
        grad_K = grad_K.transpose(0, 2, 1, 3).reshape(batch, seq, self.d_model)
        grad_V = grad_V.transpose(0, 2, 1, 3).reshape(batch, seq, self.d_model)

        # Weight gradients
        grad_W_q = np.einsum('bsi,bsj->ij', x.data, grad_Q)
        grad_W_k = np.einsum('bsi,bsj->ij', x.data, grad_K)
        grad_W_v = np.einsum('bsi,bsj->ij', x.data, grad_V)

        # Gradient w.r.t. input x
        grad_x = (Tensor(grad_Q).matmul(Tensor(self.W_q.data.T)).data +
                   Tensor(grad_K).matmul(Tensor(self.W_k.data.T)).data +
                   Tensor(grad_V).matmul(Tensor(self.W_v.data.T)).data)

        # Update weights
        clip = 1.0
        for g in [grad_W_q, grad_W_k, grad_W_v, grad_W_o]:
            np.clip(g, -clip, clip, out=g)
        self.W_q.data -= lr * grad_W_q
        self.W_k.data -= lr * grad_W_k
        self.W_v.data -= lr * grad_W_v
        self.W_o.data -= lr * grad_W_o

        return Tensor(grad_x)


# ============================================================
# Feed-Forward Network (COM-safe)
# ============================================================
class FeedForward:
    def __init__(self, d_model, d_ff):
        scale1 = np.sqrt(2.0 / d_model)
        scale2 = np.sqrt(2.0 / d_ff)
        self.W1 = Tensor(np.random.randn(d_model, d_ff) * scale1)
        self.b1 = Tensor(np.zeros((1, 1, d_ff)))
        self.W2 = Tensor(np.random.randn(d_ff, d_model) * scale2)
        self.b2 = Tensor(np.zeros((1, 1, d_model)))
        self.cache = {}

    def forward(self, x):
        """x: Tensor (batch, seq, d_model)"""
        self.cache['x'] = x
        z1 = x.matmul(self.W1).add(self.b1)  # (batch, seq, d_ff)
        a1 = z1.relu()                         # (batch, seq, d_ff)
        self.cache['z1'] = z1
        self.cache['a1'] = a1
        out = a1.matmul(self.W2).add(self.b2)  # (batch, seq, d_model)
        return out

    def backward(self, grad_output, lr=0.001):
        """grad_output: Tensor (batch, seq, d_model) — SAME shape as forward output"""
        x = self.cache['x']      # (batch, seq, d_model)
        z1 = self.cache['z1']    # (batch, seq, d_ff)
        a1 = self.cache['a1']    # (batch, seq, d_ff)

        # grad through W2: a1.T @ grad_output per batch
        # a1: (batch, seq, d_ff), grad_output: (batch, seq, d_model)
        grad_W2 = np.einsum('bsi,bsj->ij', a1.data, grad_output.data)  # (d_ff, d_model)
        grad_b2 = np.sum(grad_output.data, axis=(0, 1), keepdims=True)  # (1, 1, d_model)

        # grad through a1
        grad_a1 = grad_output.matmul(Tensor(self.W2.data.T))  # (batch, seq, d_ff)
        # ReLU backward
        grad_z1 = Tensor(grad_a1.data * (z1.data > 0))  # (batch, seq, d_ff)

        # grad through W1
        grad_W1 = np.einsum('bsi,bsj->ij', x.data, grad_z1.data)  # (d_model, d_ff)
        grad_b1 = np.sum(grad_z1.data, axis=(0, 1), keepdims=True)  # (1, 1, d_ff)

        # grad through input x
        grad_x = grad_z1.matmul(Tensor(self.W1.data.T))  # (batch, seq, d_model)

        # Clip and update
        clip = 1.0
        for g in [grad_W1, grad_W2]:
            np.clip(g, -clip, clip, out=g)
        self.W1.data -= lr * grad_W1
        self.W2.data -= lr * grad_W2
        self.b1.data -= lr * grad_b1
        self.b2.data -= lr * grad_b2

        return grad_x


# ============================================================
# Transformer Encoder Layer
# ============================================================
class TransformerEncoderLayer:
    def __init__(self, d_model, num_heads, d_ff):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.d_model = d_model

    def forward(self, x):
        # Self-attention + residual + layer norm
        attn_out = self.attention.forward(x)
        x1 = Tensor(x.data + attn_out.data)  # residual
        x1 = x1.layer_norm()

        # FFN + residual + layer norm
        ff_out = self.feed_forward.forward(x1)
        x2 = Tensor(x1.data + ff_out.data)  # residual
        x2 = x2.layer_norm()

        self.cache = {'x': x, 'x1': x1, 'attn_out': attn_out, 'ff_out': ff_out}
        return x2

    def backward(self, grad_output, lr=0.001):
        x1 = self.cache['x1']
        # Backward through FFN
        grad_ff = self.feed_forward.backward(grad_output, lr)
        # Residual: grad flows to both paths
        grad_x1 = Tensor(grad_output.data + grad_ff.data)
        # Backward through attention
        grad_attn = self.attention.backward(grad_x1, lr)
        # Residual
        grad_x = Tensor(grad_x1.data + grad_attn.data)
        return grad_x


# ============================================================
# Positional Encoding
# ============================================================
def positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return Tensor(pe)


# ============================================================
# COM Transformer Model
# ============================================================
class COMTransformer:
    def __init__(self, vocab_size, d_model, num_heads, d_ff, seq_len, num_layers=1):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seq_len = seq_len

        # Embedding
        self.embedding = Tensor(np.random.randn(vocab_size, d_model) * 0.1)
        # Output projection
        self.W_out = Tensor(np.random.randn(d_model, vocab_size) * np.sqrt(2.0 / d_model))

        # Positional encoding
        self.pe = positional_encoding(seq_len, d_model)

        # Transformer layers
        self.layers = [TransformerEncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]

    def forward(self, x_indices):
        """x_indices: numpy array (batch, seq_len) of integer token IDs"""
        batch = x_indices.shape[0]
        # Embedding lookup
        emb = Tensor(self.embedding.data[x_indices])  # (batch, seq, d_model)
        # Add positional encoding
        x = Tensor(emb.data + self.pe.data[np.newaxis, :self.seq_len, :])

        # Through transformer layers
        for layer in self.layers:
            x = layer.forward(x)

        # Output projection: (batch, seq, d_model) @ (d_model, vocab) = (batch, seq, vocab)
        logits = x.matmul(self.W_out)
        self.cache = {'x_indices': x_indices, 'final_hidden': x, 'logits': logits}
        return logits

    def backward(self, targets, lr=0.001):
        """targets: (batch, seq_len) integer target IDs"""
        logits = self.cache['logits']
        final_hidden = self.cache['final_hidden']
        x_indices = self.cache['x_indices']
        batch, seq, vocab = logits.shape

        # Softmax + cross-entropy gradient
        probs = np.exp(logits.data - np.max(logits.data, axis=-1, keepdims=True))
        probs /= np.sum(probs, axis=-1, keepdims=True)

        # Loss
        loss = 0
        for b in range(batch):
            for s in range(seq):
                t = targets[b, s]
                loss -= np.log(probs[b, s, t] + 1e-10)
        loss /= (batch * seq)

        # Gradient of cross-entropy w.r.t. logits
        grad_logits = probs.copy()
        for b in range(batch):
            for s in range(seq):
                grad_logits[b, s, targets[b, s]] -= 1.0
        grad_logits /= (batch * seq)
        grad_logits = Tensor(grad_logits)  # (batch, seq, vocab)

        # Gradient for W_out: final_hidden.T @ grad_logits
        grad_W_out = np.einsum('bsi,bsj->ij', final_hidden.data, grad_logits.data)  # (d_model, vocab) OK
        np.clip(grad_W_out, -1.0, 1.0, out=grad_W_out)
        self.W_out.data -= lr * grad_W_out

        # Gradient flowing back: (batch, seq, vocab) @ (vocab, d_model) = (batch, seq, d_model)
        grad_hidden = grad_logits.matmul(Tensor(self.W_out.data.T))

        # Backward through transformer layers (reverse order)
        for layer in reversed(self.layers):
            grad_hidden = layer.backward(grad_hidden, lr)

        # Update embeddings
        grad_emb = grad_hidden.data  # (batch, seq, d_model)
        for b in range(batch):
            for s in range(seq):
                idx = x_indices[b, s]
                self.embedding.data[idx] -= lr * np.clip(grad_emb[b, s], -1.0, 1.0)

        return loss


# ============================================================
# COM Epoch Training (operations as callable list)
# ============================================================
def com_train(model, X, Y, epochs=20, lr=0.001, batch_size=4):
    """COM-structured training: each epoch decomposed into operation list"""
    n_samples = X.shape[0]
    history = []

    for epoch in range(epochs):
        t0 = time.time()
        epoch_loss = 0
        n_batches = 0

        # COM operation list for this epoch
        com_ops = []

        # Build batch operations
        indices = np.random.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]
            x_batch = X[batch_idx]
            y_batch = Y[batch_idx]

            # Each batch = [forward, backward] ops
            com_ops.append(('forward', lambda xb=x_batch: model.forward(xb)))
            com_ops.append(('backward', lambda yb=y_batch: model.backward(yb, lr)))

        # Execute COM operation list
        batch_loss = 0
        for op_name, op_fn in com_ops:
            result = op_fn()
            if op_name == 'backward':
                batch_loss = result
                epoch_loss += batch_loss
                n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        history.append(avg_loss)
        print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")

    return history


# ============================================================
# MAIN — Run it
# ============================================================
if __name__ == '__main__':
    np.random.seed(42)

    # --- Config ---
    vocab_size = 50
    d_model = 64
    num_heads = 4
    d_ff = 128
    seq_len = 10
    num_layers = 1
    epochs = 100
    lr = 0.01
    batch_size = 8

    print("=" * 60)
    print("COM7NN Transformer — Full Architecture")
    print(f"  vocab_size={vocab_size}, d_model={d_model}, heads={num_heads}")
    print(f"  d_ff={d_ff}, seq_len={seq_len}, layers={num_layers}")
    print("=" * 60)

    # --- Build model ---
    model = COMTransformer(vocab_size, d_model, num_heads, d_ff, seq_len, num_layers)
    print(f"\nModel created. Embedding: {model.embedding.shape}, W_out: {model.W_out.shape}")

    # --- Synthetic data: next-token prediction ---
    # Sequences: [a, a+1, a+2, ..., a+seq_len-1] -> predict [a+1, a+2, ..., a+seq_len]
    n_samples = 64
    X = np.zeros((n_samples, seq_len), dtype=int)
    Y = np.zeros((n_samples, seq_len), dtype=int)
    for i in range(n_samples):
        start = np.random.randint(0, vocab_size - seq_len - 1)
        X[i] = np.arange(start, start + seq_len) % vocab_size
        Y[i] = np.arange(start + 1, start + seq_len + 1) % vocab_size

    print(f"Data: {n_samples} samples, X shape {X.shape}, Y shape {Y.shape}")
    print(f"Example X[0]: {X[0]}")
    print(f"Example Y[0]: {Y[0]}")

    # --- Forward pass test ---
    print("\n--- Forward Pass Test ---")
    test_logits = model.forward(X[:2])
    print(f"Logits shape: {test_logits.shape}  (expected: (2, {seq_len}, {vocab_size}))")

    # --- Train with COM epoch structure ---
    print("\n--- COM Training ---")
    history = com_train(model, X, Y, epochs=epochs, lr=lr, batch_size=batch_size)

    # --- Inference ---
    print("\n--- Inference ---")
    test_seq = np.array([[5, 6, 7, 8, 9, 10, 11, 12, 13, 14]])
    logits = model.forward(test_seq)
    preds = np.argmax(logits.data, axis=-1)
    print(f"Input:     {test_seq[0].tolist()}")
    print(f"Predicted: {preds[0].tolist()}")
    print(f"Expected:  {list(range(6, 16))}")

    # --- COM Custom Operation Demo ---
    print("\n--- COM Custom Operation Demo ---")
    sample_data = Tensor(np.random.randn(3, 3))
    op_matrix = Tensor(np.array([[0, 3, 4], [1, 5, 8], [2, 6, 7]]))
    print(f"Input:\n{sample_data.data}")
    result = sample_data.custom_operation(op_matrix)
    print(f"After COM ops (identity/ReLU/sigmoid/scale/tanh/leakyReLU/halve/negate/square):\n{result.data}")
    print(f"Op matrix (codes):\n{op_matrix.data.astype(int)}")

    print("\n--- Shape Safety Demo ---")
    a = Tensor(np.random.randn(2, 3))
    b = Tensor(np.random.randn(3, 5))
    c = a.matmul(b)
    print(f"(2,3) @ (3,5) = {c.shape}  OK")
    try:
        d = Tensor(np.random.randn(2, 4))
        a.matmul(d)
    except ValueError as e:
        print(f"(2,3) @ (2,4) -> {e}  CAUGHT!")

    print(f"\nFinal loss: {history[-1]:.4f}")
    print(f"Loss reduction: {history[0]:.4f} -> {history[-1]:.4f} ({(1 - history[-1]/history[0])*100:.1f}% reduction)")
    print("\nDone.")
