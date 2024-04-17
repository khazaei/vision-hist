"""
https://arxiv.org/pdf/2010.11929.pdf
"""
import math
from functools import partial

import torch
import torch.nn as nn

LEARNING_RATE = 1e-4
NUM_EPOCHS = 200
BATCH_SIZE = 256
IMAGE_DIM = 224
LEARNING_RATE_DECAY_FACTOR = 0.1
LEARNING_RATE_DECAY_STEP_SIZE = 1000
WEIGHT_DECAY = 1e-2


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()

        self.mlp = nn.Sequential(nn.Linear(in_features=embedding_dim, out_features=mlp_dim),
                                 nn.GELU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(in_features=mlp_dim, out_features=embedding_dim),
                                 nn.Dropout(p=0.5))
        self.init_params()

    def forward(self, x):
        return self.mlp(x)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)


class EncoderBlock(nn.Module):
    """Encoder block"""

    def __init__(self, embedding_dim, num_heads, mlp_dim):
        super().__init__()

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.self_attention = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)

        self.ln_mlp = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            MlpBlock(embedding_dim, mlp_dim),
        )

    def forward(self, inp):
        torch._assert(inp.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {inp.shape}")
        x = self.ln1(inp)
        x, _ = self.self_attention(x, x, x)
        x = self.dropout(x)
        x = x + inp
        y = self.ln_mlp(x)

        return x + y


class Encoder(nn.Module):
    def __init__(self, seq_length, embedding_dim, num_heads, num_layers, mlp_dim):
        super().__init__()

        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, embedding_dim).normal_(std=0.02))
        self.dropout = nn.Dropout(p=0.5)
        layers = []
        for layer in range(num_layers):
            layers.append(EncoderBlock(embedding_dim, num_heads, mlp_dim))

        self.layers = nn.Sequential(*layers)
        self.ln = nn.LayerNorm(embedding_dim)

    def forward(self, inp):
        inp = inp + self.pos_embedding
        return self.ln(self.layers(self.dropout(inp)))


class VisionTransformer(nn.Module):
    def __init__(self, patch_size, num_layers, num_heads, embedding_dim, mlp_dim, num_classes):
        super().__init__()
        self.in_dim = IMAGE_DIM
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

        self.pachify = nn.Conv2d(in_channels=3, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size)
        seq_length = (self.in_dim // patch_size) ** 2

        # Add a class token to start the classification
        self.class_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        seq_length += 1

        self.encoder = Encoder(seq_length, embedding_dim, num_heads, num_layers, mlp_dim)
        self.heads = nn.Sequential(nn.Linear(embedding_dim, num_classes))

        self.init()
        self.optim = torch.optim.AdamW(params=self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.num_epochs = NUM_EPOCHS
        self.batch_size = BATCH_SIZE
        self.in_dim = IMAGE_DIM

    def init(self):
        fan_in = self.pachify.in_channels * self.pachify.kernel_size[0] * self.pachify.kernel_size[1]
        nn.init.trunc_normal_(self.pachify.weight, std=math.sqrt(1 / fan_in))
        if self.pachify.bias is not None:
            nn.init.zeros_(self.pachify.bias)

        for m in self.heads.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def _preprocess(self, x):
        b, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.in_dim, f"Wrong image height! Expected {self.in_dim} but got {h}!")
        torch._assert(w == self.in_dim, f"Wrong image width! Expected {self.in_dim} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (b, c, h, w) -> (b, embedding_dim, n_h, n_w)
        x = self.pachify(x)

        # flatten
        # (b, embedding_dim, n_h, n_w) -> (b, embedding_dim, (n_h * n_w))
        x = x.reshape(b, self.embedding_dim, n_h * n_w)

        # (b, embedding_dim, (n_h * n_w)) -> (b, (n_h * n_w), embedding_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, inp):
        # Reshape and permute the input tensor
        x = self._preprocess(inp)
        b = x.shape[0]

        # Expand the start token to the full batch
        batch_class_token = self.class_token.expand(b, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x

    def optimizer(self):
        return self.optim

    def lr_scheduler(self):
        return torch.optim.lr_scheduler.StepLR(self.optim, step_size=LEARNING_RATE_DECAY_STEP_SIZE,
                                               gamma=LEARNING_RATE_DECAY_FACTOR)


def getViT_B_16(num_classes):
    patch_size = 16
    num_layers = 12
    num_heads = 12
    embedding_dim = 768
    mlp_dim = 3072
    return VisionTransformer(patch_size, num_layers, num_heads, embedding_dim, mlp_dim, num_classes)
