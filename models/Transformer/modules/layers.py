import torch.nn as nn
import torch
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()

        self.dim = dim
        self.num_heads = num_heads

        self.dim_Q = dim // num_heads
        self.dim_K = dim // num_heads
        self.dim_V = dim // num_heads

        # layers
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.output_layer = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(p=attn_drop)

    def forward(self, x):
        """Hint: input x tensor shape is (batch_size, num_image_tokens, dim),
        because the bidirectional transformer first will embed each token to dim dimension,
        and then pass to n_layers of encoders consist of Multi-Head Attention and MLP.
        # of head set 16
        Total d_k , d_v set to 768
        d_k , d_v for one head will be 768//16.
        """
        batch_size, num_image_tokens, dim = x.shape

        # original Q K V shape:   (batch_size, num_image_tokens, dim)
        Q, K, V = self.query(x), self.key(x), self.value(x)

        # multi-head Q K V shape: (batch_size, num_heads, num_image_tokens, dim_K)
        Q, K, V = self.permute_QKV_for_multi_head(Q, K, V, batch_size, num_image_tokens)

        # scaled dot-product attention: (batch_size, num_heads, num_image_tokens, num_image_tokens)
        """
        1.  Q @ K = (..., num_image_tokens, d_q) * (..., d_k, num_image_tokens) 
            That is, QK: (batch_size, num_heads, num_image_tokens, num_image_tokens)
        
        2.  attention = QK @ V
                    : (batch_size, num_heads, num_image_tokens, dim_V)
        """
        QK = torch.matmul(Q, K.transpose(-2, -1))
        scaled_QK = QK / math.sqrt(self.dim_K)
        scaled_QK = self.softmax(scaled_QK)

        attention = torch.matmul(scaled_QK, V)

        # concatenated multi-head attention: (batch_size, num_image_tokens, dim)
        attention = self.permute_attention_for_output(
            attention, batch_size, num_image_tokens, dim
        )

        # final output
        output = self.output_layer(attention)
        output = self.dropout(output)

        return output

    def permute_QKV_for_multi_head(self, Q, K, V, batch_size, num_image_tokens):
        """
        input:  Q, K, V: (batch_size, num_image_tokens, dim)
        output: Q, K, V: (batch_size, num_heads, num_image_tokens, dim_Q(dim_K, dim_V))
        """
        # original Q K V shape:   (batch_size, num_image_tokens, dim)

        # reshaped Q K V shape:   (batch_size, num_image_tokens, num_heads, dim_K)
        Q = Q.reshape(batch_size, num_image_tokens, self.num_heads, self.dim_Q)
        K = K.reshape(batch_size, num_image_tokens, self.num_heads, self.dim_K)
        V = V.reshape(batch_size, num_image_tokens, self.num_heads, self.dim_V)

        # transposed Q K V shape: (batch_size, num_heads, num_image_tokens, dim_K)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        return Q, K, V

    def permute_attention_for_output(
        self, attention, batch_size, num_image_tokens, dim
    ):
        """
        input:  attention: (batch_size, num_heads, num_image_tokens, dim_V)
        output: attention: (batch_size, num_image_tokens, dim)
        """
        # transposed attention: (batch_size, num_image_tokens, num_heads, dim_V)
        attention = attention.transpose(1, 2)

        # reshaped attention:   (batch_size, num_image_tokens, num_heads * dim_V)
        attention = attention.reshape(batch_size, num_image_tokens, dim)

        return attention


class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1),
        )

    def forward(self, input):
        return super().forward(input)


class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12),
        )

    def forward(self, input):
        return super().forward(input)


class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)

        x = x + attn
        x = self.LayerNorm1(x)

        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
