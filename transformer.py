# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 17:18:31 2021

@author: ainer
"""
# citation:
# 1. The Annotated Transformer: http://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding
# 2. pytorch-Transformer: https://pytorch.org/docs/master/_modules/torch/nn/modules/transformer.html#Transformer
# 3. 【Transformer】Aurora的文章 - 知乎: https://zhuanlan.zhihu.com/p/403433120
import torch
from torch import nn
import math
import numpy as np
import copy


def clone(module, N):
    # generate N identical layers
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, attn_mask, key_padding_mask):
    """
    Do self-attention.
    Parameters
    ----------
    query : Tensor
        [batch_size, n_head, q_len, d_k].
    key : Tensor
        [batch_size, n_head, k_len, d_k].
    value : Tensor
        [batch_size, n_head, k_len, d_v].
    attn_mask : None or Tensor
        [q_len, k_len], the mask for attention to avoiding label leaking.
    key_padding_mask : None or ByteTensor or BoolTensor
        [batch_size, k_len], the mask for key matrix.

    Returns
    -------
    context : Tensor
        [batch_size, n_head, q_len, d_v], the context vector after attention.
    attn : Tensor
        [batch_size, n_head, q_len, k_len], the weight matrix of query and key.

    """
    batch_size, n_head, q_len, d_k= query.shape
    k_len = key.shape[-2]
    mask = torch.zeros((batch_size, n_head, q_len, k_len))
    dot = torch.matmul(query, key.permute(0, 1, 3, 2)) / math.sqrt(d_k)
    if key_padding_mask is not None:
        batch_k_idx = torch.nonzero(key_padding_mask.int())
        batch_idx, k_idx = batch_k_idx[:, 0], batch_k_idx[:, 1]
        mask[batch_idx, :, :, k_idx] = 1.0
    if attn_mask is None:
        pass
    elif 'float' not in str(attn_mask.dtype):
        # if the mask is BoolTensor, the masked position is True value
        # if the mask is ByteTensor, the masked position is non-zero positions
        q_k_idx = torch.nonzero(attn_mask.int())
        q_idx, k_idx = q_k_idx[:, 0], q_k_idx[:, 1]
        mask[:, :, q_idx, k_idx] = 1.0    
    else:
        dot += attn_mask
    dot.masked_fill_(mask != 0, -1e9)
    # attn: [batch_size, n_head, q_len, k_len]
    attn = nn.Softmax(dim=-1)(dot)
    # context: [batch_size, n_head, q_len, d_v]
    context = torch.matmul(attn, value)
    return context, attn


class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.attn = None
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        self.WQ = nn.Linear(d_model, self.d_k*n_head, bias=False)
        self.WK = nn.Linear(d_model, self.d_k*n_head, bias=False)
        self.WV = nn.Linear(d_model, self.d_v*n_head, bias=False)
        self.WO = nn.Linear(self.d_v*n_head, d_model, bias=False)
        
    def forward(self, query, key, value, attn_mask=None,
                key_padding_mask=None):
        # query: [batch_size, q_len, d_model]
        batch_size, q_len = query.shape[0], query.shape[1]
        # Q: [batch_size, n_head, q_len, d_k]
        Q = self.WQ(query).view(batch_size, -1, self.n_head, self.d_k).transpose(1,2)
        K = self.WK(key).view(batch_size, -1, self.n_head, self.d_k).transpose(1,2)
        V = self.WV(value).view(batch_size, -1, self.n_head, self.d_v).transpose(1,2)
        context, self.attn = attention(Q, K, V, attn_mask=attn_mask,
                                       key_padding_mask=key_padding_mask)
        # context->[batch_size, q_len, n_head*d_v]
        output = self.WO(context.transpose(1, 2).reshape(batch_size, q_len, self.n_head*self.d_v))
        # output: [batch_size, q_len, d_model]
        return output


class PositionEncode(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionEncode, self).__init__()
        position_encode = torch.zeros(max_len, d_model)  # [seq_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        position = position * div_term
        position_encode[:, 0::2] = torch.sin(position)
        position_encode[:, 1::2] = torch.cos(position)
        self.pe = position_encode.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        # crop the position encode matrix with the length of seq_len
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        return x


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ffn):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ffn, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):

    def __init__(self, d_model, n_head, d_ff):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, enc_input, src_mask, src_key_padding_mask):
        attn_output = self.norm1(enc_input + self.self_attn(enc_input, enc_input,
                                                            enc_input, src_mask,
                                                            src_key_padding_mask))
        enc_output = self.norm2(attn_output + self.ff(attn_output))
        return enc_output


class Encoder(nn.Module):

    def __init__(self, d_model, n_head, d_ff, n_layer):
        super(Encoder, self).__init__()
        self.encoder_layer = EncoderLayer(d_model, n_head, d_ff)
        self.layers = clone(self.encoder_layer, n_layer)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        # x: [batch_size, src_len, d_model]
        for layer in self.layers:
            x = layer(x, src_mask, src_key_padding_mask)
        return x


class DecoderLayer(nn.Module):

    def __init__(self, d_model, n_head, d_ff):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.enc_dec_attn = MultiHeadAttention(d_model, n_head)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, dec_input, memory, tgt_mask, tgt_key_padding_mask,
                memory_mask, memory_key_padding_mask):
        # dec_input: [batch_size, tgt_len, d_model]
        # memory: [batch_size, src_len, d_model]
        # tgt_mask: [tgt_len, tgt_len]
        # tgt_key_padding_mask: [batch_size, tgt_len]
        # memory_mask: [tgt_len, src_len]
        # memory_key_padding_mask: [batch_size, src_len]
        attn_output = self.norm1(dec_input + self.self_attn(dec_input,
                                                            dec_input,
                                                            dec_input,
                                                            tgt_mask,
                                                            tgt_key_padding_mask))
        enc_dec_out = self.norm2(attn_output + self.enc_dec_attn(attn_output,
                                                                 memory,
                                                                 memory,
                                                                 memory_mask,
                                                                 memory_key_padding_mask
                                                                 ))
        dec_output = self.norm3(enc_dec_out + self.ffn(enc_dec_out))
        return dec_output


class Decoder(nn.Module):

    def __init__(self, d_model, n_head, d_ff, n_layer):
        super(Decoder, self).__init__()
        self.decoder_layer = DecoderLayer(d_model, n_head, d_ff)
        self.layers = clone(self.decoder_layer, n_layer)

    def forward(self, x, enc_output, tgt_mask=None,
                tgt_key_padding_mask=None, memory_mask=None,
                memory_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, tgt_key_padding_mask,
                      memory_mask, memory_key_padding_mask)
        return x


class Transformer(nn.Module):

    def __init__(self, d_model, n_head, d_ff, n_enc_layer, n_dec_layer):
        super(Transformer, self).__init__()
        self.enc_pe = PositionEncode(d_model)
        self.dec_pe = PositionEncode(d_model)
        self.encoder = Encoder(d_model, n_head, d_ff, n_enc_layer)
        self.decoder = Decoder(d_model, n_head, d_ff, n_dec_layer)

    def forward(self, src_embed, tgt_embed, src_mask=None, src_key_padding_mask=None,
                tgt_mask=None, tgt_key_padding_mask=None, memory_mask=None,
                memory_key_padding_mask=None):
        """

        Parameters
        ----------
        src_embed : Tensor
            The embedded sequences for encoder.
            [batch_size, src_lem, d_model].
        tgt_embed : Tensor
            The embedded sequences for decoder.
            [batch_size, tgt_lem, d_model], the embedded sequences.
        src_mask : Tensor, optional
            The mask for the src sequence.
            If not None, [src_len, src_len]. The default is None.
        src_key_padding_mask : ByteTensor or BoolTensor, optional
            The mask for src keys per batch.
            If not None, [batch_size, src_len]. The default is None.
        tgt_mask : Tensor, optional
            The mask for the tgt sequence.
            If not None, [tgt_len, tgt_len]. The default is None.
        tgt_key_padding_mask : ByteTensor or BoolTensor, optional
            The mask for tgt keys per batch.
            If not None, [batch_size, tgt_len]. The default is None.
        memory_mask : Tensor, optional
            The mask for the encoder output.
            If not None, [tgt_len, src_len]. The default is None.
        memory_key_padding_mask : ByteTensor or BoolTensor, optional
            The mask for memory keys per batch.
            If not None, [batch_size, src_len]. The default is None.

        Returns
        -------
        output : Tensor
            The output after transformer.

        """
        src_pe = self.enc_pe(src_embed)
        tgt_pe = self.dec_pe(tgt_embed)
        memory = self.encoder(src_pe, src_mask, src_key_padding_mask)
        output = self.decoder(tgt_pe, memory, tgt_mask,
                              tgt_key_padding_mask, memory_mask,
                              memory_key_padding_mask)
        return output

def generate_triu_mask(seq_len):
    # mask the position at upper tri, the masked position values True
    temp = np.triu(np.ones((seq_len, seq_len)), k=1).astype('uint8')
    return torch.from_numpy(temp) == 1


