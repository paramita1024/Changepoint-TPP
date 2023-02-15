import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.transformer.Constants as Constants
from models.transformer.Layers import EncoderLayer


def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(
        1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(
        0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask


def get_future_mask(seq, future_of_THP):
    sz_b, len_s = seq.size()
    mask = torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8)
    for i in range(len_s):
        mask[i, i+1:min(i+future_of_THP+1, len_s)] = 0
    mask = mask.unsqueeze(0).expand(sz_b, -1, -1)
    return mask

class Feed_Forward(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer1 = nn.Linear(in_features=in_dim, out_features=in_dim)
        self.layer2 = nn.Linear(in_features=in_dim, out_features=out_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        return self.layer2(x)

    def initialize(self):
        nn.init.xavier_normal_(self.layer1.weight)
        nn.init.xavier_normal_(self.layer2.weight)

class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            num_types, dim_of_THP, dim_inner_of_THP,
            num_layers_of_THP, num_head_of_THP, dim_k_of_THP, dim_v_of_THP, dropout, future_of_THP,  device):
        super().__init__()

        self.dim_of_THP = dim_of_THP
        self.future_of_THP = future_of_THP

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / dim_of_THP)
             for i in range(dim_of_THP)],
            device=device)

        # event type embedding
        self.event_emb = nn.Embedding(
            num_types + 1, dim_of_THP, padding_idx=Constants.PAD)

        self.layer_stack1 = nn.ModuleList([
            EncoderLayer(dim_of_THP, dim_inner_of_THP, num_head_of_THP, dim_k_of_THP, dim_v_of_THP,
                         dropout=dropout, normalize_before=False)
            for _ in range(num_layers_of_THP)])

        # self.layer_stack2 = nn.ModuleList([
        #     EncoderLayer(dim_of_THP, dim_inner_of_THP, num_head_of_THP, dim_k_of_THP, dim_v_of_THP,
        #                  dropout=dropout, normalize_before=False)
        #     for _ in range(num_layers_of_THP)])

        # self.enc_layer1 = EncoderLayer(dim_of_THP, dim_inner_of_THP, num_head_of_THP, dim_k_of_THP, dim_v_of_THP, dropout=dropout, normalize_before=False)
        # self.enc_layer2 = EncoderLayer(dim_of_THP, dim_inner_of_THP, num_head_of_THP, dim_k_of_THP, dim_v_of_THP, dropout=dropout, normalize_before=False)

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*dim_of_THP.
        """

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, event_time, event_type, non_pad_mask):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(event_type)
        slf_attn_mask_keypad = get_attn_key_pad_mask(
            seq_k=event_type, seq_q=event_type)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(
            slf_attn_mask_subseq)
        slf_attn_mask_past = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        slf_attn_mask_future = get_future_mask(event_type, self.future_of_THP)
        slf_attn_mask_future = (slf_attn_mask_keypad + slf_attn_mask_future).gt(0)

        tem_enc = self.temporal_enc(event_time, non_pad_mask)
        enc_output = self.event_emb(event_type)
        # enc_output += tem_enc

        # enc_output_V,_ = self.enc_layer1(enc_output, non_pad_mask = non_pad_mask, slf_attn_mask = slf_attn_mask_past)
        # enc_output_W,_ = self.enc_layer2(enc_output, non_pad_mask = non_pad_mask, slf_attn_mask = slf_attn_mask_future)

        enc_output_V = enc_output
        for enc_layer in self.layer_stack1:
            enc_output_V += tem_enc
            enc_output_V, _ = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask_past
            )

        # enc_output_W = enc_output.clone()
        # for enc_layer in self.layer_stack2:
        #     enc_output_W += tem_enc
        #     enc_output_W, _ = enc_layer(
        #         enc_output,
        #         non_pad_mask=non_pad_mask,
        #         slf_attn_mask=slf_attn_mask_future
        #     )

        return enc_output_V


class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super().__init__()
        self.linear_type = nn.Linear(dim, num_types, bias=False)

    def forward(self, data, non_pad_mask):
        out = self.linear_type(data)
        out = out * non_pad_mask
        return out

    def initialize(self):
        nn.init.xavier_normal_(self.linear_type.weight)

class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            num_types, dim_of_THP=256, dim_inner_of_THP=1024,
            num_layers_of_THP=4, num_head_of_THP=4, dim_k_of_THP=64, \
            dim_v_of_THP=64, dropout=0.1, \
            future_of_THP=10, \
            device=torch.device('cuda'), \
            len_feat=1):
        # check

        super().__init__()

        self.encoder = Encoder(
            num_types=num_types,
            dim_of_THP=dim_of_THP,
            dim_inner_of_THP=dim_inner_of_THP,
            num_layers_of_THP=num_layers_of_THP,
            num_head_of_THP=num_head_of_THP,
            dim_k_of_THP=dim_k_of_THP,
            dim_v_of_THP=dim_v_of_THP,
            dropout=dropout,
            future_of_THP=future_of_THP,
            device=device,
        )

        self.num_types = num_types

        # self.linear = nn.Linear(dim_of_THP, num_types)
        # self.linear1 = nn.Linear(dim_of_THP, num_types)
        # self.linear2 = nn.Linear(dim_of_THP, num_types)
        # self.alpha1 = nn.Parameter(torch.tensor(-0.1))
        # self.beta1 = nn.Parameter(torch.tensor(1.0))
        # self.linear2 = nn.Linear(dim_of_THP, num_types)
        # self.alpha2 = nn.Parameter(torch.tensor(-0.1))
        # self.beta2 = nn.Parameter(torch.tensor(1.0))
        
        self.linear_lambdas = Feed_Forward(dim_of_THP, 1)
        self.time_predictor = Predictor(dim_of_THP, 1)
        self.type_predictor = Predictor(dim_of_THP+len_feat, num_types)
        
        # self.initialize()

    def initialize(self):
        
        self.linear_lambdas.initialize()
        self.time_predictor.initialize()
        self.type_predictor.initialize()

    def forward(self, event_time, event_type, event_feat):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """

        non_pad_mask = get_non_pad_mask(event_type)
        V = self.encoder(event_time, event_type, non_pad_mask)
        # enc_output = self.rnn(enc_output, non_pad_mask)

        # useful for baselines
        lambdas = nn.Softplus()(self.linear_lambdas(V))
        time_prediction = self.time_predictor(V, non_pad_mask)
        
        type_prediction = self.type_predictor(\
            torch.cat((V, event_feat),dim=-1), non_pad_mask)
        type_prediction = F.softmax(type_prediction,dim=-1)

        return V, non_pad_mask, (lambdas, time_prediction, type_prediction)
