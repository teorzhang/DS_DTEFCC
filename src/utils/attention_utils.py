# -*- coding: utf-8 -*-
# coding=utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

'''门控'''
class Gate(nn.Module):
    def __init__(self, question_dim, answer_dim):
        super(Gate, self).__init__()

        self.question_linear = nn.Linear(question_dim, question_dim)
        self.answer_linear = nn.Linear(answer_dim, answer_dim)
        self.q_a_linear = nn.Linear(question_dim+answer_dim, question_dim)

        self.d_k = question_dim

    def forward(self, question_dim, answer_dim):

        question1 = self.question_linear(question_dim)
        answer1 = self.answer_linear(answer_dim)

        q_a = torch.cat((question1, answer1), dim=2) #[bs,seq,768+768]
        g_st = torch.sigmoid(self.q_a_linear(q_a)) #[bs,q+a,768]

        hidd = question_dim * g_st + answer_dim * (1-g_st) #[bs,q+a,768]
        return hidd

class MLP(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre = True, layer_num = 2, dropout = True, **kwargs):
        super(MLP, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        self.prelu = nn.PReLU()
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim, bias = True)
        else:
            self.linear_first = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.linear_out = nn.Linear(feature_dim, output_dim, bias = True)

    def forward(self, data,device):

        if self.feature_pre:
            x = self.linear_pre(data)

        x = self.prelu(x)
        for i in range(self.layer_num - 2):
            x = self.linear_hidden[i](x)
            x = F.tanh(x)
            if self.dropout:
                x = F.dropout(x, training = self.training)
        x = self.linear_out(x)
        x = F.normalize(x, p = 2, dim = -1)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads  # 每个注意头的维度
        self.h = heads

        # 参数矩阵
        self.q_linear = nn.Linear(d_model, d_model)  # 这里把W的维度都变长了，等价于多组w
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        self.scores = None

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # 注意力的公式
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)  # [PAD] 位置置位无穷小
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        self.scores = scores  # (batch, 4, 30, 30)
        output = torch.matmul(scores, v)  # 输出为注意力矩阵与值矩阵的乘积
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)  # batch size

        # perform linear operation and split into h heads 分头
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

    def get_scores(self):
        return self.scores
