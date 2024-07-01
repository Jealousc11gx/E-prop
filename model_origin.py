# this file is used to define the model
# Author： Chen Linliang
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.optim import Adam  # 导入Adam优化器
from snntorch.surrogate import  fast_sigmoid, atan, straight_through_estimator
import matplotlib.pyplot as plt


class SRNN(nn.Module):

    def __init__(self, n_in, n_rec, n_out, n_t, thr, tau_m, tau_o, b_o, gamma, dt, model, classif, w_init_gain,
                 lr_layer, device):

        super(SRNN, self).__init__()
        self.n_in = n_in  # input neuron
        self.n_rec = n_rec  # rec neuron
        self.n_out = n_out  # output neuron
        self.n_t = None  # changeable time step
        self.thr = thr  # threshold voltage
        self.dt = dt  # time step
        self.alpha = np.exp(-dt / tau_m)  # 循环层膜电位衰减因子 tau_m是时间常数
        self.kappa = np.exp(-dt / tau_o)  # 输出层膜电位衰减因子 tau_o是时间常数
        self.gamma = gamma  # Surrogate derivative magnitude parameter   γpd pseudo derivative
        self.b_o = b_o  # the bias of output
        self.model = model  # LIF
        self.classif = classif  # classification
        self.lr_layer = lr_layer  # learning layer parameters
        self.device = device
        self.v = None
        self.vo = None
        self.z = None
        self.n_b = None
        self.L = None
        self.h = None
        self.trace_in = None
        self.trace_out = None
        self.trace_rec = None
        self.lowpassz = None
        self.l2_reg = 1e-5
        self.dropout = nn.Dropout(0.2)
        # Initial Weight Parameters
        self.w_in = nn.Parameter(torch.Tensor(n_rec, n_in))  # 标记为可训练参数
        self.w_rec = nn.Parameter(torch.Tensor(n_rec, n_rec))
        self.w_out = nn.Parameter(torch.Tensor(n_out, n_rec))
        self.reset_parameters(w_init_gain)  # the initialization of weight matrix 何凯明的增益权重

        # 最后一层使用自动更新
        # self.optimizer_out = Adam([self.w_out], lr=lr_layer[2])

        # self.bn_in = nn.BatchNorm1d(n_in)
        # self.bn_rec = nn.BatchNorm1d(n_rec)

        self.step = 0
        self.writer = SummaryWriter(log_dir=f'runs/{self.__class__.__name__}')  # 创建一个SummaryWriter对象

        self.writer.add_histogram('w_in', self.w_in, 0)
        self.writer.add_histogram('w_rec', self.w_rec, 0)
        self.writer.add_histogram('w_out', self.w_out, 0)

        self.batch_count = 0

    # 何凯明初始化权重
    def reset_parameters(self, gain):

        torch.nn.init.kaiming_normal_(self.w_in)  # 对w_in进行何凯明初始化
        self.w_in.data = gain[0] * self.w_in.data
        torch.nn.init.kaiming_normal_(self.w_rec)
        self.w_rec.data = gain[1] * self.w_rec.data
        torch.nn.init.kaiming_normal_(self.w_out)
        self.w_out.data = gain[2] * self.w_out.data

    # 初始化网络
    def init_net(self, n_b, n_t, n_rec, n_out):
        # Hidden state
        self.v = torch.zeros(n_t, n_b, n_rec).to(self.device)
        # Visible state
        self.vo = torch.zeros(n_t, n_b, n_out).to(self.device)
        self.z = torch.zeros(n_t, n_b, n_rec).to(self.device)
        # Weight gradients
        self.w_in.grad = torch.zeros_like(self.w_in)
        self.w_rec.grad = torch.zeros_like(self.w_rec)
        self.w_out.grad = torch.zeros_like(self.w_out)

    # 前向传播，返回膜电位vo
    def forward(self, args, x, seq_lengths, yt, do_training, mask):  # output = model(data, onehot_labels_yt, do_training)
        self.n_b = x.shape[0]  # Extracting batch size
        self.n_t = int(seq_lengths.max().item())
        self.init_net(self.n_b, self.n_t, self.n_rec, self.n_out)
        # Network reset 为了消除之前的影响 每次都必须对梯度清0
        self.w_rec *= (1 - torch.eye(self.n_rec, self.n_rec, device=self.device))
        # 对角线上的为自循环权重 这里使得自循环权重为0 其余位置的权重的值不变
        packed_x = pack_padded_sequence(x.permute(1, 0, 2), seq_lengths, enforce_sorted=False)
        x_padded, _ = pad_packed_sequence(packed_x, batch_first=True, padding_value=0.0)
        # whole sample duration
        for t in range(self.n_t-1):  # Computing the network state and outputs for the whole sample duration
            # Forward pass - Hidden state:  v: recurrent layer membrane potential
            #                Visible state: z: recurrent layer spike output,
            #                               vo: output layer membrane potential (yo incl. activation function)
            x_t = x_padded[:, t, :].to(self.device)
            # x_t = self.bn_in(x_t)
            # x_t = self.dropout(x_t)

            self.v[t + 1] = ((self.alpha * self.v[t] + torch.mm(self.z[t], self.w_rec.t()) +
                              torch.mm(x_t, self.w_in.t()))-self.z[t] * self.thr)+torch.randn(self.v[t + 1].shape).to(self.device)

            # self.v[t + 1] = self.bn_rec(self.v[t + 1])
            self.z[t + 1] = (self.v[t + 1] > self.thr).int()
            self.vo[t + 1] = self.kappa * self.vo[t] + torch.mm(self.z[t+1], self.w_out.t()) + self.b_o


        #  if self.classif:  # 目前都是分类问题 (n_b, n_t, n_out)
        #     yo = self.vo
        #  else:  # prediction
        #     yo = self.vo  # (n_b, n_t, n_out)

        if do_training:
            # 训练的话 需要梯度更新
            # with torch.enable_grad():
            # self.optimizer_out.zero_grad()
            # loss = nn.functional.cross_entropy(self.vo.permute(1, 0, 2), yt.argmax(dim=2), reduction='none')
            # masked_loss = (loss * mask.float()).sum() / mask.sum()
            # masked_loss.backward()
            # self.optimizer_out.step()
            softmax_vo = F.softmax(self.vo.permute(1, 0, 2), dim=2)  # 对于参与error计算的vo还需要做一个softmax和转置, b , t ,o
            self.grads_batch(args, x, softmax_vo, yt, self.step, mask)  # 对于参与error计算的yt形状是 b t o
        else:
            softmax_vo = F.softmax(self.vo.permute(1, 0, 2), dim=2)

        return softmax_vo, self.vo  # 网络输出为t, b, o

    # 梯度以及学习信号的计算(Batch)
    def grads_batch(self, args, x, yo, yt, step, mask):  # 用于梯度计算的辅助量
        v_scale = ((self.v - self.thr) / self.thr)
        v_scale_clamp = torch.clamp(v_scale, min=-20, max=20)
        # Surrogate derivative
        if args.surro_deri == 'linear':
            self.h = self.gamma * torch.max(torch.zeros_like(self.v), 1 - torch.abs(v_scale))
        else:
            self.h = self.gamma / ((0.001 * torch.abs(self.v - self.thr) + 1)**2)

        # # liang sigmoid
        # self.h = self.gamma / ((0.001 * torch.abs(self.v - self.thr) + 1)**2)
        # fast sigmoid
        # self.h = self.gamma * ((self.v - self.thr) / self.thr) / (1 + torch.abs((self.v - self.thr) / self.thr))
        # ATan
        # self.h = (self.gamma / self.thr) * (1 - torch.atan((self.v - self.thr) / self.thr) ** 2)

        h_compare = 1 - torch.abs((self.v - self.thr) / self.thr)

        # 使用SummaryWriter记录v_scale和h_compare的直方图
        self.writer.add_histogram('v_scale', v_scale, step)
        self.writer.add_histogram('h_compare', h_compare, step)
        self.writer.add_histogram('h', self.h, step)
        self.writer.add_histogram('v_scale_clamp', v_scale_clamp, step)

        # 计算L2正则化项
        l2_loss_in = self.l2_reg * torch.sum(self.w_in ** 2)
        l2_loss_rec = self.l2_reg * torch.sum(self.w_rec ** 2)
        l2_loss_out = self.l2_reg * torch.sum(self.w_out ** 2)

        # 计算L1正则化项
        l1_loss_in = self.l2_reg * torch.sum(torch.abs(self.w_in))
        l1_loss_rec = self.l2_reg * torch.sum(torch.abs(self.w_rec))
        l1_loss_out = self.l2_reg * torch.sum(torch.abs(self.w_out))

        # conv_kernel
        alpha_conv = torch.tensor([self.alpha ** (self.n_t - i - 1) for i in range(self.n_t)])\
            .float().view(1, 1, -1).to(self.device)
        kappa_conv = torch.tensor([self.kappa ** (self.n_t - i - 1) for i in range(self.n_t)])\
            .float().view(1, 1, -1).to(self.device)
        mask = mask.unsqueeze(1).unsqueeze(2).float()  # 创建掩码,忽略填充部分

        # compute input eligibility trace
        # shape n_b, n_rec, n_in , n_t
        self.trace_in = F.conv1d(x.permute(0, 2, 1), alpha_conv.expand(self.n_in, -1, -1), padding=self.n_t,
                                 groups=self.n_in)[:, :, 1:self.n_t + 1].unsqueeze(1).expand(-1, self.n_rec, -1, -1)
        self.trace_in = torch.einsum('tbr,brit->brit', self.h, self.trace_in * mask.to(self.device))  # STE * ET

        self.trace_in = F.conv1d(self.trace_in.reshape(self.n_b, self.n_in * self.n_rec, self.n_t),
                                 kappa_conv.expand(self.n_in * self.n_rec, -1, -1), padding=self.n_t,
                                 groups=self.n_in * self.n_rec)[:, :, 1:self.n_t + 1].reshape(self.n_b, self.n_rec,
                                                                                              self.n_in, self.n_t)

        # compute recurrent eligibility trace
        # shape: n_b, n_rec, n_rec, n_t
        self.trace_rec = F.conv1d(self.z.permute(1, 2, 0), alpha_conv.expand(self.n_rec, -1, -1), padding=self.n_t,
                                  groups=self.n_rec)[:, :, :self.n_t].unsqueeze(1).expand(-1, self.n_rec, -1, -1)

        self.trace_rec = torch.einsum('tbr,brit->brit', self.h, self.trace_rec * mask.to(self.device))  # STE * ET

        self.lowpassz = self.trace_rec

        self.trace_rec = F.conv1d(self.trace_rec.reshape(self.n_b, self.n_rec * self.n_rec, self.n_t),
                                  kappa_conv.expand(self.n_rec * self.n_rec, -1, -1), padding=self.n_t,
                                  groups=self.n_rec * self.n_rec)[:, :, 1:self.n_t + 1].reshape(self.n_b, self.n_rec,
                                                                                                self.n_rec, self.n_t)

        # compute output eligibility trace
        # shape: n_b, n_rec, n_t
        self.trace_out = F.conv1d(self.z.permute(1, 2, 0), kappa_conv.expand(self.n_rec, -1, -1), padding=self.n_t,
                                  groups=self.n_rec)[:, :, 1:self.n_t + 1]


        # compute the error
        err = yo - yt

        # Learning Signal: error * w_out
        self.L = torch.einsum('tbo,or->brt', err.permute(1, 0, 2), self.w_out)

        # compute the gradient
        self.w_in.grad += self.lr_layer[0] * torch.sum(self.L.unsqueeze(2).expand(-1, -1, self.n_in, -1) * self.trace_in,
                                                       dim=(0, 3))
        # print(f"w_in  is{self.w_in}")
        self.w_rec.grad += self.lr_layer[1] * torch.sum(self.L.unsqueeze(2).expand(-1, -1, self.n_rec, -1) * self.trace_rec,
                                                        dim=(0, 3))
        # print(f"grad is{self.w_rec.grad}")
        # print(f"w_rec is{self.w_rec}")
        # print(f"差值是 {self.w_rec - self.w_rec.grad}")
        self.w_out.grad += self.lr_layer[2] * torch.einsum('tbo,brt->or', err.permute(1, 0, 2), self.trace_out)


    def __repr__(self):

        return self.__class__.__name__ + ' (' \
            + str(self.n_in) + ' -> ' \
            + str(self.n_rec) + ' -> ' \
            + str(self.n_out) + ') '


