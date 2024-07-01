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
        # forward
        self.v_fw = None
        self.z_fw = None
        self.trace_in_fw = None
        self.trace_rec_fw = None
        self.h_fw = None
        self.lowpassz_fw = None
        # backward
        self.z_bw = None
        self.v_bw = None
        self.trace_in_bw = None
        self.trace_rec_bw = None
        self.h_bw = None
        self.lowpassz_bw = None
        self.trace_out = None

        self.z_cat = None
        self.vo = None
        self.n_b = None
        self.L = None
        self.L_fw = None
        self.L_bw = None
        self.l2_reg = 1e-5
        self.dropout = nn.Dropout(0.2)
        # Initial Weight Parameters
        self.w_in_fw = nn.Parameter(torch.Tensor(n_rec, n_in))  # 标记为可训练参数
        self.w_in_bw = nn.Parameter(torch.Tensor(n_rec, n_in))
        self.w_rec_fw = nn.Parameter(torch.Tensor(n_rec, n_rec))
        self.w_rec_bw = nn.Parameter(torch.Tensor(n_rec, n_rec))
        self.w_out = nn.Parameter(torch.Tensor(n_out, n_rec * 2))
        self.reset_parameters(w_init_gain)  # the initialization of weight matrix 何凯明的增益权重
        self.decay = 0.6

        # classifier layer
        self.classifier = nn.Linear(2 * n_rec, n_out)

        self.step = 0
        self.writer = SummaryWriter(log_dir=f'runs/{self.__class__.__name__}')  # 创建一个SummaryWriter对象

        self.writer.add_histogram('w_in_bw', self.w_in_bw, 0)
        self.writer.add_histogram('w_rec_bw', self.w_rec_bw, 0)
        self.writer.add_histogram('w_in_fw', self.w_in_fw, 0)
        self.writer.add_histogram('w_rec_fw', self.w_rec_fw, 0)
        self.writer.add_histogram('w_out', self.w_out, 0)

        self.batch_count = 0

    # 何凯明初始化权重
    def reset_parameters(self, gain):
        # forward
        torch.nn.init.kaiming_normal_(self.w_in_fw)  # 对w_in进行何凯明初始化
        self.w_in_fw.data = gain[0] * self.w_in_fw.data
        torch.nn.init.kaiming_normal_(self.w_rec_fw)
        self.w_rec_fw.data = gain[1] * self.w_rec_fw.data
        # backward
        torch.nn.init.kaiming_normal_(self.w_in_bw)  # 对w_in进行何凯明初始化
        self.w_in_bw.data = gain[0] * self.w_in_bw.data
        torch.nn.init.kaiming_normal_(self.w_rec_bw)
        self.w_rec_bw.data = gain[1] * self.w_rec_bw.data
        # output
        torch.nn.init.kaiming_normal_(self.w_out)
        self.w_out.data = gain[2] * self.w_out.data

    # 初始化网络
    def init_net(self, n_b, n_t, n_rec, n_out):

        # forward propagation
        # Hidden state
        self.v_fw = torch.zeros(n_t, n_b, n_rec).to(self.device)
        # Visible state
        self.z_fw = torch.zeros(n_t, n_b, n_rec).to(self.device)

        # backward propagation
        # Hidden state
        self.v_bw = torch.zeros(n_t, n_b, n_rec).to(self.device)
        # Visible state
        self.z_bw = torch.zeros(n_t, n_b, n_rec).to(self.device)
        self.z_cat = torch.zeros(n_t, n_b, 2 * n_rec).to(self.device)

        self.vo = torch.zeros(n_t, n_b, n_out).to(self.device)

        # Weight gradients
        self.w_in_bw.grad = torch.zeros_like(self.w_in_bw)
        self.w_rec_bw.grad = torch.zeros_like(self.w_rec_bw)

        self.w_in_fw.grad = torch.zeros_like(self.w_in_fw)
        self.w_rec_fw.grad = torch.zeros_like(self.w_rec_fw)

        self.w_out.grad = torch.zeros_like(self.w_out)


    # def exp_convolve(self, tensor, decay):
    #     # 计算衰减因子
    #     t = torch.arange(tensor.size(1), device=tensor.device).float()
    #     decay_factors = decay ** t
    #
    #     # 计算累积和
    #     cumsum = torch.cumsum(tensor * decay_factors.unsqueeze(0).unsqueeze(-1), dim=1)
    #
    #     # 应用衰减因子的倒数
    #     result = cumsum / decay_factors.unsqueeze(0).unsqueeze(-1)
    #
    #     return result


    # @staticmethod
    # def exp_convolve(tensor, decay):
    #     # 优化版本的指数衰减卷积
    #     result = torch.zeros_like(tensor)
    #     result[:, 0] = tensor[:, 0]
    #     for t in range(1, tensor.size(1)):
    #         result[:, t] = result[:, t - 1] * decay + tensor[:, t]
    #     return result



    # 前向传播，返回膜电位vo
    def forward(self, args, x, seq_lengths, yt, do_training, mask):  # output = model(data, onehot_labels_yt, do_training)
        self.n_b = x.shape[0]  # Extracting batch size
        self.n_t = int(seq_lengths.max().item())
        self.init_net(self.n_b, self.n_t, self.n_rec, self.n_out)
        # Network reset 为了消除之前的影响 每次都必须对梯度清0
        self.w_rec_bw *= (1 - torch.eye(self.n_rec, self.n_rec, device=self.device))
        self.w_rec_fw *= (1 - torch.eye(self.n_rec, self.n_rec, device=self.device))
        # 对角线上的为自循环权重 这里使得自循环权重为0 其余位置的权重的值不变
        packed_x = pack_padded_sequence(x.permute(1, 0, 2), seq_lengths, enforce_sorted=False)

        # forward
        x_padded, _ = pad_packed_sequence(packed_x, batch_first=True, padding_value=0.0)
        # backward
        x_padded_bw = torch.flip(x_padded, [1])
        for t in range(self.n_t-1):  # Computing the network state and outputs for the whole sample duration
            # Forward pass - Hidden state:  v: recurrent layer membrane potential
            #                Visible state: z: recurrent layer spike output,
            #                               vo: output layer membrane potential (yo incl. activation function)
            x_t = x_padded[:, t, :].to(self.device)
            self.v_fw[t + 1] = (((self.alpha * self.v_fw[t] + torch.mm(self.z_fw[t], self.w_rec_fw.t()) +
                                torch.mm(x_t, self.w_in_fw.t()))-self.z_fw[t] * self.thr)+torch.randn(self.v_fw[t + 1].shape)
                                .to(self.device))
            self.z_fw[t + 1] = (self.v_fw[t + 1] > self.thr).int()

            x_t_bw = x_padded_bw[:, t, :].to(self.device)
            self.v_bw[t + 1] = (((self.alpha * self.v_bw[t] + torch.mm(self.z_bw[t], self.w_rec_bw.t()) +
                                torch.mm(x_t_bw, self.w_in_bw.t()))-self.z_bw[t] * self.thr)+torch.randn(self.v_bw[t + 1].shape)
                                .to(self.device))
            self.z_bw[t + 1] = (self.v_bw[t + 1] > self.thr).int()
            # 计算vo
            self.z_cat[t + 1] = torch.cat([self.z_fw[t + 1], self.z_bw[self.n_t - t - 2]], dim=1)  # 注意后向的索引
            self.vo[t + 1] = self.kappa * self.vo[t] + torch.mm(self.z_cat[t + 1], self.w_out.t()) + self.b_o



        if do_training:  # 这里的逻辑可以删除
            softmax_vo = F.softmax(self.vo.permute(1, 0, 2), dim=2)  # 对于参与error计算的vo还需要做一个softmax和转置, b , t ,o
            self.grads_batch(args, x, softmax_vo, yt, self.step, mask)  # 对于参与error计算的yt形状是 b t o
        else:
            softmax_vo = F.softmax(self.vo.permute(1, 0, 2), dim=2)

        return softmax_vo, self.vo  # 网络输出为t, b, o

    # 梯度以及学习信号的计算(Batch)
    def grads_batch(self, args, x, yo, yt, step, mask):  # 用于梯度计算的辅助量
        v_scale_fw = ((self.v_fw - self.thr) / self.thr)
        v_scale_bw = ((self.v_bw - self.thr) / self.thr)
        v_scale_clamp_fw = torch.clamp(v_scale_fw, min=-20, max=20)
        v_scale_clamp_bw = torch.clamp(v_scale_bw, min=-20, max=20)
        # Surrogate derivative
        if args.surro_deri == 'linear':
            self.h_fw = self.gamma * torch.max(torch.zeros_like(self.v_fw), 1 - torch.abs(v_scale_fw))
            self.h_bw = self.gamma * torch.max(torch.zeros_like(self.v_bw), 1 - torch.abs(v_scale_bw))
        else:
            self.h_fw = self.gamma / ((0.001 * torch.abs(self.v_fw - self.thr) + 1) ** 2)
            self.h_bw = self.gamma / ((0.001 * torch.abs(self.v_bw - self.thr) + 1) ** 2)

        # # liang sigmoid
        # self.h = self.gamma / ((0.001 * torch.abs(self.v - self.thr) + 1)**2)
        # fast sigmoid
        # self.h = self.gamma * ((self.v - self.thr) / self.thr) / (1 + torch.abs((self.v - self.thr) / self.thr))
        # ATan
        # self.h = (self.gamma / self.thr) * (1 - torch.atan((self.v - self.thr) / self.thr) ** 2)

        h_compare_fw = 1 - torch.abs((self.v_fw - self.thr) / self.thr)
        h_compare_bw = 1 - torch.abs((self.v_bw - self.thr) / self.thr)

        # 使用SummaryWriter记录v_scale和h_compare的直方图
        self.writer.add_histogram('v_scale_fw', v_scale_fw, step)
        self.writer.add_histogram('h_compare_fw', h_compare_fw, step)
        self.writer.add_histogram('h_fw', self.h_fw, step)
        self.writer.add_histogram('v_scale_clamp_fw', v_scale_clamp_fw, step)

        self.writer.add_histogram('v_scale_bw', v_scale_bw, step)
        self.writer.add_histogram('h_compare_bw', h_compare_bw, step)
        self.writer.add_histogram('h_bw', self.h_bw, step)
        self.writer.add_histogram('v_scale_clamp_bw', v_scale_clamp_bw, step)

        # 计算L2正则化项
        l2_loss_in_fw = self.l2_reg * torch.sum(self.w_in_fw ** 2)
        l2_loss_rec_fw = self.l2_reg * torch.sum(self.w_rec_fw ** 2)
        l2_loss_in_bw = self.l2_reg * torch.sum(self.w_in_bw ** 2)
        l2_loss_rec_bw = self.l2_reg * torch.sum(self.w_rec_bw ** 2)

        # 计算L1正则化项
        l1_loss_in_fw = self.l2_reg * torch.sum(torch.abs(self.w_in_fw))
        l1_loss_rec_fw = self.l2_reg * torch.sum(torch.abs(self.w_rec_fw))
        l1_loss_in_bw = self.l2_reg * torch.sum(torch.abs(self.w_in_bw))
        l1_loss_rec_bw = self.l2_reg * torch.sum(torch.abs(self.w_rec_bw))

        # conv_kernel
        alpha_conv = torch.tensor([self.alpha ** (self.n_t - i - 1) for i in range(self.n_t)])\
            .float().view(1, 1, -1).to(self.device)
        kappa_conv = torch.tensor([self.kappa ** (self.n_t - i - 1) for i in range(self.n_t)])\
            .float().view(1, 1, -1).to(self.device)

        mask = mask.unsqueeze(1).unsqueeze(2).float()  # 创建掩码,忽略填充部分

        x_bw = torch.flip(x, [1])
        mask_bw = torch.flip(mask, [1])

        # compute input eligibility trace
        # shape n_b, n_rec, n_in , n_t
        self.trace_in_fw = F.conv1d(x.permute(0, 2, 1), alpha_conv.expand(self.n_in, -1, -1), padding=self.n_t,
                                    groups=self.n_in)[:, :, 1:self.n_t + 1].unsqueeze(1).expand(-1, self.n_rec, -1, -1)
        self.trace_in_fw = torch.einsum('tbr,brit->brit', self.h_fw, self.trace_in_fw * mask.to(self.device))  # STE * ET

        self.trace_in_fw = F.conv1d(self.trace_in_fw.reshape(self.n_b, self.n_in * self.n_rec, self.n_t),
                                    kappa_conv.expand(self.n_in * self.n_rec, -1, -1), padding=self.n_t,
                                    groups=self.n_in * self.n_rec)[:, :, 1:self.n_t + 1].reshape(self.n_b, self.n_rec,
                                                                                                 self.n_in, self.n_t)
        # backward
        self.trace_in_bw = F.conv1d(x_bw.permute(0, 2, 1), alpha_conv.expand(self.n_in, -1, -1), padding=self.n_t,
                                    groups=self.n_in)[:, :, 1:self.n_t + 1].unsqueeze(1).expand(-1, self.n_rec, -1, -1)
        self.trace_in_bw = torch.einsum('tbr,brit->brit', self.h_bw, self.trace_in_bw * mask_bw.to(self.device))  # STE * ET

        self.trace_in_bw = F.conv1d(self.trace_in_bw.reshape(self.n_b, self.n_in * self.n_rec, self.n_t),
                                    kappa_conv.expand(self.n_in * self.n_rec, -1, -1), padding=self.n_t,
                                    groups=self.n_in * self.n_rec)[:, :, 1:self.n_t + 1].reshape(self.n_b, self.n_rec,
                                                                                                 self.n_in, self.n_t)

        # compute recurrent eligibility trace
        # shape: n_b, n_rec, n_rec, n_t
        # forward
        self.trace_rec_fw = F.conv1d(self.z_fw.permute(1, 2, 0), alpha_conv.expand(self.n_rec, -1, -1), padding=self.n_t,
                                     groups=self.n_rec)[:, :, :self.n_t].unsqueeze(1).expand(-1, self.n_rec, -1, -1)

        self.trace_rec_fw = torch.einsum('tbr,brit->brit', self.h_fw, self.trace_rec_fw * mask.to(self.device))  # STE * ET

        self.lowpassz_fw = self.trace_rec_fw

        self.trace_rec_fw = F.conv1d(self.trace_rec_fw.reshape(self.n_b, self.n_rec * self.n_rec, self.n_t),
                                     kappa_conv.expand(self.n_rec * self.n_rec, -1, -1), padding=self.n_t,
                                     groups=self.n_rec * self.n_rec)[:, :, 1:self.n_t + 1].reshape(self.n_b, self.n_rec,
                                                                                                   self.n_rec, self.n_t)

        # backward
        self.trace_rec_bw = F.conv1d(self.z_bw.permute(1, 2, 0), alpha_conv.expand(self.n_rec, -1, -1), padding=self.n_t,
                                     groups=self.n_rec)[:, :, :self.n_t].unsqueeze(1).expand(-1, self.n_rec, -1, -1)

        self.trace_rec_bw = torch.einsum('tbr,brit->brit', self.h_bw, self.trace_rec_bw * mask_bw.to(self.device))  # STE * ET

        self.lowpassz_bw = self.trace_rec_bw

        self.trace_rec_bw = F.conv1d(self.trace_rec_bw.reshape(self.n_b, self.n_rec * self.n_rec, self.n_t),
                                     kappa_conv.expand(self.n_rec * self.n_rec, -1, -1), padding=self.n_t,
                                     groups=self.n_rec * self.n_rec)[:, :, 1:self.n_t + 1].reshape(self.n_b, self.n_rec,
                                                                                                   self.n_rec, self.n_t)

        # compute output eligibility trace
        # shape: n_b, n_rec, n_t
        self.trace_out = F.conv1d(self.z_cat.permute(1, 2, 0), kappa_conv.expand(self.n_rec * 2, -1, -1), padding=self.n_t,
                                  groups=self.n_rec * 2)[:, :, 1:self.n_t + 1]


        # compute the error
        err = yo - yt

        # Learning Signal: error * w_out
        self.L = torch.einsum('tbo,or->brt', err.permute(1, 0, 2), self.w_out)
        self.L_fw = self.L[:, :self.n_rec, :]
        self.L_bw = self.L[:, self.n_rec:, :]

        # compute the gradient
        self.w_in_fw.grad += self.lr_layer[0] * torch.sum(self.L_fw.unsqueeze(2).expand(-1, -1, self.n_in, -1) * self.trace_in_fw,
                                                          dim=(0, 3))
        self.w_in_bw.grad += self.lr_layer[0] * torch.sum(self.L_bw.unsqueeze(2).expand(-1, -1, self.n_in, -1) * self.trace_in_bw,
                                                          dim=(0, 3))
        # print(f"w_in  is{self.w_in}")
        self.w_rec_fw.grad += self.lr_layer[1] * torch.sum(self.L_fw.unsqueeze(2).expand(-1, -1, self.n_rec, -1) * self.trace_rec_fw,
                                                           dim=(0, 3))
        self.w_rec_bw.grad += self.lr_layer[1] * torch.sum(self.L_bw.unsqueeze(2).expand(-1, -1, self.n_rec, -1) * self.trace_rec_bw,
                                                           dim=(0, 3))
        self.w_out.grad += self.lr_layer[2] * torch.einsum('tbo,brt->or', err.permute(1, 0, 2), self.trace_out)
        # print(f"grad is{self.w_rec.grad}")
        # print(f"w_rec is{self.w_rec}")
        # print(f"差值是 {self.w_rec - self.w_rec.grad}")
        # self.w_out.grad += self.lr_layer[2] * torch.einsum('tbo,brt->or', err.permute(1, 0, 2), self.trace_out)


    def __repr__(self):

        return self.__class__.__name__ + ' (' \
            + str(self.n_in) + ' -> ' \
            + str(self.n_rec) + ' -> ' \
            + str(self.n_out) + ') '


