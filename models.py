# this file is used to define the model
# Author： Chen Linliang
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.parameter import Parameter
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import mfcc_normalized
import matplotlib.pyplot as plt


class SRNN(nn.Module):

    def __init__(self, n_in, n_rec, n_out, n_t, thr, tau_m, tau_o, b_o, gamma, dt, model, classif, w_init_gain,
                 lr_layer, device):

        super(SRNN, self).__init__()
        self.n_in = n_in  # input neuron
        self.n_rec = n_rec  # rec neuron
        self.n_out = n_out  # output neuron
        self.n_t = n_t  # number of steps 11
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
        # Initial Weight Parameters
        self.w_in = Parameter(torch.Tensor(n_rec, n_in))  # 标记为可训练参数
        self.w_rec = Parameter(torch.Tensor(n_rec, n_rec))
        self.w_out = Parameter(torch.Tensor(n_out, n_rec))
        self.reset_parameters(w_init_gain)  # the initialization of weight matrix 何凯明的增益权重

        self.bn_in = nn.BatchNorm1d(n_in)
        self.bn_rec = nn.BatchNorm1d(n_rec)

    # 何凯明初始化权重
    def reset_parameters(self, gain):

        torch.nn.init.kaiming_normal_(self.w_in)  # 对w_in进行何凯明初始化
        self.w_in.data = gain[0] * self.w_in.data  # 0.5 增益系数越高 其标准差也越高 同时这里的增益系数和选用的激活函数也有关系
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
        # print(f"\t shape of vo is {self.vo.shape}")
        # print(f"\t shape of z is {self.z.shape}")
        # Weight gradients
        self.w_in.grad = torch.zeros_like(self.w_in)
        self.w_rec.grad = torch.zeros_like(self.w_rec)
        self.w_out.grad = torch.zeros_like(self.w_out)

    # 前向传播，返回膜电位vo
    def forward(self, x, yt, do_training):  # output = model(data, onehot_labels_yt, do_training)
        # print(f"do_training is {do_training}")
        # print(f"forward data input x is {x}")
        # print(f"forward data input x shape is {x.shape}")
        self.n_b = x.shape[0]  # Extracting batch size
        # print(f"n_b is {self.n_b}")
        # print(f"reference output yt is {yt}")
        # print(f"reference output yt shape is{yt.shape}")
        self.init_net(self.n_b, self.n_t, self.n_rec, self.n_out)
        # Network reset 为了消除之前的影响 每次都必须对梯度清0
        self.w_rec *= (1 - torch.eye(self.n_rec, self.n_rec, device=self.device))
        # 对角线上的为自循环权重 这里使得自循环权重为0 其余位置的权重的值不变
        for t in range(self.n_t-1):  # Computing the network state and outputs for the whole sample duration
            # Forward pass - Hidden state:  v: recurrent layer membrane potential
            #                Visible state: z: recurrent layer spike output,
            #                               vo: output layer membrane potential (yo incl. activation function)

            x_t = x[:, t, :].to(self.device)
            x_t = self.bn_in(x_t)
            # print(f"shape of x_t is {x_t.shape}")
            # 在需要使用特征转换和可视化的地方调用相应的函数

            # log_transformed_x = mfcc_normalized.log_transform_with_bias(x_t, bias=1.0)
            # min_max_normalized_x = mfcc_normalized.min_max_normalize_with_scale(x_t, scale_factor=1.0)
            # z_score_normalized_x = mfcc_normalized.z_score_normalize_with_bias(x_t, bias=1.0)
            #
            # pause_interval = 5
            # display_indices = [0]  # 指定要显示的数据序号
            # if t == 1:
            #     mfcc_normalized.visualize_features(x_t, log_transformed_x, min_max_normalized_x, z_score_normalized_x,
            #                                    pause_interval=pause_interval, display_indices=display_indices)

            # print(f"t step is {t}\n  step t x_t is{x_t}, x_t shape is{x_t.shape}")
            self.v[t + 1] = ((self.alpha * self.v[t] + torch.mm(self.z[t], self.w_rec.t()) + torch.mm(x_t, self.w_in.t()))-self.z[t] * self.thr)  # 衰减因子乘电压加上递归的影响再加上输入的影响
            self.v[t + 1] = self.bn_rec(self.v[t + 1])
            self.z[t + 1] = (self.v[t + 1] > self.thr).float()

            self.vo[t + 1] = self.kappa * self.vo[t] + torch.mm(self.z[t + 1], self.w_out.t()) + self.b_o

            # print(f"v[{t}] is {self.v[t]}")
            # print(f"z[{t}] is {self.z[t]}")
            # print(f"vo[{t}] is {self.vo[t]}")
            # print(f"v[10] is {self.v[t+1]}")
            # print(f"z[10] is {self.z[t+1]}")
            # print(f"vo[10] is {self.vo[t+1]}")
            # print(f"v[{t}] shape is {self.v[t].shape}")
            # print(f"z[{t}] shape is {self.z[t].shape}")
            # print(f"vo[{t}] shape is {self.vo[t].shape}")

        #  if self.classif:  # 目前都是分类问题 (n_b, n_t, n_out)
        #     yo = self.vo
        #  else:  # prediction
        #     yo = self.vo  # (n_b, n_t, n_out)

        if do_training:
            # 训练的话 需要梯度更新
            # with torch.no_grad()
            # for t in range(self.n_t):
                # print(f"x[{t}] is {x[:, t, :]}")
                # print(f"v[{t}] is {self.v[t]}")
                # print(f"z[{t}] is {self.z[t]}")
                # print(f"vo[{t}] is {self.vo[t]}")
                # print(f"v[{t}] shape is {self.v[t].shape}")
                # print(f"z[{t}] shape is {self.z[t].shape}")
                # print(f"vo[{t}] shape is {self.vo[t].shape}")
            softmax_vo = F.softmax(self.vo.permute(1, 0, 2), dim=2)  # 对于参与error计算的vo还需要做一个softmax和转置, b , t ,o
            # print(f"softmax_yo  is {softmax_yo}")
            # print(f"softmax_yo shape  is {softmax_yo.shape}")
            # print(f"yo is {self.vo}")
            # print(f"yo shape is {self.vo.shape}")
            # print(f"actual output is {torch.argmax(self.vo, dim=2)}")
            # print(f"reference output is {torch.argmax(yt, dim=2)}")
            self.grads_batch(x, softmax_vo, yt)  # 对于参与error计算的yt形状是 b t o

        return softmax_vo, self.vo  # 网络输出为t, b, o

    # 梯度以及学习信号的计算(Batch)
    def grads_batch(self, x, yo, yt):  # 用于梯度计算的辅助量

        # Surrogate derivative
        self.h = self.gamma * torch.max(torch.zeros_like(self.v), 1 - torch.abs((self.v - self.thr) / self.thr))
        # print(f"\t shape of h is {self.h.shape}")
        # print(f"\t  h is {self.h}")

        # conv_kernel
        alpha_conv = torch.tensor([self.alpha ** (self.n_t - i - 1) for i in range(self.n_t)])\
            .float().view(1, 1, -1).to(self.device)  # compose a tensor which shape is [1,1,11]
        kappa_conv = torch.tensor([self.kappa ** (self.n_t - i - 1) for i in range(self.n_t)])\
            .float().view(1, 1, -1).to(self.device)
        # print(f"x permute is {(x.permute(0, 2, 1)).shape}\n")  # 1024 39 11
        # print(f"alpha conv expand shape is {alpha_conv.expand(self.n_in, -1, -1).shape}\n")  # 39 1 11
        # print(f"alpha conv expand is {alpha_conv.expand(self.n_in, -1, -1)}\n")  # 39 1 11

        # compute input eligibility trace
        # shape n_b, n_rec, n_in , n_t
        self.trace_in = F.conv1d(x.permute(0, 2, 1), alpha_conv.expand(self.n_in, -1, -1), padding=self.n_t,
                                 groups=self.n_in)[:, :, 1:self.n_t + 1].unsqueeze(1).expand(-1, self.n_rec, -1, -1)
        self.trace_in = torch.einsum('tbr,brit->brit', self.h, self.trace_in)  # STE * ET

        self.trace_in = F.conv1d(self.trace_in.reshape(self.n_b, self.n_in * self.n_rec, self.n_t),
                                 kappa_conv.expand(self.n_in * self.n_rec, -1, -1), padding=self.n_t,
                                 groups=self.n_in * self.n_rec)[:, :, 1:self.n_t + 1].reshape(self.n_b, self.n_rec,
                                                                                              self.n_in, self.n_t)

        # compute recurrent eligibility trace
        # shape: n_b, n_rec, n_rec, n_t
        self.trace_rec = F.conv1d(self.z.permute(1, 2, 0), alpha_conv.expand(self.n_rec, -1, -1), padding=self.n_t,
                                  groups=self.n_rec)[:, :, :self.n_t].unsqueeze(1).expand(-1, self.n_rec, -1, -1)
        self.trace_rec = torch.einsum('tbr,brit->brit', self.h, self.trace_rec)  # STE * ET

        self.trace_rec = F.conv1d(self.trace_rec.reshape(self.n_b, self.n_rec * self.n_rec, self.n_t),
                                  kappa_conv.expand(self.n_rec * self.n_rec, -1, -1), padding=self.n_t,
                                  groups=self.n_rec * self.n_rec)[:, :, 1:self.n_t + 1].reshape(self.n_b, self.n_rec,
                                                                                                self.n_rec, self.n_t)

        # compute output eligibility trace
        # shape: n_b, n_rec, n_t
        self.trace_out = F.conv1d(self.z.permute(1, 2, 0), kappa_conv.expand(self.n_rec, -1, -1), padding=self.n_t,
                                  groups=self.n_rec)[:, :, 1:self.n_t + 1]



        # 此处的问题在于不能直接用pytorch的方法，这里采用的是硬编码的方式，因此yo实际上是一个张量，其输出神经元维度上是一个softmax之后的值，这里
        # 可以参考之前NC的论文，但是目前来说，标签由于是直接传入的独热码形式，此处可以考虑增加一个中间变量，传入中间变量
        # 换句话说 setup 中就产生标准的函数，而在其他需要独热码的时候再转换成独热码 方便后续的操作
        # print(f"\n\t\t shape of yo is {yo.shape}")
        # print(f"\n\t\t shape of yt is {yt.shape}")
        # print(f"\n\t\t shape of err is {err.shape}")
        # n_b n_t n_o

        # compute the error
        err = yo - yt

        # print(f"error is {err} = yo {yo} - yt{yt}")


        # Learning Signal: error * w_out
        self.L = torch.einsum('tbo,or->brt', err.permute(1, 0, 2), self.w_out)
        # print(f"Learning signal is {self.L}")
        # print(f"shape of learning signal is {self.L.shape}")

        # compute the gradient
        self.w_in.grad += self.lr_layer[0] * torch.sum(self.L.unsqueeze(2).expand(-1, -1, self.n_in, -1) * self.trace_in,
                                                       dim=(0, 3))
        self.w_rec.grad += self.lr_layer[1] * torch.sum(self.L.unsqueeze(2).expand(-1, -1, self.n_rec, -1) * self.trace_rec,
                                                        dim=(0, 3))
        self.w_out.grad += self.lr_layer[2] * torch.einsum('tbo,brt->or', err.permute(1, 0, 2), self.trace_out)
        # #
        # print(f"w_in grad is {self.w_in.grad}")
        # print(f"w_out grad is {self.w_out.grad}")
        # print(f"w_rec grad is {self.w_rec.grad}")
        # print(f"w_in  is {self.w_in}")
        # print(f"w_out  is {self.w_out}")
        # print(f"w_rec  is {self.w_rec}")
    def __repr__(self):

        return self.__class__.__name__ + ' (' \
            + str(self.n_in) + ' -> ' \
            + str(self.n_rec) + ' -> ' \
            + str(self.n_out) + ') '


