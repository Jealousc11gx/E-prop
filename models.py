# this file is used to define the model
# Author： Chen Linliang
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
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
        plt.ion()
        self.fig, self.ax_list = plt.subplots(2 + self.n_out + 5, sharex=True)

        # Initial Weight Parameters
        self.w_in = Parameter(torch.Tensor(n_rec, n_in))  # 标记为可训练参数
        self.w_rec = Parameter(torch.Tensor(n_rec, n_rec))
        self.w_out = Parameter(torch.Tensor(n_out, n_rec))
        self.reset_parameters(w_init_gain)  # the initialization of weight matrix 何凯明的增益权重

    def reset_parameters(self, gain):  # 使用何凯明初始化

        torch.nn.init.kaiming_normal_(self.w_in)  # 对w_in进行何凯明初始化
        self.w_in.data = gain[0] * self.w_in.data  # 0.5 增益系数越高 其标准差也越高 同时这里的增益系数和选用的激活函数也有关系
        torch.nn.init.kaiming_normal_(self.w_rec)
        self.w_rec.data = gain[1] * self.w_rec.data
        torch.nn.init.kaiming_normal_(self.w_out)
        self.w_out.data = gain[2] * self.w_out.data

    # grad and state set to zero
    def init_net(self, n_b, n_t, n_in, n_rec, n_out):
        # Hidden state
        self.v = torch.zeros(n_t, n_b, n_rec).to(self.device)
        # Visible state
        self.vo = torch.zeros(n_t, n_b, n_out).to(self.device)
        self.z = torch.zeros(n_t, n_b, n_rec).to(self.device)
        # Weight gradients
        self.w_in.grad = torch.zeros_like(self.w_in)  # 形状和w_in一样
        self.w_rec.grad = torch.zeros_like(self.w_rec)
        self.w_out.grad = torch.zeros_like(self.w_out)

    """self.v: 隐藏状态，表示循环神经元的膜电位（membrane potential）。膜电位是一个反映神经元内部电压变化的值，
    它受到来自输入、循环连接和激活函数的影响。在这个代码中，self.v 是一个三维张量，
    其形状为 (n_t, n_b, n_rec)，表示在时间 t、batch 中的第 b 个样本，对应于 n_rec 个循环神经元的膜电位。
    self.vo: 可见状态，表示输出层的膜电位。输出层通常是循环神经网络的最后一层，负责产生模型的输出。self.vo 是一个三维张量，
    形状为 (n_t, n_b, n_out)，表示在时间 t、batch 中的第 b 个样本，对应于 n_out 个输出神经元的膜电位。
    self.z: 可见状态，表示循环神经元的输出或激活状态。在这个代码中，self.z 是一个三维张量，形状为 (n_t, n_b, n_rec)，
    表示在时间 t、batch 中的第 b 个样本，对应于 n_rec 个循环神经元的输出。
    通常，self.z 是通过比较隐藏状态 self.v 与阈值（self.thr）来计算的，即 self.z[t + 1] = (self.v[t + 1] > self.thr).float()。
    """

    def forward(self, x, yt, do_training):

        self.n_b = x.shape[0]  # Extracting batch size 1024
        self.init_net(self.n_b, self.n_t, self.n_in, self.n_rec, self.n_out)
        # Network reset 为了消除之前的影响 每次都必须对梯度清0
        self.w_rec *= (1 - torch.eye(self.n_rec, self.n_rec, device=self.device))
        # Making sure recurrent self excitation/inhibition is cancelled
        # 对角线上的为自循环权重 这里使得自循环权重为0 其余位置的权重的值不变

        # print(f"batchsize n_b = {self.n_b},\n"
        #       f"number of step = {self.n_t},\n"
        #       f"number of rec neuron = {self.n_rec},\n"
        #       f"number of input neuron = {self.n_in},\n"
        #       f"number of output neuron = {self.n_out},\n")

        for t in range(self.n_t - 1):  # Computing the network state and outputs for the whole sample duration

            # 前向过程中的数据维度不对 慢慢调整
            # if t == 1:
            #     print("x[t].shape:", x[t].shape)
            #     print("yt[t].shape:", yt[t].shape)
            #     print("self.z[t].shape:", self.z[t].shape)
            #     print("self.v[t].shape:", self.v[t].shape)
            #     print("self.vo[t].shape:", self.vo[t].shape)
            #     print("self.w_rec.t().shape:", self.w_rec.t().shape)
            #     print("self.w_out.t().shape:", self.w_out.t().shape)
            #     print("self.w_in.t().shape:", self.w_in.t().shape)

            # Forward pass - Hidden state:  v: recurrent layer membrane potential
            #                Visible state: z: recurrent layer spike output,
            #                               vo: output layer membrane potential (yo incl. activation function)
            x_t = x[:, t, :].to(self.device)
            self.v[t + 1] = ((self.alpha * self.v[t] + torch.mm(self.z[t], self.w_rec.t()) + torch.mm(x_t, self.w_in.t()))-self.z[t] * self.thr)  # 衰减因子乘电压加上递归的影响再加上输入的影响

            self.z[t + 1] = (self.v[t + 1] > self.thr).float()  # 更新z状态

            self.vo[t + 1] = self.kappa * self.vo[t] + torch.mm(self.z[t + 1], self.w_out.t()) + self.b_o



        if self.classif:  # Apply a softmax function for classification problems  损失函数没有显式应用softmax
            yo = F.softmax(self.vo.permute(1, 0, 2), dim=2)  # 模型第二维是输出层神经元，一共39个，所以是在输出层做了一个softmax作为输出
        else:  # prediction
            yo = self.vo  # (n_t, n_b, n_out)

        if do_training:
            # 训练的话 需要梯度更新
            # with torch.no_grad()
            self.grads_batch(x, yo, yt)
        return yo

    def grads_batch(self, x, yo, yt):  # 用于梯度计算的辅助量
        # print(f"x shape is {x.shape}\n")
        # Surrogate derivatives
        h = self.gamma * torch.max(torch.zeros_like(self.v), 1 - torch.abs((self.v - self.thr) / self.thr))
        # print(f"h is {h.shape}\n")

        # Input and recurrent eligibility vectors for the 'LIF' model (vectorized computation, model-dependent)
        alpha_conv = torch.tensor([self.alpha ** (self.n_t - i - 1) for i in range(self.n_t)]).\
            float().view(1, 1, -1).to(self.device)  # compose a tensor which shape is [1,1,11]

        # print(f"x permute is {(x.permute(0, 2, 1)).shape}\n")  # 1024 39 11
        # print(f"alpha conv expand is {alpha_conv.expand(self.n_in, -1, -1).shape}\n")  # 39 1 11

        trace_in = F.conv1d(x.permute(0, 2, 1), alpha_conv.expand(self.n_in, -1, -1), padding=self.n_t,
                            groups=self.n_in)[:, :, 1:self.n_t + 1].unsqueeze(1).expand(-1, self.n_rec, -1, -1)
        # shape n_b, n_rec, n_in , n_t

        # print(f"trace_in shape process is {trace_in.shape}\n")
        # print(f"h shape is {h.shape}\n")

        # n out  = (  n in + 2* n padding - n kernel )+1

        trace_in = torch.einsum('tbr,brit->brit', h, trace_in)

        # print(f"trace_in shape process is {trace_in.shape}\n")
        # BATCHSIZE set up tp 512, as same as the parameter numbers before

        trace_rec = F.conv1d(self.z.permute(1, 2, 0), alpha_conv.expand(self.n_rec, -1, -1), padding=self.n_t,
                             groups=self.n_rec)[:, :, :self.n_t].unsqueeze(1).expand(-1, self.n_rec, -1,
                                                                                     -1)
        # shape: n_b, n_rec, n_in, n_t
        trace_rec = torch.einsum('tbr,brit->brit', h, trace_rec)
        # shape: n_b, n_rec, n_in, n_t
        # print(f"trace_rec shape process is {trace_rec.shape}\n")  # 5 100 40 2250 = 45，000，000
        # Output eligibility vector (vectorized computation, model-dependent)
        kappa_conv = torch.tensor([self.kappa ** (self.n_t - i - 1) for i in range(self.n_t)]).float().view(1, 1,
                                                                                                            -1).to(
            self.device)
        trace_out = F.conv1d(self.z.permute(1, 2, 0), kappa_conv.expand(self.n_rec, -1, -1), padding=self.n_t,
                             groups=self.n_rec)[:, :, 1:self.n_t + 1]
        # shape: n_b, n_rec, n_t
        # print(f"trace_out shape process is {trace_out.shape}\n")
        # Eligibility traces
        trace_in = F.conv1d(trace_in.reshape(self.n_b, self.n_in * self.n_rec, self.n_t),
                            kappa_conv.expand(self.n_in * self.n_rec, -1, -1), padding=self.n_t,
                            groups=self.n_in * self.n_rec)[:, :, 1:self.n_t + 1].reshape(self.n_b, self.n_rec,
                                                                                         self.n_in,
                                                                                         self.n_t)
        # print(f"trace_in eligibility trace shape is {trace_in.shape}\n")
        # shape: n_b, n_rec, n_in , n_t
        trace_rec = F.conv1d(trace_rec.reshape(self.n_b, self.n_rec * self.n_rec, self.n_t),
                             kappa_conv.expand(self.n_rec * self.n_rec, -1, -1), padding=self.n_t,
                             groups=self.n_rec * self.n_rec)[:, :, 1:self.n_t + 1].reshape(self.n_b, self.n_rec,
                                                                                           self.n_rec,
                                                                                           self.n_t)
        # shape: n_b, n_rec, n_rec, n_t
        # print(f"trace_rec eligibility trace shape is {trace_rec.shape}\n")
        # Learning signals
        # print(f"yo shape is{yo.shape}\n")
        # print(f"yt shape is {yt.shape}\n")
        err = yo - yt
        L = torch.einsum('tbo,or->brt', err.permute(1, 0, 2), self.w_out)
        # self.update_plot(x, self.z, yo, yt, L, trace_in, trace_rec, trace_out, h)
        # print(f"L shape is {L.shape}\n")
        # print(f"error shape is {err.shape}\n")
        # Weight gradient updates

        self.w_in.grad += self.lr_layer[0] * torch.sum(L.unsqueeze(2).expand(-1, -1, self.n_in, -1) * trace_in,
                                                       dim=(0, 3))
        self.w_rec.grad += self.lr_layer[1] * torch.sum(L.unsqueeze(2).expand(-1, -1, self.n_rec, -1) * trace_rec,
                                                        dim=(0, 3))
        self.w_out.grad += self.lr_layer[2] * torch.einsum('tbo,brt->or', err.permute(1, 0, 2), trace_out)
    #
    # def update_plot(self, x, z, yo, yt, L, trace_in, trace_rec, trace_out, h):
    #     # Clear the axis to print new plots
    #     for k in range(len(self.ax_list)):
    #         ax = self.ax_list[k]
    #         ax.clear()
    #
    #     # Plot input and recurrent spikes
    #     for k, spike_ref in enumerate(zip(['In spikes', 'Rec spikes'], [x, z])):
    #         spikes = spike_ref[1][:, 0, :].cpu().numpy()
    #         ax = self.ax_list[k]
    #         ax.imshow(spikes.T, aspect='auto', cmap='hot_r', interpolation="none")
    #         ax.set_xlim([0, self.n_t])
    #         ax.set_ylabel(spike_ref[0])
    #
    #     # Plot outputs and targets
    #     for i in range(self.n_out):
    #         ax = self.ax_list[i + 2]
    #         ax.set_ylabel('Output ' + str(i))
    #
    #         ax.plot(np.arange(self.n_t), yo[:, 0, i].cpu().numpy(), linestyle='dashed', label='Output', alpha=0.8)
    #         ax.plot(np.arange(self.n_t), yt[:, 0, i].cpu().numpy(), linestyle='solid', label='Target', alpha=0.8)
    #         ax.set_xlim([0, self.n_t])
    #
    #     # Plot traces: trace_in, trace_rec, trace_out, h
    #     for i, trace in enumerate([trace_in, trace_rec, trace_out, h]):
    #         ax = self.ax_list[i + 2 + self.n_out]
    #         ax.set_ylabel("Traces in" if i == 0 else "Traces rec" if i == 1 else "Traces out" if i == 2 else "h")
    #
    #         ax.plot(np.arange(self.n_t), trace[0, :, :].T.cpu().numpy(), linestyle='dashed', label='Output', alpha=0.8)
    #         ax.set_xlim([0, self.n_t])
    #
    #     # Plot learning signals
    #     ax = self.ax_list[-1]
    #     ax.set_ylabel("Learning signals")
    #
    #     ax.plot(np.arange(self.n_t), L[0, :, :].T.cpu().numpy(), linestyle='dashed', label='Output', alpha=0.8)
    #     ax.set_xlim([0, self.n_t])
    #
    #     ax.set_xlabel('Time in ms')
    #
    #     # Short wait time to draw with interactive python
    #     plt.draw()
    #     plt.pause(0.1)

    def __repr__(self):

        return self.__class__.__name__ + ' (' \
            + str(self.n_in) + ' -> ' \
            + str(self.n_rec) + ' -> ' \
            + str(self.n_out) + ') '
