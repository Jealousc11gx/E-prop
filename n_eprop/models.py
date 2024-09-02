import torch
import torch.nn as nn

import numpy as np
from collections import namedtuple


class SpikeFunction(torch.autograd.Function):

    scale = 0.3

    @staticmethod
    def pseudo_derivative(v):
        # return 1.0 / (10 * torch.abs(v) + 1.0) ** 2
        return torch.maximum(1 - torch.abs(v), torch.tensor(0)) * SpikeFunction.scale

    @staticmethod
    def forward(ctx, v_scaled):
        ctx.save_for_backward(v_scaled)
        return (v_scaled > 0).type(v_scaled.dtype)

    # ctx is a context object that can be used to stash information for backward computation
    @staticmethod
    def backward(ctx, dy):
        (v_scaled,) = ctx.saved_tensors

        dE_dz = dy  # dy是损失函数对脉冲的梯度
        dz_dv_scaled = SpikeFunction.pseudo_derivative(v_scaled)    # 伪导数
        dE_dv_scaled = dE_dz * dz_dv_scaled  # 损失函数对膜电位的梯度

        return dE_dv_scaled  # 返回损失函数对膜电位的梯度


activation = SpikeFunction.apply  # SpikeFunction.apply是一个静态方法 这里还需要看看 自定义求导函数的标准写法


class Network(nn.Module):
    NeuronState = namedtuple(
        "NeuronState",
        (
            "V_rec",
            "S_rec",
            "R_rec",
            "A_rec",
            "V_out",
            "S_out",
            "e_trace_in",
            "e_trace_rec",
            "epsilon_v_in",
            "epsilon_v_rec",
            "epsilon_v_out",
            "epsilon_a_in",
            "epsilon_a_rec",
        ),
    )

    def __init__(self, n_in, n_rec, n_out, args):
        super(Network, self).__init__()

        self.dt = args.dt
        self.n_in = n_in
        self.n_rec = n_rec
        self.n_out = n_out
        self.n_refractory = args.n_ref
        self.recurrent = args.recurrent

        # Weight matrix creation
        self.W_in = torch.nn.Parameter(
            torch.tensor(0.2 * np.random.randn(n_in, n_rec) / np.sqrt(n_in)).float(),
            requires_grad=True,
        )
        if self.recurrent:
            recurrent_weights = 0.2 * np.random.randn(n_rec, n_rec) / np.sqrt(n_rec)
            self.W_rec = torch.nn.Parameter(
                torch.tensor(
                    recurrent_weights - recurrent_weights * np.eye(n_rec, n_rec)
                ).float(),
                requires_grad=args.train_rec,
            )
        self.W_out = torch.nn.Parameter(
            torch.tensor(np.random.randn(n_rec, n_out) / np.sqrt(n_rec)).float(),
            requires_grad=True,
        )
        self.register_buffer(
            "b_out",
            torch.tensor(np.random.randn(n_rec, n_out) / np.sqrt(n_rec)).float(),
        )   # 防止被优化器更新

        # Self recurrency
        self.register_buffer("identity_diag_rec", torch.eye(n_rec, n_rec))

        # Parameters creation
        distribution = torch.distributions.gamma.Gamma(3, 3 / args.tau_v)
        tau_v = distribution.rsample((1, n_rec)).clamp(3, 100)
        self.register_buffer("decay_v", torch.exp(-args.dt / tau_v).float())
        # self.register_buffer('decay_v', torch.tensor(np.exp(-dt/tau_v)).float())
        self.register_buffer(
            "decay_o", torch.tensor(np.exp(-args.dt / args.tau_o)).float()
        )
        self.register_buffer(
            "decay_a", torch.tensor(np.exp(-args.dt / args.tau_a)).float()
        )   # "decay_v", "decay_o", "decay_a": 电压、输出和阈值适应的衰减因子
        self.register_buffer("thr", torch.tensor(args.thr).float())
        self.register_buffer("theta", torch.tensor(args.theta).float())

        self.state = None

    def initialize_state(self, input):
        state = self.NeuronState(
            V_rec=torch.zeros(input.shape[0], self.n_rec, device=input.device),
            S_rec=torch.zeros(input.shape[0], self.n_rec, device=input.device),
            R_rec=torch.zeros(input.shape[0], self.n_rec, device=input.device),
            A_rec=torch.zeros(input.shape[0], self.n_rec, device=input.device),
            V_out=torch.zeros(input.shape[0], self.n_out, device=input.device),
            S_out=torch.zeros(input.shape[0], self.n_out, device=input.device),
            e_trace_in=torch.zeros(
                input.shape[0], self.n_in, self.n_rec, device=input.device
            ),
            e_trace_rec=torch.zeros(
                input.shape[0], self.n_rec, self.n_rec, device=input.device
            ),
            epsilon_v_in=torch.zeros(
                input.shape[0], self.n_in, self.n_rec, device=input.device
            ),
            epsilon_v_rec=torch.zeros(
                input.shape[0], self.n_rec, self.n_rec, device=input.device
            ),
            # epsilon_v_in  = torch.zeros(input.shape[0], self.n_in, device=input.device),
            # epsilon_v_rec = torch.zeros(input.shape[0], self.n_rec, device=input.device),
            epsilon_v_out=torch.zeros(input.shape[0], self.n_rec, device=input.device),
            epsilon_a_in=torch.zeros(
                input.shape[0], self.n_in, self.n_rec, device=input.device
            ),
            epsilon_a_rec=torch.zeros(
                input.shape[0], self.n_rec, self.n_rec, device=input.device
            ),
        )
        return state

    def reset(self):
        self.state = None

    def forward(self, input):
        if self.state is None:
            self.state = self.initialize_state(input)

        # Neuron parameters
        V_rec = self.state.V_rec
        S_rec = self.state.S_rec

        V_out = self.state.V_out
        S_out = self.state.S_out

        R_rec = self.state.R_rec  # refractory period
        A_rec = self.state.A_rec  # Threshold adaptation

        # Algorithm parameters
        e_trace_in = self.state.e_trace_in
        epsilon_a_in = self.state.epsilon_a_in
        epsilon_v_in = self.state.epsilon_v_in
        e_trace_rec = self.state.e_trace_rec
        epsilon_v_rec = self.state.epsilon_v_rec
        epsilon_a_rec = self.state.epsilon_a_rec
        epsilon_v_out = self.state.epsilon_v_out

        with torch.no_grad():
            A = self.thr + self.theta * A_rec  # adaptive threshold
            psi = SpikeFunction.pseudo_derivative((V_rec - A) / self.thr)   # pseudo-derivative
            # epsilon_a_in = psi[:,None,:] * epsilon_v_in[:,:,None] + (self.decay_a  - psi[:,None,:]*self.theta)*epsilon_a_in
            epsilon_a_in = (
                psi[:, None, :] * epsilon_v_in
                + (self.decay_a - psi[:, None, :] * self.theta) * epsilon_a_in
            )   # adaption threshold membrane eligibility trace for input layer
            if self.recurrent:
                # epsilon_a_rec = psi[:,None,:] * epsilon_v_rec[:,:,None] + (self.decay_a  - psi[:,None,:]*self.theta)*epsilon_a_rec
                epsilon_a_rec = (
                    psi[:, None, :] * epsilon_v_rec
                    + (self.decay_a - psi[:, None, :] * self.theta) * epsilon_a_rec
                )   # adaption threshold membrane eligibility trace for rec layer

        # Threshold adaptation
        A_rec = self.decay_a * A_rec + S_rec    # Threshold adaptation
        A = self.thr + A_rec * self.theta   # adaptive threshold

        # Detach previous spike for recurrency and reset
        S_rec = S_rec.detach()  # detach previous spikes

        # Current calculation
        if self.recurrent:
            I_in = torch.mm(input, self.W_in) + torch.mm(S_rec, self.W_rec)
        else:
            I_in = torch.mm(input, self.W_in)
        # I_reset = S_rec * self.thr

        # Recurrent neurons update
        # V_rec_new = (self.decay_v * V_rec + I_in) * (1-S_rec)
        V_rec_new = self.decay_v * V_rec + I_in - self.thr * S_rec

        # Spike generation
        is_refractory = R_rec > 0
        zeros_like_spikes = torch.zeros_like(S_rec)
        S_rec_new = torch.where(
            is_refractory, zeros_like_spikes, activation((V_rec_new - A) / self.thr)
        )
        R_rec_new = R_rec + self.n_refractory * S_rec_new - 1
        R_rec_new = torch.clip(R_rec_new, 0.0, self.n_refractory).detach()

        # torch.clip(input, min, max)将输入张量的值限制在[min, max]范围内。这里使用clip的原因是： 确保不应期计数器（R_rec_new）始终在0
        # 到self.n_refractory之间。防止不应期计数器变为负值或超过最大不应期。.detach()用于切断梯度流，因为不应期计数器不需要梯度

        # Forward pass of the data to output weights
        I_out = torch.mm(S_rec_new, self.W_out)

        # Recurrent neurons update
        V_out_new = self.decay_o * V_out + I_out - self.thr * S_out
        # V_out_new = (self.decay_o * V_out + I_out) * (1-S_out)
        S_out_new = activation((V_out - self.thr) / self.thr)

        with torch.no_grad():
            if input.is_sparse:  # 输入层电压资格迹 is_sparse是一个属性 没有定义的话都属于密集张量
                epsilon_v_in = (
                    self.decay_v[:, None, :] * epsilon_v_in
                    + input.to_dense()[:, :, None]
                )
                # epsilon_v_in  = self.decay_v * epsilon_v_in + input.to_dense()
            else:
                epsilon_v_in = (
                    self.decay_v[:, None, :] * epsilon_v_in + input[:, :, None]
                )
                # epsilon_v_in  = self.decay_v * epsilon_v_in + input
            if self.recurrent:  # 循环层电压资格迹
                epsilon_v_rec = (
                    self.decay_v[:, None, :] * epsilon_v_rec + S_rec[:, :, None]
                )
                # epsilon_v_rec = self.decay_v * epsilon_v_rec + S_rec
            epsilon_v_out = self.decay_o * epsilon_v_out + S_rec_new  # 输出层电压资格迹

            v_scaled = (V_rec_new - A) / self.thr
            is_refractory = R_rec > 0

            psi_no_ref = SpikeFunction.pseudo_derivative(v_scaled)  # 没有不应期的情况
            psi = torch.where(is_refractory, torch.zeros_like(psi_no_ref), psi_no_ref)

            # torch.where(condition, x, y)
            # 的作用是：当条件为真时选择 x，否则选择 y
            # 这里的用途是：如果神经元处于不应期（is_refractory
            # 为真），则输出为0；否则，计算正常的激活

            e_trace_in = e_trace_in * self.decay_o + (
                psi[:, None, :] * (epsilon_v_in - self.theta * epsilon_a_in)
            )   # 输入层资格迹
            if self.recurrent:
                e_trace_rec = e_trace_rec * self.decay_o + (
                    psi[:, None, :] * (epsilon_v_rec - self.theta * epsilon_a_rec)
                )   # 循环层资格迹


            # e_trace_in = e_trace_in * self.decay_o + (psi[:,None,:] * (epsilon_v_in[:,:,None] - self.theta*epsilon_a_in)) # psi[:,None,:] * epsilon_v_in
            # e_trace_rec = e_trace_rec * self.decay_o + (psi[:,None,:] * (epsilon_v_rec[:,:,None] - self.theta*epsilon_a_rec)) # psi[:,None,:] * epsilon_v_rec
            # e_trace_rec -= self.identity_diag_rec[None,:,:] * e_trace_rec # No self recurrency

        # e_trace_in 和 e_trace_rec：输入层和循环层的综合资格迹
        # epsilon_v_in, epsilon_v_rec, epsilon_v_out：输入层、循环层和输出层的电压资格迹
        # epsilon_a_in 和 epsilon_a_rec：输入层和循环层的阈值适应资格迹
        # 这些资格迹的作用：
        # 电压资格迹（epsilon_v_ *）：跟踪神经元膜电位的历史变化
        # 阈值适应资格迹（epsilon_a_ *）：跟踪神经元阈值的历史变化
        # 综合资格迹（e_trace_ *）：结合电压和阈值信息，用于权重更新


        new_state = self.NeuronState(
            V_rec=V_rec_new,
            S_rec=S_rec_new,
            R_rec=R_rec_new,
            A_rec=A_rec,
            V_out=V_out_new,
            S_out=S_out_new,
            e_trace_in=e_trace_in.detach(),
            e_trace_rec=e_trace_rec.detach(),
            epsilon_v_in=epsilon_v_in.detach(),
            epsilon_v_rec=epsilon_v_rec.detach(),
            epsilon_v_out=epsilon_v_out.detach(),
            epsilon_a_in=epsilon_a_in.detach(),
            epsilon_a_rec=epsilon_a_rec.detach(),
        )

        # V_rec: 循环层神经元的膜电位
        # S_rec: 循环层神经元的脉冲输出
        # V_out: 输出层神经元的膜电位
        # S_out: 输出层神经元的脉冲输出
        # R_rec: 循环层神经元的不应期计数器
        # A_rec: 循环层神经元的自适应阈值
        #
        # e_trace_in: 输入层的资格迹
        # epsilon_a_in: 输入层的阈值适应资格迹
        # epsilon_v_in: 输入层的电压资格迹
        # e_trace_rec: 循环层的资格迹
        # epsilon_v_rec: 循环层的电压资格迹
        # epsilon_a_rec: 循环层的阈值适应资格迹
        # epsilon_v_out: 输出层的电压资格迹

        self.state = new_state

        return S_out_new

    def detach(self):
        for state in self.state:
            state.detach_()
