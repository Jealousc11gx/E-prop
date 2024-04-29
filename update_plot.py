import numpy as np
import matplotlib.pyplot as plt


def update_plot1(self, x, z, yo, yt, L, trace_reg, trace_in, trace_rec, trace_out):

    # Clear the axis to print new plots
    for k in range(self.ax_list.shape[0]):
        ax = self.ax_list[k]
        ax.clear()

    # Plot input signals
    for k, spike_ref in enumerate(zip(['In spikes', 'Rec spikes'], [x, z])):
        spikes = spike_ref[1][:, 0, :].cpu().numpy()
        ax = self.ax_list[k]

        ax.imshow(spikes.T, aspect='auto', cmap='hot_r', interpolation="none")
        ax.set_xlim([0, self.n_t])
        ax.set_ylabel(spike_ref[0])

    for i in range(self.n_out):
        ax = self.ax_list[i + 2]
        if self.classif:
            ax.set_ylim([-0.05, 1.05])
        ax.set_ylabel('Output ' + str(i))

        ax.plot(np.arange(self.n_t), yo[:, 0, i].cpu().numpy(), linestyle='dashed', label='Output', alpha=0.8)
        if self.t_crop != 0:
            ax.plot(np.arange(self.n_t)[-self.t_crop:], yt[-self.t_crop:, 0, i].cpu().numpy(), linestyle='solid',
                    label='Target', alpha=0.8)
        else:
            ax.plot(np.arange(self.n_t), yt[:, 0, i].cpu().numpy(), linestyle='solid', label='Target', alpha=0.8)

        ax.set_xlim([0, self.n_t])

    for i in range(5):
        ax = self.ax_list[i + 2 + self.n_out]
        ax.set_ylabel(
            "Trace reg" if i == 0 else "Traces out" if i == 1 else "Traces rec" if i == 2 else "Traces in" if i == 3 else "Learning sigs")

        if i == 0:
            if self.visu_l:
                ax.plot(np.arange(self.n_t), trace_reg[0, :, 0, :].T.cpu().numpy(), linestyle='dashed',
                        label='Output', alpha=0.8)
            else:
                ax.plot(np.arange(self.n_t),
                        trace_reg[0, :, :, :].reshape(self.n_rec * self.n_rec, self.n_t).T.cpu().numpy(),
                        linestyle='dashed', label='Output', alpha=0.8)
        elif i < 4:
            if self.visu_l:
                ax.plot(np.arange(self.n_t), trace_out[0, :, :].T.cpu().numpy() if i == 1 \
                    else trace_rec[0, :, 0, :].T.cpu().numpy() if i == 2 \
                    else trace_in[0, :, 0, :].T.cpu().numpy(), linestyle='dashed', label='Output', alpha=0.8)
            else:
                ax.plot(np.arange(self.n_t), trace_out[0, :, :].T.cpu().numpy() if i == 1 \
                    else trace_rec[0, :, :, :].reshape(self.n_rec * self.n_rec, self.n_t).T.cpu().numpy() if i == 2 \
                    else trace_in[0, :, :, :].reshape(self.n_rec * self.n_in, self.n_t).T.cpu().numpy(),
                        linestyle='dashed', label='Output', alpha=0.8)
        elif self.t_crop != 0:
            ax.plot(np.arange(self.n_t)[-self.t_crop:], L[0, :, -self.t_crop:].T.cpu().numpy(), linestyle='dashed',
                    label='Output', alpha=0.8)
        else:
            ax.plot(np.arange(self.n_t), L[0, :, :].T.cpu().numpy(), linestyle='dashed', label='Output', alpha=0.8)

    ax.set_xlim([0, self.n_t])
    ax.set_xlabel('Time in ms')

    # Short wait time to draw with interactive python
    plt.draw()
    plt.pause(0.1)


def __repr__(self):
    return self.__class__.__name__ + ' (' \
        + str(self.n_in) + ' -> ' \
        + str(self.n_rec) + ' -> ' \
        + str(self.n_out) + ') '



