import numpy as np


def update_plot(model, x, z, L, trace_in, trace_rec, trace_out, h, fig, ax_list):

    # Clear the axis to print new plots
    for ax in ax_list:
        ax.clear()

    # Plot input signals (heatmap)
    ax = ax_list[0]          # shape n_b, n_rec, n_in , n_t
    ax.imshow(x[0].T.cpu().numpy(), aspect='auto', cmap='hot_r', interpolation="none")
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Input MFCC features')
    ax.set_title('Input signals (x)')

    # Plot recurrent layer spikes (raster plot)  # model.z shape: torch.Size([11, 128, 300])
    ax = ax_list[1]          # shape: n_b, n_rec, n_rec, n_t         # shape: n_b, n_rec, n_t
    rec_spikes = z[:, 0, :].cpu().numpy()
    ax.imshow(rec_spikes.T, aspect='auto', cmap='hot_r', interpolation='nearest')
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Recurrent Neuron')
    ax.set_title('Recurrent layer spikes (z)')

    # Plot input traces
    ax = ax_list[2]
    ax.plot(np.arange(model.n_t), trace_in[0, 1, 0, :].T.cpu().numpy())  # Light
    # ax.plot(np.arange(model.n_t), trace_in[0, :, :, :].reshape(model.n_rec * model.n_in, model.n_t).T.cpu().numpy())
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Trace_in')
    ax.set_title('Input traces')

    # Plot recurrent traces
    ax = ax_list[3]
    ax.plot(np.arange(model.n_t), trace_rec[0, 1, 0, :].T.cpu().numpy())  # Light
    # ax.plot(np.arange(model.n_t), trace_rec[0, :, :, :].reshape(model.n_rec * model.n_rec, model.n_t).T.cpu().numpy())
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Trace_rec')
    ax.set_title('Recurrent traces')

    # Plot output traces
    ax = ax_list[4]
    ax.plot(np.arange(model.n_t), trace_out[0, 1, :].T.cpu().numpy())
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Trace_out')
    ax.set_title('Output traces')

    # Plot learning signals
    ax = ax_list[5]
    for i in range(1, 6):  # change the i-th rec neuron
        ax.plot(np.arange(model.n_t), L[0, i, :].T.cpu().numpy())  # shape: n_b, n_rec, n_t
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Learning signal')
    ax.set_title('Learning signals')

    # # Plot pseudo-derivatives  # hot_pot
    # ax = ax_list[6]
    # ax.imshow(h[:, 0, :].T.cpu().numpy(), aspect='auto', cmap='coolwarm', interpolation='nearest')
    # ax.set_xlabel('Time steps')
    # ax.set_ylabel('Surrogate Derivative Value')
    # ax.set_title('Pseudo-derivatives (h)')

    # Plot pseudo-derivatives  # model.h shape: torch.Size([11, 128, 300])
    ax = ax_list[6]
    for j in range(1, 6):  # change the i-th rec neuron
        ax.plot(np.arange(model.n_t), h[:, 0, j].T.cpu().numpy())
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Surrogate Derivative')
    ax.set_title('Pseudo-derivatives (h)')

    # Adjust the layout and display the plots
    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
