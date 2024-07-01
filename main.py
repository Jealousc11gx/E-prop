# this file is used to set some hyperparameters
# Author： Chen Linliang
import argparse
import train
import train_origin
import train_snn
import setup_timit
import wandb
import datetime


def main():
    parser = argparse.ArgumentParser(description='Spiking RNN Pytorch training for TIMIT')
    # General
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Disable CUDA training and run training on CPU')
    parser.add_argument('--benchtype', type=str, choices=['train', 'test', 'val'], default='val',
                        help='type of the mission')
    parser.add_argument('--trials', type=int, default=1,
                        help='Number of trial experiments to do (i.e. repetitions with different initializations)')
    parser.add_argument('--seed', type=int, default=42,
                        help='value of seed , the ultimate answer of the universe)')
    parser.add_argument('--visualize', type=bool, default='True',
                        help='visualize the training process in tensorboard')
    parser.add_argument('--n_inputs', type=int, default=39,
                        help='number of input neuron)')
    parser.add_argument('--n_classes', type=int, default=39,
                        help='number of output neuron)')
    parser.add_argument('--dt', type=int, default=0.01,
                        help='dt)')
    parser.add_argument('--classif', type=bool, default=1,
                        help='classification')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train')

    parser.add_argument('--optimizer', type=str, choices=['SGD', 'NAG', 'Adam', 'RMSProp'], default='Adam',
                        help='Choice of the optimizer')

    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--lr_layer_norm', type=float, nargs='+', default=(0.1, 0.5, 1),
                        help='Per-layer modulation factor of the learning rate')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training (limited by the available GPU memory)')

    # Network model parameters
    parser.add_argument('--bias_out', type=float, default=0.0, help='Bias of the output layer')
    parser.add_argument('--gamma', type=float, default=0.3, help='Surrogate derivative magnitude parameter')
    parser.add_argument('--w_init_gain', type=float, nargs='+', default=(0.5, 1, 1),
                        help='Gain parameter for the He Normal initialization of the input, recurrent and output layer '
                             'weights')
    # changeable parameters
    parser.add_argument('--n_rec', type=int, default=100, help='Number of recurrent units')
    parser.add_argument('--model', type=str, choices=['LIF'], default='LIF',
                        help='Neuron model in the recurrent layer. Support for the ALIF neuron model has been removed.')
    parser.add_argument('--threshold', type=float, default=0.02, help='Firing threshold in the recurrent layer')
    parser.add_argument('--tau_mem', type=float, default=5e-3,
                        help='Membrane potential leakage time constant in the recurrent layer (in seconds)')
    parser.add_argument('--tau_out', type=float, default=20e-3,
                        help='Membrane potential leakage time constant in the output layer (in seconds)')
    parser.add_argument('--l2', type=float, default=1e-5, help='L2 regularization coefficient')
    parser.add_argument('--data_path', type=str, default='E:/TIMIT/timit_11/raw_TIMIT/output_dir/dataset/',
                        help='Absolute Directory of data')
    parser.add_argument('--label_path', type=str, default='E:/TIMIT/timit_11/raw_TIMIT/output_dir/dataset/',
                        help='Absolute Directory of labels')

    parser.add_argument('--surro_deri', type=str, choices=['linear', 'nonlinear'], default='nonlinear',
                        help='type of surrogate_derivate')
    parser.add_argument('--use_wandb', type=int, default=1,
                        help='use wandb to visualize or not')


    args = parser.parse_args()
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.use_wandb:
        wandb.init(
            project="E-Prop_TIMIT",
            name=f"TIMIT_{current_time}_biSRNN",
            config={
                "n_rec": args.n_rec,
                "threshold": args.threshold,
                "tau_mem": args.tau_mem,
                "tau_out": args.tau_out,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "gamma": args.gamma,
                "surro_deri": args.surro_deri,
                "w_init_gain": args.w_init_gain,
                "lr_layer_norm": args.lr_layer_norm,
            }

        )

    (device, train_loader, val_loader, test_loader) = setup_timit.setup(args)
    train_snn.train(args, device, train_loader, val_loader, test_loader)


if __name__ == '__main__':
    main()
    wandb.finish()
