# this file is used to set some hyperparameters
# Author： Chen Linliang
import argparse
import train
import setup


def main():
    parser = argparse.ArgumentParser(description='Spiking RNN Pytorch training for TIMIT')
    # General
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Disable CUDA training and run training on CPU')
    parser.add_argument('--benchtype', type=str, choices=['train', 'test', 'val'], default='val',
                        help='type of the mission')
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='Enables shuffling sample order in datasets after each epoch')
    parser.add_argument('--trials', type=int, default=1,
                        help='Number of trial experiments to do (i.e. repetitions with different initializations)')
    parser.add_argument('--seed', type=int, default=42,
                        help='value of seed , the ultimate answer of the universe)')
    parser.add_argument('--n_inputs', type=int, default=39,
                        help='number of input neuron)')
    parser.add_argument('--n_classes', type=int, default=39,
                        help='number of output neuron)')
    parser.add_argument('--dt', type=int, default=0.01,
                        help='dt)')
    parser.add_argument('--classif', type=bool, default=1,
                        help='classification')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')

    parser.add_argument('--optimizer', type=str, choices=['SGD', 'NAG', 'Adam', 'RMSProp'], default='Adam',
                        help='Choice of the optimizer')
    parser.add_argument('--loss', type=str, choices=['MSE', 'BCE', 'CE'], default='BCE',
                        help='Choice of the loss function (only for performance monitoring purposes, does not influence'
                             'learning)')

    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--lr-layer-norm', type=float, nargs='+', default=(0.5, 0.1, 1),
                        help='Per-layer modulation factor of the learning rate')

    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training (limited by the available GPU memory)')
    parser.add_argument('--test-batch-size', type=int, default=128,
                        help='Batch size for testing (limited by the available GPU memory)')
    parser.add_argument('--val-batch-size', type=int, default=128,
                        help='Batch size for validation (limited by the available GPU memory)')

    parser.add_argument('--full_train-len', type=int, default=1106939, help='Number of training set samples')
    parser.add_argument('--full_test-len', type=int, default=61497, help='Number of test set samples')
    parser.add_argument('--full_val-len', type=int, default=61497, help='Number of val set samples')

    parser.add_argument('--visualize', type=bool, default=True, help='Enable network visualization')
    parser.add_argument('--visualize-light', type=bool, default=True,
                        help='Enable light mode in network visualization, plots traces only for a single neuron')
    # Network model parameters
    parser.add_argument('--bias-out', type=float, default=0.0, help='Bias of the output layer')
    parser.add_argument('--gamma', type=float, default=0.3, help='Surrogate derivative magnitude parameter')
    parser.add_argument('--w-init-gain', type=float, nargs='+', default=(0.1, 0.1, 0.5),
                        help='Gain parameter for the He Normal initialization of the input, recurrent and output layer '
                             'weights')
    # changeable parameters
    parser.add_argument('--n-steps', type=int, default=11, help='Number of time steps in each input sample')
    parser.add_argument('--n-rec', type=int, default=300, help='Number of recurrent units')
    parser.add_argument('--model', type=str, choices=['LIF'], default='LIF',
                        help='Neuron model in the recurrent layer. Support for the ALIF neuron model has been removed.')
    parser.add_argument('--threshold', type=float, default=0.4, help='Firing threshold in the recurrent layer')
    parser.add_argument('--tau-mem', type=float, default=2000e-3,
                        help='Membrane potential leakage time constant in the recurrent layer (in seconds)')
    parser.add_argument('--tau-out', type=float, default=20e-3,
                        help='Membrane potential leakage time constant in the output layer (in seconds)')

    args = parser.parse_args()

    (device, train_loader, val_loader, test_loader) = setup.setup(args)
    train.train(args, device, train_loader, val_loader, test_loader)


if __name__ == '__main__':
    main()
