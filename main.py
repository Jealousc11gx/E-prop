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
    parser.add_argument('--benchtype', type=bool, default=True,
                        help='Do a train for 1 and 0 for test')
    '''
    # dataset固定，因此不需要在参数中检查
    parser.add_argument('--dataset', type=str, choices=['cue_accumulation'], default='cue_accumulation',
                        help='Choice of the dataset')
    '''
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
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'NAG', 'Adam', 'RMSProp'], default='Adam',
                        help='Choice of the optimizer')
    parser.add_argument('--loss', type=str, choices=['MSE', 'BCE', 'CE'], default='BCE',
                        help='Choice of the loss function (only for performance monitoring purposes, does not influence'
                             'learning)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--lr-layer-norm', type=float, nargs='+', default=(0.05, 0.05, 1.0),
                        help='Per-layer modulation factor of the learning rate')
    # 为每一层指定一个不同的学习率调制因子。这个调制因子会被应用于全局的学习率，从而得到每一层实际使用的学习率。
    # 允许对网络不同层次的权重更新速率进行更细粒度的控制。
    # --lr-layer-norm 是一个包含三个值的列表，分别对应输入层、循环层和输出层的学习率调制因子。
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size for training (limited by the available GPU memory)')
    parser.add_argument('--test-batch-size', type=int, default=512,
                        help='Batch size for testing (limited by the available GPU memory)')
    parser.add_argument('--full_train-len', type=int, default=1229932, help='Number of training set samples')
    parser.add_argument('--full_test-len', type=int, default=451552, help='Number of test set samples')
    parser.add_argument('--visualize', type=bool, default=True, help='Enable network visualization')
    parser.add_argument('--visualize-light', type=bool, default=True,
                        help='Enable light mode in network visualization, plots traces only for a single neuron')
    # Network model parameters
    parser.add_argument('--bias-out', type=float, default=0.0, help='Bias of the output layer')
    parser.add_argument('--gamma', type=float, default=0.3, help='Surrogate derivative magnitude parameter')
    parser.add_argument('--w-init-gain', type=float, nargs='+', default=(0.5, 0.1, 0.5),
                        help='Gain parameter for the He Normal initialization of the input, recurrent and output layer '
                             'weights')
    # changeable parameters
    parser.add_argument('--n-steps', type=int, default=11, help='Number of time steps in each input sample')
    parser.add_argument('--n-rec', type=int, default=300, help='Number of recurrent units')
    parser.add_argument('--model', type=str, choices=['LIF'], default='LIF',
                        help='Neuron model in the recurrent layer. Support for the ALIF neuron model has been removed.')
    parser.add_argument('--threshold', type=float, default=0.6, help='Firing threshold in the recurrent layer')
    # tau-mem 是指循环层（recurrent layer）的膜电位漏电时间常数。在循环神经网络（RNN）的循环单元中，膜电位是用来表示神经元内部状态的一个变量。
    # 漏电时间常数决定了这个膜电位如何随时间变化。较大的漏电时间常数表示膜电位的变化较为缓慢，神经元能够在较长时间内保留信息。
    # tau-out 是指输出层（output layer）的膜电位漏电时间常数。在神经网络的输出层，膜电位通常是一个表示神经元激活程度的变量。
    # 较小的漏电时间常数可以使输出层的激活更为灵活，更快地适应输入的变化。
    parser.add_argument('--tau-mem', type=float, default=2000e-3,
                        help='Membrane potential leakage time constant in the recurrent layer (in seconds)')
    parser.add_argument('--tau-out', type=float, default=20e-3,
                        help='Membrane potential leakage time constant in the output layer (in seconds)')

    args = parser.parse_args()

    (device, train_loader, test_loader) = setup.setup(args)
    train.train(args, device, train_loader, test_loader)


if __name__ == '__main__':
    main()
