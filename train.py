import torch
import torch.nn.functional as F
import torch.optim as optim
import models
from tqdm import tqdm


def train(args, device, train_loader, test_loader):
    torch.manual_seed(42)  # the ultimate answer of the universe

    for trial in range(1, args.trials + 1):

        # Network topology
        model = models.SRNN(n_in=args.n_inputs,
                            n_rec=args.n_rec,
                            n_out=args.n_classes,
                            n_t=args.n_steps,
                            thr=args.threshold,
                            tau_m=args.tau_mem,
                            tau_o=args.tau_out,
                            b_o=args.bias_out,
                            gamma=args.gamma,
                            dt=args.dt,
                            model=args.model,
                            classif=args.classif,
                            w_init_gain=args.w_init_gain,
                            lr_layer=args.lr_layer_norm,
                            device=device)

        # Use CUDA for GPU-based computation if enabled
        if args.cuda:
            model.cuda()

        # Optimizer
        if args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr)
        elif args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        elif args.optimizer == 'NAG':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
        elif args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
        else:
            raise NameError("=== ERROR: optimizer " + str(args.optimizer) + " not supported")

        # Loss function (only for performance monitoring purposes, does not influence learning as e-prop learning\
        # is hardcoded)
        if args.loss == 'MSE':
            loss = (F.mse_loss, (lambda l: l))
        elif args.loss == 'BCE':
            loss = (F.binary_cross_entropy, (lambda l: l))
        elif args.loss == 'CE':
            loss = (F.cross_entropy, (lambda l: torch.max(l, 1)[1]))
        else:
            raise NameError("=== ERROR: loss " + str(args.loss) + " not supported")

        # Training and performance monitoring
        print("\n=== Starting model training with %d epochs:\n" % (args.epochs,))
        for epoch in range(1, args.epochs + 1):
            print("\t Epoch " + str(epoch) + "...")
            # Training:
            do_epoch(args, True, model, device, train_loader, optimizer, loss, 'train')
            # Will display the average accuracy on the training set during the epoch (changing weights)
            # Check performance on the training set and on the test set:
            # if not args.skip_test:
                # Uncomment do_epoch(args, False, model, device, val_loader, optimizer, loss, 'train')
                # to display the final accuracy on the training set after the epoch (fixed weights)
                #do_epoch(args, False, model, device, test_loader, optimizer, loss, 'test')


def do_epoch(args, do_training, model, device, loader, optimizer, loss_fct, benchType):
    model.eval()  # 没有自动梯度反向传播，这里也没有使用drop out和BN层，但是eval的计算是为了禁止梯度的更新
    score = 0
    loss = 0
    batch = args.batch_size if (benchType == 'train') else args.test_batch_size
    length = args.full_train_len if (benchType == 'train') else args.full_test_len  # 根据bench type的类型修改长度
    with torch.no_grad():  # 在训练过程中禁止梯度的自动变换，因为梯度的计算是hard code
        # For each batch
        for batch_idx, (data, label) in enumerate(loader):

            data, label = data.to(device), label.to(device)

            # print(f"Batch {batch_idx + 1}:")
            # print("data:", data)
            # print("data shape:", data.shape)
            # print("Label shape:", label.shape)
            # print("Label data:", label)
            # if batch_idx == 0:
            #     break


            # Label shape: torch.Size([256, 11, 39])
            # data shape: torch.Size([256, 11, 39])


            # Evaluate the model for all the time steps of the input data, then either do the weight updates on
            # a per-timestep basis, or on a per-sample basis (sum of all per-timestep updates).
            optimizer.zero_grad()
            # output = model(data.permute(1, 0, 2), targets, do_training)  修改了数据因此不需要适配维度
            output = model(data, label, do_training)
            if do_training:
                optimizer.step()

            # Compute the loss function, inference and score
            loss += loss_fct[0](output, loss_fct[1](label), reduction='mean')
            per_loss = loss_fct[0](output, loss_fct[1](label), reduction='mean')

            # 其中n_t是时间步数，n_b是批量大小，n_o是输出类别数
            # 输出实际上是 b t o
            # 计算预测：对每个时间步的输出进行求和，然后在输出类别维度上使用argmax获取预测的类别索引
            inference = torch.argmax(torch.sum(output, dim=1), dim=1)  # 结果形状 [n_b]
            # print(f"inference shape is {inference.shape}")
            # print(f"inference example is {inference[:3]}")

            # 将独热码标签转换为类别索引以便比较
            actual_indices = torch.argmax(label[:, 0, :], dim=1)  # 假设每个时间步的标签相同，只取第一个时间步
            # print(f"actual indices is {actual_indices[:3]}")

            # 计算正确预测的数量，将布尔型Tensor转换为整型Tensor再求和
            correct_predictions = torch.sum((inference == actual_indices).int()).item()

            # 更新总分数
            score += correct_predictions

            # 计算准确率
            accuracy = correct_predictions / label.size(0) * 100  # 计算这个批次的准确率

            if benchType == "train" and do_training:
                info = "on training set (while training): "
            elif benchType == "train":
                info = "on training set                 : "
            elif benchType == "test":
                info = "on test set                     : "

            # 输出信息，对于loss直接使用而不调用.item()
            print(f"\t\t Score {info} {score}/{length} ({accuracy}%), loss: {per_loss}")
            # 这里的loss有问题 原因是输出的是累加的loss而不是单次的loss输出


'''
            inference = torch.argmax(torch.sum(output, axis=0), axis=1)
            score += torch.sum(torch.eq(inference, label[:, 0]))


    if benchType == "train" and do_training:
        info = "on training set (while training): "
    elif benchType == "train":
        info = "on training set                 : "
    elif benchType == "test":
        info = "on test set                     : "

    print("\t\t Score " + info + str(score.item()) + '/' + str(length) + ' (' + str(
            score.item() / length * 100) + '%), loss: ' + str(loss.item()))
'''
