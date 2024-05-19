import torch
import torch.nn.functional as F
import torch.optim as optim
import models
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import update_plot
import matplotlib.pyplot as plt


def train(args, device, train_loader, val_loader, test_loader):
    torch.manual_seed(42)  # the ultimate answer of the universe
    for trial in range(1, args.trials + 1):

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

        # update_plot
        fig, ax_list = plt.subplots(nrows=7, figsize=(8, 10))
        plt.ion()

        # checkpoint
        best_acc = 0
        save_model_path = 'e:/TIMIT/timit_11/ckpt/model.ckpt'

        # SummaryWriter
        writer = SummaryWriter(log_dir=f'runs/trial_{trial}')

        # Optimizer
        if args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr)
        elif args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5, betas=(0.9, 0.999))
        elif args.optimizer == 'NAG':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
        elif args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
        else:
            raise NameError("=== ERROR: optimizer " + str(args.optimizer) + " not supported")

        # Loss function
        if args.loss == 'MSE':
            loss = (F.mse_loss, (lambda l: l))
        elif args.loss == 'BCE':
            loss = (F.binary_cross_entropy, (lambda l: l))
        elif args.loss == 'CE':
            loss = (F.cross_entropy, (lambda l: torch.max(l, 1)[1]))  # 接受一个张量 l，并返回在每行上最大值的索引。
        else:
            raise NameError("=== ERROR: loss " + str(args.loss) + " not supported")

        # Training and performance monitoring
        print("\n=== Starting model training with %d epochs: ===\n" % (args.epochs,))

        for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):

            print("\t Epoch " + str(epoch) + "...")

            # Training:
            train_epoch(args, True, model, device, test_loader, optimizer, loss, 'train',
                        writer, epoch, fig, ax_list)
            # # Validation
            # if val_loader is not None:
            #
            #     print("\t\t Validating...")
            #
            #     val_acc = train_epoch(args, False, model, device, val_loader, optimizer, loss, 'val',
            #                           writer, epoch, fig, ax_list)
            #     if val_acc > best_acc:
            #         best_acc = val_acc
            #         torch.save(model.state_dict(), save_model_path)
            #
            #         print("\t\t Saving model with Validation Accuracy: {:.2f}%".format(best_acc))
            #
            #     else:
            #         # Save the model at the last epoch if no validation set
            #         torch.save(model.state_dict(), save_model_path)
            #
            #         print("\t\t Saving model at the last epoch")

        writer.close()


# onehot convertor
def onehot_convertor(labels, n_t, n_o):
    n_b = labels.size(0)
    one_hot_labels = torch.zeros(n_b, n_t, n_o)
    for b in range(n_b):
        one_hot_labels[b, :, labels[b]] = 1
    return one_hot_labels


def train_epoch(args, do_training, model, device, loader, optimizer, loss_fct, benchType, writer, epoch, fig, ax_list):
    model.eval()  # 没有自动梯度反向传播，这里也没有使用drop out和BN层，但是eval的计算是为了禁止梯度的更新
    # model.train() 后续如果需要把最后一层设置为梯度的自动更新该如何设置
    epoch_loss = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()
    epoch_correct_predictions = 0

    # length (val is small dataset just for adjust the parameter, not for validation)
    if benchType == 'train':
        length = args.full_train_len
    elif benchType == 'val':
        length = args.full_val_len
    else:
        length = args.full_test_len

    with torch.no_grad():  # 在训练过程中禁止梯度的自动变换，因为梯度的计算是hard code
        # For each batch
        pbar = tqdm(enumerate(loader), total=len(loader), desc="Batches", postfix={"batch_acc": 0.0, "loss": 0.0})
        # pbar = tqdm(enumerate(loader), total=len(loader), desc="Batches")
        global_step = 0
        for batch_idx, (data, label) in pbar:
            # Label shape: torch.Size([256, 11, 39])
            # data shape: torch.Size([256, 11, 39])
            # print(f"data shape: {data.shape}")
            # print(f"data: {data}")
            # print(f"Label shape: {label.shape}")
            # print(f"label:{label}")
            onehot_labels_yt = onehot_convertor(label, n_t=11, n_o=39)  # ([128])->([128, 11, 39])
            ce_labels = label.unsqueeze(1).expand(-1, 11)  # (128, 11)
            # print(f"shape of ce_labels is \t {ce_labels.shape}")
            # print(f"ce labels is \t{ce_labels}")
            # print(f"shape of onehot_Label_yt is \t {onehot_labels_yt.shape}")  # 这是E-prop算法需要的yt
            # print(f"onehot_Label_yt is \t {onehot_labels_yt}")  # 这是E-prop算法需要的yt
            data, onehot_labels_yt, ce_labels = data.to(device), onehot_labels_yt.to(device), ce_labels.to(device)
            optimizer.zero_grad()
            softmax_output, output = model(data, onehot_labels_yt, do_training)  # def forward(self, x, yt, do_training):
            if do_training:
                optimizer.step()

            # 将网络输出重塑为 [batch_size * n_t, output_features]
            output_reshaped = output.view(-1, output.size(-1))
            softmax_output_reshaped = softmax_output.view(-1, softmax_output.size(-1))

            # 将标签重塑为 [batch_size * n_t]
            label_reshaped = ce_labels.view(-1)
            # print(f"output_reshape is shape: {output_reshaped.shape}")
            # print(f"label_reshape is shape: {label_reshaped.shape}")

            # loss_fct = ['MSE', 'BCE', 'CE'] CE 需要 bto和bt 对于输出还得是没有经过softmax的，这里对应之前models中的vo
            # Compute the loss function, inference and score
            batch_loss = criterion(softmax_output_reshaped, label_reshaped)  # 这里实际上已经做了时间步长上的平均
            epoch_loss += batch_loss.item() * data.size(0)

            # 其中n_t是时间步数，n_b是批量大小，n_o是输出类别数
            # 输出实际上是 b t o
            # 在重塑后的网络输出上执行argmax操作，得到每个时间步的预测类别索引
            predictions = torch.argmax(softmax_output_reshaped, dim=1)

            # 计算正确预测的数量
            correct_predictions = torch.sum((predictions == label_reshaped).int()).item()

            # 更新总分数
            epoch_correct_predictions += correct_predictions

            # 计算准确率
            batch_samples = ce_labels.size(0) * ce_labels.size(1)  # 批次大小 × 时间步数
            total_samples += batch_samples
            batch_acc = correct_predictions / batch_samples * 100  # 每个批次上的正确率

            # writer.add_scalar('Batch/Accuracy', batch_acc, global_step)
            # writer.add_scalar('Batch/Loss', batch_loss.item(), global_step)
            #
            # global_step += 1

            # pbar.set_postfix({"batch_acc": batch_acc, "batch_loss": batch_loss.item()})
            # 每隔50次迭代调用update_plot函数
            # update_plot.update_plot(model, data, model.z, model.L, model.trace_in, model.trace_rec,
            #                             model.trace_out, model.h, fig, ax_list)
            # plt.show()

    # 计算整个epoch的平均准确率和平均loss
    avg_loss = epoch_loss / total_samples
    avg_accuracy = epoch_correct_predictions / total_samples * 100

    writer.add_scalar('Epoch/Accuracy', avg_accuracy, epoch)
    writer.add_scalar('Epoch/Loss', avg_loss, epoch)

    print(f"\t\t {'Training' if do_training else 'Validation'} Accuracy: {avg_accuracy:.2f}%, Average Loss: {avg_loss:.6f}")

    # 可视化权重直方图
    for name, param in model.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

    return avg_accuracy
