import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
import model_origin
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import update_plot
import matplotlib.pyplot as plt
import numpy as np


# torch.set_printoptions(threshold=np.inf)


def train(args, device, train_loader, val_loader, test_loader):
    torch.manual_seed(42)  # the ultimate answer of the universe
    for trial in range(1, args.trials + 1):
        model = model_origin.SRNN(n_in=args.n_inputs,
                                n_rec=args.n_rec,
                                n_out=args.n_classes,
                                n_t=None,
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
    fig, ax_list = plt.subplots(nrows=8, figsize=(8, 10))
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
        optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    elif args.optimizer == 'NAG':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    elif args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    else:
        raise NameError("=== ERROR: optimizer " + str(args.optimizer) + " not supported")

    # Training and performance monitoring
    print("\n=== Starting model training with %d epochs: ===\n" % (args.epochs,))

    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs", position=0):

        print("\t Epoch " + str(epoch) + "...")

        # Training:
        train_acc, train_loss = train_epoch(args, True, model, device, train_loader, test_loader, val_loader,
                                            optimizer, 'train', writer, epoch, fig, ax_list)
        if args.use_wandb:
            wandb.log({"train_acc": train_acc, "train_loss": train_loss})
        # Validation
        if val_loader is not None:

            print("\t\t Validating...")

            val_acc, val_loss = train_epoch(args, False, model, device, train_loader, test_loader, val_loader,
                                            optimizer, 'val', writer, epoch, fig, ax_list)
            if args.use_wandb:
                wandb.log({"val_acc": val_acc, "val_loss": val_loss})
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), save_model_path)

                print("\t\t Saving model with Validation Accuracy: {:.2f}%".format(best_acc))

            else:
                # Save the model at the last epoch if no validation set
                torch.save(model.state_dict(), save_model_path)

                print("\t\t Saving model at the last epoch")
        model.writer.close()
        writer.close()


def onehot_convertor(labels, n_t, n_o):
    n_b = labels.size(0)
    one_hot_labels = torch.zeros(n_b, n_t, n_o, device=labels.device)
    mask = (labels != -1).float()
    labels = torch.clamp(labels, min=0)
    one_hot_labels.scatter_(2, labels.unsqueeze(-1), 1.0)
    one_hot_labels *= mask.unsqueeze(-1)

    return one_hot_labels


def train_epoch(args, do_training, model, device, train_loader, test_loader, val_loader, optimizer, benchType,
                writer, epoch, fig, ax_list):
    # model.eval()  # 没有自动梯度反向传播，这里也没有使用drop out和BN层，但是eval的计算是为了禁止梯度的更新
    model.train()
    epoch_loss = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    epoch_correct_predictions = 0

    # 根据 benchType 选择相应的数据加载器
    if benchType == 'train':
        loader = train_loader
    elif benchType == 'val':
        loader = val_loader
    else:
        loader = test_loader

    with torch.no_grad():  # 在训练过程中禁止梯度的自动变换，因为梯度的计算是hard code
        # For each batch
        pbar = tqdm(enumerate(loader), total=len(loader), desc="Batches", postfix={"batch_acc": 0.0, "loss": 0.0}, position=0)
        global_step = 0
        for batch_idx, (seq_lengths, data, label) in pbar:
            n_t = int(seq_lengths.max().item())
            onehot_labels_yt = onehot_convertor(label, n_t=n_t, n_o=int(args.n_classes))  # b , t , o
            ce_labels = label
            mask_model = (ce_labels != -1)

            data, onehot_labels_yt, ce_labels = data.to(device), onehot_labels_yt.to(device), ce_labels.to(device)
            optimizer.zero_grad()
            softmax_output, output = model(args, data, seq_lengths, onehot_labels_yt, do_training,
                                           mask_model)  # def forward(self, x, yt, do_training):

            model.grads_batch(args, data, softmax_output, onehot_labels_yt, global_step, mask=mask_model)
            optimizer.step()
            global_step += 1
            # 将网络输出重塑为 [batch_size * n_t, output_features]
            softmax_output_reshaped = softmax_output.view(-1, softmax_output.size(-1))

            # 将标签重塑为 [batch_size * n_t]
            ce_labels_reshaped = ce_labels.view(-1)

            # 掩码操作 防止影响最后的结果
            mask = (ce_labels_reshaped != -1)  # 创建一个掩码,忽略填充部分
            softmax_output_reshaped = softmax_output_reshaped[mask]  # 仅考虑非填充部分的输出
            ce_labels_reshaped = ce_labels_reshaped[mask]  # 仅考虑非填充部分的标

            # Compute the loss function, inference and score
            batch_loss = criterion(softmax_output_reshaped, ce_labels_reshaped)  # 这里实际上已经做了时间步长上的平均
            epoch_loss += batch_loss.item() * ce_labels_reshaped.size(0)

            # 其中n_t是时间步数，n_b是批量大小，n_o是输出类别数
            # 输出实际上是 b t o
            # 在重塑后的网络输出上执行argmax操作，得到每个时间步的预测类别索引
            predictions = torch.argmax(softmax_output_reshaped, dim=1)

            # 计算正确预测的数量
            correct_predictions = torch.sum((predictions == ce_labels_reshaped).int()).item()

            # 更新总分数
            epoch_correct_predictions += correct_predictions

            # 计算准确率
            batch_samples = mask.sum().item()  # 实际的样本数量
            total_samples += batch_samples
            batch_acc = correct_predictions / batch_samples * 100  # 每个批次上的正确率

            # writer.add_scalar('Batch/Accuracy', batch_acc, global_step)
            # writer.add_scalar('Batch/Loss', batch_loss.item(), global_step)
            #
            global_step += 1

            # pbar.set_postfix({"batch_acc": batch_acc, "batch_loss": batch_loss.item()})
            # 每隔50次迭代调用update_plot函数
            # if batch_idx % 50 == 0:
            #     update_plot.update_plot(model, data, model.z, model.L, model.trace_in, model.trace_rec,
            #                             model.trace_out, model.h, model.lowpassz, fig, ax_list)
            # plt.show()

    # 计算整个epoch的平均准确率和平均loss
    avg_loss = epoch_loss / total_samples
    avg_accuracy = epoch_correct_predictions / total_samples * 100

    writer.add_scalar('Epoch/Accuracy', avg_accuracy, epoch)
    writer.add_scalar('Epoch/Loss', avg_loss, epoch)

    print(
        f"\t\t {'Training' if do_training else 'Validation'} Accuracy: {avg_accuracy:.2f}%, Average Loss: {avg_loss:.6f}")

    # 可视化权重直方图
    for name, param in model.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

    return avg_accuracy, avg_loss
