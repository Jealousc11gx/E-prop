# import torch
# import matplotlib.pyplot as plt
# import setup
# import argparse
#
# parser = argparse.ArgumentParser(description='Spiking RNN Pytorch training for TIMIT')
# parser.add_argument('--cpu', action='store_true', default=False,
#                     help='Disable CUDA training and run training on CPU')
# args = parser.parse_args()
# (device, train_loader, val_loader, test_loader) = setup.setup(args)
#
#
# def log_transform_with_bias(x, bias=1.0):
#     return torch.log(x + bias)
#
#
# def min_max_normalize_with_scale(x, scale_factor=1.0):
#     min_val = torch.min(x)
#     max_val = torch.max(x)
#     normalized_x = (x - min_val) / (max_val - min_val)
#     return normalized_x * scale_factor
#
#
# def z_score_normalize_with_bias(x, bias=1.0):
#     mean = torch.mean(x)
#     std = torch.std(x)
#     normalized_x = (x - mean) / std
#     return normalized_x + bias
#
#
# def get_x(loader):
#     x_batch = []
#     for batch_idx, (data, label) in enumerate(loader):
#         x_batch.append(data)
#     return x_batch
#
#
# # 假设你的MFCC特征为x
# x_batches = get_x(test_loader)  # 将x替换为你实际的MFCC特征张量
# print(f"x shape is {len(x_batches)}")
#
# # 应用不同的特征转换方法
#
# for i, x in enumerate(x_batches):
#     print(f"length of x_flat is {len(x.flatten())}")
#     log_transformed_x = log_transform_with_bias(x, bias=1.0)
#     min_max_normalized_x = min_max_normalize_with_scale(x, scale_factor=10)
#     z_score_normalized_x = z_score_normalize_with_bias(x, bias=1.0)
#
#     fig1, axes1 = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
#     axes1 = axes1.flatten()
#
#     fig2, axes2 = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
#     axes2 = axes2.flatten()
#
#     # 遍历每个子图并绘制相应的数据
#     for i, (ax1, ax2, data, title) in enumerate(zip(
#             axes1, axes2,
#             [x, log_transformed_x, min_max_normalized_x, z_score_normalized_x],
#             ["Original MFCC Features", "Log Transformed Features", "Min-Max Normalized Features",
#              "Z-Score Normalized Features"])):
#
#         ax1.hist(data.flatten().cpu().numpy(), bins=50)
#         ax1.set_title(f"{title} Distribution")
#         ax1.set_xlabel("Value")
#         ax1.set_ylabel("Frequency")
#
#         data_flat = data.flatten().cpu().numpy()
#         ax2.plot(range(len(data_flat)), data_flat)
#         ax2.set_title(f"{title} (Flattened)")
#         ax2.set_xlabel("Sample Index")
#         ax2.set_ylabel("Feature Value")
#
#     fig1.subplots_adjust(hspace=0.5, wspace=0.4)
#     fig2.subplots_adjust(hspace=0.5, wspace=0.4)
#
#     # 显示图形
#     plt.show()
