# from tensorflow.python import pywrap_tensorflow
#
# path = 'D:/Python_code/007_fine_tuning/VGG16/vgg_16_2016_08_28/vgg_16.ckpt'
# reader = pywrap_tensorflow.NewCheckpointReader(path)
# param_dict = reader.get_variable_to_shape_map()
#
# for key, val in param_dict.items():
#     print(key, val)

# from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
#
# savedir = '/mnt/hdd1/scw/code/Mask4D/mask_4d/scripts/experiments/mask_4d/lightning_logs/version_116/checkpoints/last.ckpt'
#
# print_tensors_in_checkpoint_file(savedir, False, True)

import torch
# checkpoint = torch.load('/mnt/hdd1/scw/code/Mask4D/mask_4d/scripts/experiments/mask_4d/lightning_logs/version_116/checkpoints/last.ckpt')
# # print(checkpoint.keys())
# encoder_weights = {k: v for k, v in checkpoint["state_dict"].items() if k.startswith("encoder.")}
# decoder_weights = {k: v for k, v in checkpoint["state_dict"].items() if k.startswith("decoder.")}
#
# # 将权重保存到文本文件
# with open("encoder_decoder_weights.txt", "w") as f:
#     f.write("Encoder Weights:\n")
#     for key, value in encoder_weights.items():
#         f.write(f"{key}: {value}\n")
#
#     f.write("\nDecoder Weights:\n")
#     for key, value in decoder_weights.items():
#         f.write(f"{key}: {value}\n")
#
# print("Weights saved to encoder_decoder_weights.txt")


import torch

# 加载模型检查点
checkpoint = torch.load('/mnt/hdd1/scw/code/Mask4D/mask_4d/scripts/experiments/mask_4d/lightning_logs/version_116/checkpoints/last.ckpt')

# 将所有键保存到文本文件
with open("116ckpt.txt", "w") as f:
    for key in checkpoint["state_dict"].keys():
        f.write(f"{key}\n")

