import torch
from torchvision.utils import save_image
from spectral import Diffusion
# from utils import get_data
import argparse


import numpy as np
import torch
from torch import nn, optim
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from torch.utils.data import TensorDataset, DataLoader


parser = argparse.ArgumentParser()
args = parser.parse_args()
args.batch_size = 128
args.spectral_label_size = 161
args.conditional_size = 39

args.dataset_dir_path = "/home/gao/project/Diffusion-Models-pytorch/datasets/20200828_MP_dos"
args.data_path = args.dataset_dir_path + '/MP_comp39_dos161.npy'
args.test_idx = args.dataset_dir_path + '/test_idx.npy'
args.mean_path = args.dataset_dir_path + '/label_mean.npy'
args.std_path = args.dataset_dir_path + '/label_std.npy'



data = np.load(args.data_path, allow_pickle=True).astype(float)
test_idx = np.load(args.test_idx)
data = data[test_idx, :]   # (3432, 200)


condition= torch.Tensor(data[:, 0:args.conditional_size]) 
spectral_label = torch.Tensor(data[:, args.conditional_size:]) 
print('该数据集中前39维作为条件标签,后161维度作为生成标签，计算161维度的标签均值和方差并保存下来，测试时使用')
mean = torch.from_numpy(np.load(args.mean_path))
std = torch.from_numpy(np.load(args.std_path))
labels_standard = (spectral_label - mean) / (std + 1e-6)  # 标准化之后的标签，1e-6作为了平滑项



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Diffusion(device = device)


model_cpkt_path = "/home/gao/project/Diffusion-Models-pytorch/models/DDPM_conditional/"
model.load(model_cpkt_path)


gen_data = model.sample(use_ema=False,condition=condition[:100,:].to(device=device)).cpu()
MAE = torch.mean(torch.abs(gen_data-spectral_label[:100,:]))
# gen_data = model.sample(use_ema=True,condition=condition.to(device=device)).cpu()
# MAE = torch.mean(torch.abs(gen_data-spectral_label))
print('MAE:', MAE.numpy())

# save testing results
# np.save(args.dataset_dir_path+'/labels_standard.npy', labels_standard)
# np.save(args.dataset_dir_path+'/test_pred.npy', gen_data)

# print(spectral_label[:100,:].size())
# print(gen_data.size())








