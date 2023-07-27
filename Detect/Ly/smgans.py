#%%
from audioop import rms
from lib2to3.pgen2.driver import Driver
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
import os
import random
import time
import argparse
import pandas as pd
from model import NetD,NetG

# from model_bf import NetD,NetG
import matplotlib.pyplot as plt
#%%
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default=r'D:\deep_learning\Detect\Ly\P0.25_sm.txt')
parser.add_argument('--zchr',default='zchr5')
parser.add_argument('--output',default='./')
parser.add_argument('--models',default='.')

# parser.add_argument('--outfile',default='.')
opt = parser.parse_args()
dataset = opt.dataset
zchr = opt.zchr
output = opt.output
models = opt.models


def mkdir(dirpath):
    if os.path.exists(dirpath) is False:
        os.mkdir(dirpath)
mkdir(output)
#%%
def weight_init(m):
# 1. 根据网络层的不同定义不同的初始化方式  
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
# 也可以判断是否为conv2d，使用相应的初始化方式 
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def sample_M(mis,p):
    A = np.random.uniform(0., 1., size=[mis.shape[0],mis.shape[1]])
    B = A*mis > p
    C = 1. * B
    return C

# def sample_z(n_rows,m_cols,feature_range=(0, 1)): 
def sample_z(n_rows,m_cols,feature_range=(0.0, 1.0)): 
    return np.random.uniform(low=feature_range[0],high=feature_range[1],size=[n_rows, m_cols]) #(-0.01,0.01)随机采样

def sample_batch_index(total,batch_size):
    total_idx = np.random.permutation(total)
    batch_idx = total_idx[:batch_size]
    return batch_idx


def compute_gradient_penalty(D, real_samples, fake_samples):
    # Random weight term for interpolation between real and fake samples
    alpha = torch.FloatTensor(np.random.random((real_samples.size(0), 1))).to(device)
    alpha = alpha.expand(real_samples.size())
    
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(torch.FloatTensor(real_samples.shape[0], m_dim).fill_(1.0), requires_grad=False).to(device)
    # fake = Variable(torch.FloatTensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad_norm = torch.sqrt(epsilon + torch.sum(gradients ** 2, dim=1))
    gradient_penalty = lambda_gp * torch.mean((grad_norm - 1) ** 2)
    return gradient_penalty

#%% if __name__ == '__main__':
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

#%%
n_critic = 1 # number of additional iterations to train the critic
# lambda_gp = 100000 #Loss weight for gradient penalty
lambda_gp = 5
decay =0.600 # RMSProp optimizer hyper-parameter
momentum = 0.0000  # RMSProp optimizer hyper-parameter
epsilon = 1e-6
# alpha =0.8
alpha =10
Glr = 1e-4 #1e-3
Dlr = 5e-5

n_iterations =  int(np.ceil(3000))
# batch_size = 256
batch_size =512

#%%

def WSGAIN_GP(data,netD,netG,models,stf,endf,or_data):

    def discriminator_loss(x,m,z):
        G_sample = netG(z,m)
        D_real = netD(x)
        D_fake = netD(G_sample) 
        gradient_penalty = compute_gradient_penalty(netD,(m*x).data,((1-m)*G_sample).data)  #梯度惩罚，我也不是太懂这玩意
        # print(gradient_penalty)
        D_loss = torch.mean(m*D_real) - torch.mean((1-m)*D_fake) + gradient_penalty.to(device)
    
        return D_loss

    def generator_loss(x,m,z,threshold=0.5):
        G_sample = netG(z,m)
        D_fake = netD(G_sample)
        #loss
        G_loss1 = -torch.mean((1-m)*D_fake) # 生成的缺失数据
        MSE_loss = torch.mean((m*x - m*G_sample)**2)/torch.mean(m)
  
        G_loss = G_loss1 + alpha * MSE_loss
        return G_loss,MSE_loss

    data = data.copy()
    data_miss = data
    
    data_mask = 1. - np.isnan(or_data)
    # data_mask = 1. - np.isnan(data) # 定义Mask矩阵(缺失数据为0,非缺失数据为1)
    data_miss = np.nan_to_num(data_miss, nan=0.00)
    # optimizer_D = torch.optim.RMSprop(netD.parameters(),lr=Dlr,weight_decay=decay,momentum=momentum,eps=epsilon)
    # optimizer_G = torch.optim.RMSprop(netG.parameters(),lr=Glr,weight_decay=decay,momentum=momentum,eps=epsilon)
    optimizer_D = torch.optim.Adam(netD.parameters(),lr=Dlr, weight_decay=decay, eps=epsilon, betas=(0.50,0.999))
    optimizer_G = torch.optim.Adam(netG.parameters(),lr=Glr, weight_decay=decay, eps=epsilon, betas=(0.50,0.999))

    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=500, gamma=0.90)  #学习率衰减
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=500, gamma=0.75)
    nof = data_miss.shape[0]
    # if models != '.':
    #     n_iterations = 10
    #     n_critic = 5
    models = output + '/' + 'models'

    sliding_start = 0
    # criterion = nn.BCELoss()
    current_batch_start = 0
    epochs = 100
    D_losses = []
    G_losses = []
    MSE_losses = []
    for epoch in range(epochs):
        interation = nof // batch_size
        D_loss = 0
        G_loss = 0
        MSE_loss = 0
        for it in range(nof // batch_size):
            for _ in range(n_critic):  # 每训练n_critic次判别器训练一次生成器
                batch_idx_start = current_batch_start
                batch_idx_end = current_batch_start + batch_size
                if batch_idx_end >= nof:  # If we reach the end of data, start from the beginning
                    batch_idx_start = 0
                    batch_idx_end = batch_size
                x_mb = data_miss[batch_idx_start:batch_idx_end, :]
                m_mb = data_mask[batch_idx_start:batch_idx_end, :]
                current_batch_start = batch_idx_end

                # z_mb = m_mb*x_mb + (1-m_mb) * sample_z(n_rows=batch_size, m_cols=m_dim)
                z_mb = m_mb * x_mb + (1 - m_mb) * sample_z(n_rows=x_mb.shape[0], m_cols=m_dim)
                x_mb = torch.as_tensor(x_mb).float().to(device)
                m_mb = torch.as_tensor(m_mb).float().to(device)
                z_mb = torch.as_tensor(z_mb).float().to(device)
                #  c_mb = torch.as_tensor(c_mb).float().to(device)

                optimizer_D.zero_grad()
                D_loss_curr = discriminator_loss(x=x_mb, m=m_mb, z=z_mb).to(device)
                D_loss_curr.backward()
                optimizer_D.step()
                scheduler_D.step()
                for name, param in netD.named_parameters():
                    print(f'Gradient of {name} in Discriminator: {param.grad}',
                          file=open(f'{output}/mig/D_grad.txt', 'a'))

            optimizer_G.zero_grad()
            G_loss_curr, MSE_loss_curr = generator_loss(x=x_mb, m=m_mb, z=z_mb)
            G_loss_curr.backward()
            optimizer_G.step()
            scheduler_G.step()

            for name, param in netG.named_parameters():
                print(f'Gradient of {name} in Generator: {param.grad}', file=open(f'{output}/mig/G_grad.txt', 'a'))

            # scheduler_G.step()

            # if it % 100 == 0:

            D_loss += D_loss_curr.item()
            G_loss += G_loss_curr.item()
            MSE_loss += MSE_loss_curr.item()

        D_losses.append(D_loss/interation)
        G_losses.append(G_loss/interation)
        MSE_losses.append(MSE_loss/interation)

    if len(G_losses) > 0 and len(D_losses) > 0 and len(MSE_losses):
            plt.figure(figsize=(12, 6))
            plt.plot(G_losses, label='Generator Loss')
            plt.plot(D_losses, label='Discriminator Loss')
            plt.plot(MSE_losses, label='MSE Loss')
            plt.title(f'Losses for samples {epoch}:{stf}-{endf}')
            plt.xlabel('epoch ')
            plt.ylabel('Loss')

            locs, labels = plt.xticks()
            new_labels = [item  for item in locs]
            plt.xticks(locs, new_labels)
            plt.legend()

            save_path = f"{output}/mig/{stf}_{endf}_{epochs}.png"  # Assuming 'output' is your desired directory
            plt.savefig(save_path)

    #impute data
    mkdir(models)
    torch.save({'NetD_state_dict':netD.state_dict(),
                'NetG_state_dict':netG.state_dict(),
                'optimizerD_state_dict':optimizer_D.state_dict(),
                'optimizerG_state_dict':optimizer_G.state_dict(),
                },models+'/'+zchr +'Net.pth')


    z_all = data_mask*data_miss+(1-data_mask)*sample_z(nof,m_dim)

    z_all = torch.as_tensor(z_all).float().to(device)
    data_mask = torch.as_tensor(data_mask).float().to(device)

    imputed_data = netG(z=z_all,m=data_mask)
    imputed_data = imputed_data.cpu().detach().numpy()
    data_mask = data_mask.cpu().detach().numpy()

    imputed_data[imputed_data < 0] = 0.
    imputed_data[imputed_data > 1] = 1.
    imputed_pre = imputed_data
    imputed_data = data_mask * data_miss+ (1 - data_mask) * imputed_data

    return imputed_data,imputed_pre, D_losses, G_losses,MSE_losses
    
#%%
# 这里是心宇师兄说，补得太差是因为给进去的数据太少，然后他就自己先对数据做了点修改，我感觉有问题，但是先不管这个影响
import re
def get_max_min(list, count_threshold): #obtain max, min and other stat value for a list with NA
    remove_na_list = []
    non_na_count = 0
    for i in list:
        if i != "NA":
            remove_na_list.append(float(i))
            non_na_count = non_na_count + 1
    
    mean_value = ""
    max_value = ""
    min_value = ""
    if(non_na_count >= count_threshold):
        max_value = max(remove_na_list)
        min_value = min(remove_na_list)
        mean_value = sum(remove_na_list) / len(remove_na_list)
    else:
        max_value = "NA"
        min_value = "NA"
        mean_value = "NA"
    return(max_value, min_value, mean_value, non_na_count)

infile = dataset
inf = open(infile, "r")

header = inf.readline().rstrip().split("\t")

count_threshold = 2
range_threshold = 0.2
data_lis = []
need_gan_impute_count = 0
last_line = []
k = 0
for line in inf:
    line = line.rstrip()
    arr = line.split("\t")
    last_line = arr
    stat_info = get_max_min(arr[2:], count_threshold) #max, min, mean, na_count
    if stat_info[3] >= count_threshold: #only print the CGs (>= count_threshold)
        if stat_info[0] - stat_info[1] >= range_threshold:
            data_lis.append(arr)
            k=0
            # need_gan_impute_count = need_gan_impute_count + 1
        else:
            new_arr = [re.sub(r"NA", "{:.3f}".format(stat_info[2]), i) for i in arr]
            # if k<=20 and k !=0:
                # new_arr = ['NA' if (x != y and y != 'NA') else x for x, y in zip(arr, last_line)]
            data_lis.append(new_arr)
            last_line = new_arr
    elif stat_info[3] >= 1:
        data_lis.append(arr)
    k = k+1


miss_data = pd.DataFrame(data_lis, columns=header)

# data = MyDataset('/share/pub/liuy/SMgans/data_test/P0.25_sm.txt',args)
# print(need_gan_impute_count)
#%%

start =time.time()
# miss_path = dataset + f'/{zchr}_sm_data.txt'
# miss_path = dataset 
# miss_data = pd.read_csv(miss_path,sep='\t')
# miss_data_np = miss_data.iloc[:,2:].to_numpy()

miss_data_np = miss_data.iloc[:,2:].replace('NA', np.nan).astype(float).to_numpy()

parser = argparse.ArgumentParser()
args = parser.parse_args([])
args.workers = 0


no,dim = miss_data_np.shape

args.no = no
args.dim = dim
# data = MyDataset('/share/pub/liuy/SMgans/data_test/P0.25_sm.txt',args)

# 这里是因为我每次要是把整个矩阵一起放进去，就会过拟合，然后补出来的效果很差，我就把每个矩阵切成长度为batch_hf的样本给扔进网络重新训练
# batch_hf = 1500
batch_hf = 10000
nt = 3000  #保证迭代次数最小是3000

stf = 0
endf = 0
n_iterations = batch_hf // batch_size - 1
if n_iterations < nt:
    n_iterations = nt

# 初始化
m_dim = int(dim)
netD = NetD(m_dim).to(device)
netD.apply(weight_init)
netG = NetG(m_dim).to(device)
netG.apply(weight_init)

mkdir(f'{output}/mig')
khf = 0
# from fancyimpute import KNN
from sklearn.impute import KNNImputer  #这里我用了knn先补了一遍，可以不用这个
while stf < no:
    if stf + batch_hf <= no:
        endf = stf + batch_hf
    else:
        endf = no
    miss_data_fh = miss_data_np[stf:endf,:]
    
    netD = NetD(m_dim).to(device)
    netD.apply(weight_init)
    netG = NetG(m_dim).to(device)
    netG.apply(weight_init)
        
    G_losses = []
    D_losses = []
    MSE_losses = []
    
    # k = 5
    # knnimputer = KNNImputer(n_neighbors=k, weights='distance') # 这里是knn
    # miss_data_filled_knn = knnimputer.fit_transform(miss_data_fh)


    # miss_data_fh,imputed_pre,D_losses,G_losses,MSE_losses = WSGAIN_GP(miss_data_filled_knn,netD,netG,models,stf,endf,miss_data_fh) #要是用knn就用这个
    miss_data_fh,imputed_pre,D_losses,G_losses,MSE_losses = WSGAIN_GP(miss_data_fh,netD,netG,models,stf,endf,miss_data_fh)

    if khf==0 or khf %10 ==0:  #画图
        if khf == 0:
            with open(f'{output}/mig/D_loss.txt', 'w') as f:
                for item in D_losses:
                    f.write("%s\n" % item)

            with open(f'{output}/mig/G_loss.txt', 'w') as f:
                for item in G_losses:
                    f.write("%s\n" % item)

            with open(f'{output}/mig/iter.txt', 'w') as f:
                f.write("%s\n" % n_iterations)
                f.write("%s\n" % use_cuda)
                f.write("%s\n" % device)

        if len(G_losses) > 0 and len(D_losses) > 0 and len(MSE_losses):
            plt.figure(figsize=(12, 6))
            plt.plot(G_losses, label='Generator Loss')
            plt.plot(D_losses, label='Discriminator Loss')
            plt.plot(MSE_losses, label='MSE Loss')
            plt.title(f'Losses for samples {stf}-{endf}')
            plt.xlabel('Iterations ')
            plt.ylabel('Loss')

            locs, labels = plt.xticks()
            new_labels = [item*100 for item in locs]
            plt.xticks(locs, new_labels)
            plt.legend()

    save_path = f"{output}/mig/loss_plot_samples_{stf}_{endf}_{khf}.png"  # Assuming 'output' is your desired directory
    plt.savefig(save_path)

    miss_data_fh = np.around(miss_data_fh, 3)
    imputed_pre = np.around(imputed_pre, 3)

    imputed_data = pd.concat([miss_data.iloc[stf:endf,0:2].reset_index(drop=True),pd.DataFrame(miss_data_fh)],axis=1,ignore_index=True)

    imputed_data.columns = header
    imputed_pre = pd.concat([miss_data.iloc[stf:endf,0:2].reset_index(drop=True),pd.DataFrame(imputed_pre)],axis=1,ignore_index=True)
    imputed_pre.columns =header
    if stf == 0:
        imputed_data.to_csv(f'{output}/{zchr}_sm_imputed_data.txt',sep='\t',index=False,na_rep='NA')
        # imputed_pre.to_csv(f'{output}/{zchr}_sm_imputed_pre.txt',sep='\t',index=False,na_rep='NA')
    else:
        imputed_data.to_csv(f'{output}/{zchr}_sm_imputed_data.txt',mode = 'a',header=False,sep='\t',index=False,na_rep='NA')
        # imputed_pre.to_csv(f'{output}/{zchr}_sm_imputed_pre.txt',mode = 'a',header=False,sep='\t',index=False,na_rep='NA')
    
    stf = endf
    khf = khf + 1

#%%
"""
python D:/end_snap/Smgans/Smgans.py --dataset D:/snapmethy/data/split_wig/mx_dir --zchr chr3 --output D:/snapmethy/data/split_wig/out

python /share2/pub/liuy/liuy/endwork/Data/dmr_data/SMgans/py/wsgain_fb.py --dataset /share2/pub/liuy/liuy/endwork/Data/dmr_data/chr_data/mx_dir --zchr chr1 --output /share2/pub/liuy/liuy/endwork/Data/dmr_data/SMgans/impute
"""