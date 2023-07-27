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
parser.add_argument('--dataset',default='.')
parser.add_argument('--zchr',default='.')
parser.add_argument('--output',default='.')
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#%%
n_critic = 5 # number of additional iterations to train the critic
lambda_gp = 10
decay =0.600 # RMSProp optimizer hyper-parameter
momentum = 0.0000  # RMSProp optimizer hyper-parameter
epsilon = 1e-6
alpha =100
# alpha =100
Glr = 1e-4 #1e-3
Dlr = 5e-5
batch_size =512

#%%
mkdir(f'{output}/mig')
file = open(f'{output}/mig/G_loss.txt', 'w')
file = open(f'{output}/mig/D_loss.txt', 'w')

file = open(f'{output}/mig/G_grad.txt', 'w')
file = open(f'{output}/mig/D_grad.txt', 'w')
def WSGAIN_GP(data,netD,netG,models,stf,endf,or_data):

    def discriminator_loss(x,m,z):
        G_sample = netG(z,m)
        D_real = netD(x)
        D_fake = netD(G_sample)

        gradient_penalty = compute_gradient_penalty(netD,(m*x).detach(),((1-m)*G_sample).detach())
        # print(gradient_penalty)
        D_loss = torch.mean(m*D_real) - torch.mean((1-m)*D_fake) + gradient_penalty.to(device)

        with open(f'{output}/mig/D_loss.txt', 'a') as f:
            f.write(f'D_real:{torch.mean(m*D_real)}  D_fake:{- torch.mean((1-m)*D_fake)}  gradient_penalty:{gradient_penalty}   D_loss:{D_loss}\n')

        return D_loss

    def generator_loss(x,m,z,threshold=0.5):
        G_sample = netG(z,m)
        D_fake = netD(G_sample)
        #loss
        G_loss1 = -torch.mean((1-m)*D_fake)
        MSE_loss = torch.mean((m*x - m*G_sample)**2)/torch.mean(m)
        G_loss = G_loss1 + alpha * MSE_loss
        # G_loss = g1*G_loss1 + alpha * MSE_loss + beta * condition_loss
        # print(f'G_loss:{G_loss},{G_loss1},{alpha * MSE_loss},{beta * condition_loss}')
        with open(f'{output}/mig/G_loss.txt', 'a') as f:
            f.write(f'G_loss1:{G_loss1}  alpha:{alpha}  MSE_loss:{MSE_loss}   G_loss:{G_loss}\n')

        return G_loss,MSE_loss

    data = data.copy()
    data_miss = data
    
    data_mask = 1. - np.isnan(or_data)
    # data_mask = 1. - np.isnan(data) # 定义Mask矩阵(缺失数据为0,非缺失数据为1)
    data_miss = np.nan_to_num(data_miss, nan=-1)
    # optimizer_D = torch.optim.RMSprop(netD.parameters(),lr=Dlr,weight_decay=decay,momentum=momentum,eps=epsilon)
    # optimizer_G = torch.optim.RMSprop(netG.parameters(),lr=Glr,weight_decay=decay,momentum=momentum,eps=epsilon)
    optimizer_D = torch.optim.Adam(netD.parameters(),lr=Dlr, weight_decay=decay, eps=epsilon, betas=(0.90,0.999))
    optimizer_G = torch.optim.Adam(netG.parameters(),lr=Glr, weight_decay=decay, eps=epsilon, betas=(0.90,0.999))

    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=500, gamma=0.90)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=500, gamma=0.90)
    nof = data_miss.shape[0]

    models = output + '/' + 'models'

    sliding_start = 0
    # criterion = nn.BCELoss()
    current_batch_start = 0
    epochs = 10
    for epoch in range(epochs):
        for it in range(nof // batch_size):
            for _ in range(n_critic): # 每训练n_critic次判别器训练一次生成器
                batch_idx_start = current_batch_start
                batch_idx_end = current_batch_start + batch_size
                if batch_idx_end >= nof:  # If we reach the end of data, start from the beginning
                    batch_idx_start = 0
                    batch_idx_end = batch_size
                x_mb = data_miss[batch_idx_start:batch_idx_end,:]
                m_mb = data_mask[batch_idx_start:batch_idx_end,:]
                current_batch_start = batch_idx_end
                
                # z_mb = m_mb*x_mb + (1-m_mb) * sample_z(n_rows=batch_size, m_cols=m_dim)
                z_mb = m_mb * x_mb + (1-m_mb) * sample_z(n_rows=x_mb.shape[0], m_cols=m_dim)
                x_mb = torch.as_tensor(x_mb).float().to(device)
                m_mb = torch.as_tensor(m_mb).float().to(device)
                z_mb = torch.as_tensor(z_mb).float().to(device)
                #  c_mb = torch.as_tensor(c_mb).float().to(device)
                


                optimizer_D.zero_grad()
                D_loss_curr = discriminator_loss(x=x_mb,m=m_mb,z=z_mb).to(device)
                D_loss_curr.backward()
                optimizer_D.step()
                scheduler_D.step()
                for name, param in netD.named_parameters():
                    print(f'Gradient of {name} in Discriminator: {param.grad}',file=open(f'{output}/mig/D_grad.txt','a'))

            optimizer_G.zero_grad()
            G_loss_curr,MSE_loss_curr = generator_loss(x=x_mb, m=m_mb,z=z_mb)
            G_loss_curr.backward()
            optimizer_G.step()
            scheduler_G.step()

            for name, param in netG.named_parameters():
                print(f'Gradient of {name} in Generator: {param.grad}',file=open(f'{output}/mig/G_grad.txt','a'))

            # scheduler_G.step()

            # if it % 100 == 0:
            D_losses.append(D_loss_curr.item())
            G_losses.append(G_loss_curr.item())
            MSE_losses.append(MSE_loss_curr.item())

        if len(G_losses) > 0 and len(D_losses) > 0 and len(MSE_losses):
            plt.figure(figsize=(12, 6))
            plt.plot(G_losses, label='Generator Loss')
            plt.plot(D_losses, label='Discriminator Loss')
            plt.plot(MSE_losses, label='MSE Loss')
            plt.title(f'Losses for samples {epoch}:{stf}-{endf}')
            plt.xlabel('Iterations ')
            plt.ylabel('Loss')

            locs, labels = plt.xticks()
            new_labels = [item*100 for item in locs]
            plt.xticks(locs, new_labels)
            plt.legend()

        save_path = f"{output}/mig/{stf}_{endf}_{epoch}.png"  # Assuming 'output' is your desired directory
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
        else:
            new_arr = [re.sub(r"NA", "{:.3f}".format(stat_info[2]), i) for i in arr]
            data_lis.append(new_arr)
            last_line = new_arr
    elif stat_info[3] >= 1:
        data_lis.append(arr)
    k = k+1


miss_data = pd.DataFrame(data_lis, columns=header)

#%%

miss_data_np = miss_data.iloc[:,2:].replace('NA', np.nan).astype(float).to_numpy()

parser = argparse.ArgumentParser()
args = parser.parse_args([])
args.workers = 0


no,dim = miss_data_np.shape

args.no = no
args.dim = dim

batch_hf = 10000
nt = 3000

stf = 0
endf = 0
n_iterations = batch_hf // batch_size - 1
if n_iterations < nt:
    n_iterations = nt
m_dim = int(dim)
netD = NetD(m_dim).to(device)
netD.apply(weight_init)
netG = NetG(m_dim).to(device)
netG.apply(weight_init)


khf = 0
# from fancyimpute import KNN
from sklearn.impute import KNNImputer
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
    
    k = 5
    knnimputer = KNNImputer(n_neighbors=k, weights='distance')
    miss_data_filled_knn = knnimputer.fit_transform(miss_data_fh)
    # miss_data_filled_knn = KNN(k=k,weights='distance').fit_transform(miss_data_fh)

    miss_data_fh,imputed_pre,D_losses,G_losses,MSE_losses = WSGAIN_GP(miss_data_filled_knn,netD,netG,models,stf,endf,miss_data_fh)
    # miss_data_fh,imputed_pre,D_losses,G_losses,MSE_losses = WSGAIN_GP(miss_data_fh,netD,netG,models,stf,endf)

    if khf==0 or khf %50 ==0:
        if khf == 0:
            with open(f'{output}/mig/D_loss_.txt', 'w') as f:
                for item in D_losses:
                    f.write("%s\n" % item)

            with open(f'{output}/mig/G_loss_.txt', 'w') as f:
                for item in G_losses:
                    f.write("%s\n" % item)

            with open(f'{output}/mig/iter.txt', 'w') as f:
                f.write("%s\n" % n_iterations)
                f.write("%s\n" % use_cuda)
                f.write("%s\n" % device)
                f.write("%s\n" % torch.version.cuda)
                f.write("%s\n" % torch.cuda.device_count())   

    miss_data_fh = np.around(miss_data_fh, 3)
    imputed_pre = np.around(imputed_pre, 3)

    imputed_data = pd.concat([miss_data.iloc[stf:endf,0:2].reset_index(drop=True),pd.DataFrame(miss_data_fh)],axis=1,ignore_index=True)
    # imputed_data.columns =miss_data.columns[:].tolist()
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