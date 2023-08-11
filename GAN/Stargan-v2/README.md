1. 权重
   训练过程中会保存三个权重，分别是:
   - ****_net.ckpt: 完整的模型权重，包括所有模块【generator，mapping network，style encoder，discriminator和fan】,**主要用于接着训练**
   - ****_net_ema：没有discriminator模块，只有转化部分的模型和权重，**主要用于验证和sample**
   - ****—optimis：没有fan模块
 
