import torch.nn as nn
import torch
#这个是生成器和判别器，都是只用线性层连的，我后面想加一个卷积层去保证局部平滑，但是那个文章里面说只用线性层就行了
class NetD(nn.Module):
    def __init__(self, dim):
        super(NetD, self).__init__()
        
        # First layer
        self.D_W1 = nn.Linear(dim, dim)
        nn.init.xavier_normal_(self.D_W1.weight)
        self.D_b1 = nn.LeakyReLU(0.2)
        
        # Second layer
        self.D_W2 = nn.Linear(dim, dim)
        nn.init.xavier_normal_(self.D_W2.weight)

        self.main = nn.Sequential(self.D_W1, self.D_b1, self.D_W2)
    def forward(self, x):
        return self.main(x)

class NetG(nn.Module):
    def __init__(self, dim):
        super(NetG, self).__init__()
        
        # First layer
        self.G_W1 = nn.Linear(dim*2, dim)
        nn.init.xavier_normal_(self.G_W1.weight)
        self.G_b1 = nn.ReLU()
        
        # Second layer
        self.G_W2 = nn.Linear(dim, dim)
        nn.init.xavier_normal_(self.G_W2.weight)
        self.G_b2 = nn.Tanh()
        
        # Combining layers sequentially
        self.main = nn.Sequential(self.G_W1, self.G_b1, self.G_W2, self.G_b2)

    def forward(self, z, m):
        return self.main(torch.cat((z, m), dim=1))