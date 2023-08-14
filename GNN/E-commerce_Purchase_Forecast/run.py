from torch_geometric.data import InMemoryDataset
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TopKPooling, SAGEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F

embed_dim = 128


# 数据只需要加载一次，保存在data即可，之后会调用数据中的processed_file_name查看文件夹中是否有当前的文件，如果有就直接加载，没有会进行process
# 粗看数据
# df = pd.read_csv('yoochoose-clicks.dat', header=None)
# df.columns = ['session_id', 'timestamp', 'item_id', 'category']
# buy_df = pd.read_csv('yoochoose-buys.dat', header=None)
# buy_df.columns = ['session_id', 'timestamp', 'item_id', 'price', 'quantity']
#
# item_encoder = LabelEncoder()   # LabelEncoder会把数据按照大小依次从0向后排序
# df['item_id'] = item_encoder.fit_transform(df.item_id)
# print(df.head())
# #数据有点多，选择其中一小部分来建模
# sampled_session_id = np.random.choice(df.session_id.unique(), 10000, replace=False)
# df = df.loc[df.session_id.isin(sampled_session_id)] # 在click中挑选sampled_id
# print(df.nunique())
# df['label'] = df.session_id.isin(buy_df.session_id)#    挑选buy中sampled_id的label
# print(df.head())

class YooChooseBinaryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(YooChooseBinaryDataset, self).__init__(root, transform, pre_transform)  # transform就是数据增强，对每一个数据都执行
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):  # 检查self.raw_dir目录下是否存在raw_file_names()属性方法返回的每个文件
        # 如有文件不存在，则调用download()方法执行原始文件下载
        return []

    @property
    def processed_file_names(self):  # 检查self.processed_dir目录下是否存在self.processed_file_names属性方法返回的所有文件，没有就会走process
        return ['yoochoose_click_binary_1M_sess.dataset']

    def download(self):
        pass

    def process(self):
        data_list = []
        # process by session_id
        grouped = df.groupby('session_id')
        for session_id, group in tqdm(grouped):
            sess_item_id = LabelEncoder().fit_transform(group.item_id)  # 将所有的节点变成从0开始的编码
            group = group.reset_index(drop=True)
            group['sess_item_id'] = sess_item_id
            node_features = group.loc[group.session_id == session_id, ['sess_item_id', 'item_id']].sort_values(
                'sess_item_id').item_id.drop_duplicates().values  # 从零开始编码之后，新的编码和之前编码的对应关系

            node_features = torch.LongTensor(node_features).unsqueeze(1)
            target_nodes = group.sess_item_id.values[1:]
            source_nodes = group.sess_item_id.values[:-1]

            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            x = node_features

            y = torch.FloatTensor([group.label.values[0]])

            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# 加载数据
dataset = YooChooseBinaryDataset(root='data/')
print(dataset)


#  模型定义
class Net(torch.nn.Module):  # 针对图进行分类任务
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = SAGEConv(embed_dim, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = SAGEConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = SAGEConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)
        self.item_embedding = torch.nn.Embedding(num_embeddings=len(dataset.data.x) + 10, embedding_dim=embed_dim)
        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch  # x:n*1,其中每个图里点的个数是不同的
        # print(x)
        x = self.item_embedding(x)  # n*1*128 特征编码后的结果
        # print('item_embedding',x.shape)
        x = x.squeeze(1)  # n*128
        # print('squeeze',x.shape)
        x = F.relu(self.conv1(x, edge_index))  # n*128
        # print('conv1',x.shape)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)  # pool之后得到 n*0.8个点
        # print('self.pool1',x.shape)
        # print('self.pool1',edge_index)
        # print('self.pool1',batch)
        # x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x1 = gap(x, batch)
        # print('gmp',gmp(x, batch).shape) # batch*128
        # print('cat',x1.shape) # batch*256
        x = F.relu(self.conv2(x, edge_index))
        # print('conv2',x.shape)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        # print('pool2',x.shape)
        # print('pool2',edge_index)
        # print('pool2',batch)
        # x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x2 = gap(x, batch)
        # print('x2',x2.shape)
        x = F.relu(self.conv3(x, edge_index))
        # print('conv3',x.shape)
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        # print('pool3',x.shape)
        # x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x3 = gap(x, batch)
        # print('x3',x3.shape)# batch * 256
        x = x1 + x2 + x3  # 获取不同尺度的全局特征

        x = self.lin1(x)
        # print('lin1',x.shape)
        x = self.act1(x)
        x = self.lin2(x)
        # print('lin2',x.shape)
        x = self.act2(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = torch.sigmoid(self.lin3(x)).squeeze(1)  # batch个结果
        # print('sigmoid',x.shape)
        return x


def train():
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data
        # print('data',data)
        optimizer.zero_grad()
        output = model(data)
        label = data.y
        loss = crit(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(dataset)


train_loader = DataLoader(dataset, batch_size=64)
# 模型初始化
model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
crit = torch.nn.BCELoss()
for epoch in range(10):
    print('epoch:', epoch)
    loss = train()
    print(loss)

quit()
# 如何制作自己的图数据？
import warnings

warnings.filterwarnings('ignore')
import torch
from torch_geometric.data import Data

# x表示每个点的特征，这里一共有四个点，每个点都是二维特征，y表示点的label
x = torch.tensor([[2, 1], [5, 6], [3, 7], [12, 0]], dtype=torch.float)
y = torch.tensor([0, 1, 0, 1], dtype=torch.float)
# 边的顺序定义无所谓，可以是终止点，起始点
edge_index = torch.tensor([[0, 1, 2, 0, 3],  # 起始点
                           [1, 0, 1, 3, 2]],  # 终止点
                          dtype=torch.long)
# 调用torch_geometric中的Data自己的图数据【需要点的特征矩阵，点的label和邻接矩阵即边的信息】
data = Data(x=x, y=y, edge_index=edge_index)
print(data)
