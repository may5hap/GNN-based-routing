#this example is implemented based on pyg,make sure that you have successfully installed torch_geometric
#the dataset used in this example is 'Cora',please download the dataset in folder 'datasets' to your computer and 
#change the path which means you should replace the 'root' in line 26 with your own path.

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv   
from torch_geometric.datasets import Planetoid

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)         #defined a two-layer-GCN
 
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)                                         #activation func
        x = F.dropout(x, training=self.training)              #dropout
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)	

dataset = Planetoid(root = '/home/wanghao/gnn/datasets/Cora',name = 'Cora')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = dataset[0].to(device)                                  #the data used is the first data in the dataset,you can change the data what you like
optimizer = torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=5e-4)
model.train()

for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss= F.nll_loss(out[data.train_mask],data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    model.eval()
    _, pred = model(data).max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / int(data.test_mask.sum())
    print('Accuracy:{:.4f}'.format(acc))                      #for every epoch,we print the accuracy
