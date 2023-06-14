import torch
from torch_geometric.nn import (
        GAE, GCNConv, GATConv, GATv2Conv, GINConv, 
        Linear, to_hetero, global_mean_pool, global_add_pool, global_max_pool)
import torch.nn as nn
from torch.nn import Sequential, BatchNorm1d, ReLU, ELU
import torch.nn.functional as F

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, dropout):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_size, cached=True) # cached only for transductive learning
        self.conv2 = GCNConv(hidden_size, out_channels, cached=True) # cached only for transductive learning
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x_temp1 = self.conv1(x, edge_index).relu()
        x_temp2 = self.dropout(x_temp1)
        return self.conv2(x_temp2, edge_index)




class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, heads=1):
        super().__init__()
        self.conv1 = GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, heads=heads)
        self.lin1 = Linear(-1, hidden_channels)
        
        self.conv2 = GATv2Conv(hidden_channels*heads, hidden_channels, add_self_loops=False, heads=heads)
        self.lin2 = Linear(-1, hidden_channels)
        self.conv3 = GATv2Conv(hidden_channels*heads, out_channels, add_self_loops=False, heads=1)
        self.lin3 = Linear(-1, out_channels)
        
        

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) #+ self.lin1(x)
        #x = self.lin1(x)
        x = F.elu(x)
        x = self.conv2(x, edge_index) #+ self.lin2(x)
        #x = self.lin2(x)
        x = F.elu(x)
        x = self.conv3(x, edge_index)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        #x = F.elu(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        #x = self.lin2(x)

        
        return x

    
    
    
class GIN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GINConv(
            Sequential(Linear(-1, hidden_channels),
                       BatchNorm1d(hidden_channels), ELU(),
                       Linear(hidden_channels, hidden_channels), ELU()))
        self.conv2 = GINConv(
            Sequential(Linear(hidden_channels, hidden_channels),
                       BatchNorm1d(hidden_channels), ELU(),
                       Linear(hidden_channels, hidden_channels), ELU()))
        
        self.conv3 = GINConv(
            Sequential(Linear(hidden_channels, hidden_channels), 
                       BatchNorm1d(hidden_channels), ELU(),
                       Linear(hidden_channels, hidden_channels), ELU()))
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        
        
    def forward(self, x, edge_index):
            # Node embeddings 
            x = self.conv1(x, edge_index)
            x = self.conv2(x, edge_index)
            x = self.conv3(x, edge_index)

            # Classifier
            x = self.lin1(x)
            x = F.elu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.lin2(x)

            return x
    
    
    
    


class GIN_mol(torch.nn.Module):
    def __init__(self, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GINConv(
            Sequential(Linear(-1, hidden_channels),
                       BatchNorm1d(hidden_channels), ELU(),
                       Linear(hidden_channels, hidden_channels), ELU()))
        self.conv2 = GINConv(
            Sequential(Linear(hidden_channels, hidden_channels),
                       BatchNorm1d(hidden_channels), ELU(),
                       Linear(hidden_channels, hidden_channels), ELU()))
        
        self.conv3 = GINConv(
            Sequential(Linear(hidden_channels, hidden_channels), 
                       BatchNorm1d(hidden_channels), ELU(),
                       Linear(hidden_channels, hidden_channels), ELU()))
        
        
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)
        
        
    def forward(self, x, edge_index, batch):
            # Node embeddings 
            x = self.conv1(x, edge_index)
            x = self.conv2(x, edge_index)
            #x = self.conv3(x, edge_index)
            x = global_max_pool(x, batch)  # [batch_size, hidden_channels]

            # Classifier
            x = self.lin1(x)
            x = F.elu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.lin2(x)
            x = F.softmax(x, dim=1)

            return x
    





class GAT_mol(torch.nn.Module):
    def __init__(self, hidden_channels, num_classes, heads=4):
        super().__init__()
        self.conv1 = GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, heads=heads)
        self.lin1 = Linear(-1, hidden_channels)
        
        self.conv2 = GATv2Conv(hidden_channels*heads, hidden_channels, add_self_loops=True, heads=heads)
        self.lin2 = Linear(-1, hidden_channels)
        self.conv3 = GATv2Conv(hidden_channels*heads, hidden_channels, add_self_loops=True, heads=1)
        self.lin3 = Linear(-1, num_classes)
        
        

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index) #+ self.lin1(x)
        #x = self.lin1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index) #+ self.lin2(x)
        #x = self.lin2(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        x = F.softmax(x, dim=1)
        
        #x = F.elu(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        #x = self.lin2(x)

        
        return x





