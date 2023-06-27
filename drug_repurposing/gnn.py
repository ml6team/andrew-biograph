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
    def __init__(self, num_conv_layers, hidden_channels, out_channels, heads=6, dropout=0.5):
        super().__init__()
        
        #self.conv_model = GATv2Conv()
        self.dropout = dropout
        self.num_conv_layers = num_conv_layers
        
        """# Add the input layer.
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv((-1,-1), hidden_channels, add_self_loops=False, heads=heads))
        
        # Add intermediate convolution layers.
        for l in range(self.num_conv_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels*heads, hidden_channels, add_self_loops=False, heads=heads))
        
        self.convs.append(GATv2Conv(hidden_channels*heads, hidden_channels, add_self_loops=False, heads=1))
        
        self.lin1 = Linear(hidden_channels, hidden_channels)
        #self.drop1 = nn.Dropout(self.dropout)
        self.lin2 = Linear(hidden_channels, 1)
        
        
        # Add the post provessing layers
        self.post_mp = Sequential(
            Linear(hidden_channels*heads, hidden_channels),
            nn.ELU(),
            nn.Dropout(self.dropout),
            Linear(hidden_channels, out_channels))"""
            
        
        self.conv1 = GATv2Conv((-1, -1), hidden_channels, add_self_loops=False, heads=heads)
        self.conv2 = GATv2Conv(hidden_channels*heads, hidden_channels, add_self_loops=False, heads=heads)
        #self.conv3 = GATv2Conv(hidden_channels*heads, hidden_channels, add_self_loops=False, heads=heads)
        self.conv3 = GATv2Conv(hidden_channels*heads, hidden_channels, add_self_loops=False, heads=heads)
        self.conv4 = GATv2Conv(hidden_channels*heads, hidden_channels, add_self_loops=False, heads=1)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(-1, out_channels)
        
        self.skip1 = Linear(-1, hidden_channels*heads)
        self.skip2 = Linear(-1, hidden_channels*heads)
        self.skip3 = Linear(-1, hidden_channels*heads)
        self.skip4 = Linear(-1, hidden_channels)

        

    def forward(self, x, edge_index):
        
        """for l in range(self.num_conv_layers):
            x = self.convs[l](x, edge_index)
        #x = self.post_mp(x)
        x = self.lin1(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)"""
        
        x = self.conv1(x, edge_index) + self.skip1(x)
        #x = F.elu(x)
        x = self.conv2(x, edge_index) + self.skip2(x)
        #x = F.elu(x)
        x = self.conv3(x, edge_index) + self.skip3(x)
        #x = F.elu(x)
        x = self.conv4(x, edge_index) + self.skip4(x)
        #x = self.conv5(x, edge_index) #+ self.skip4(x)
        #x = F.elu(x)
        x = self.lin1(x)
        x = F.elu(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        
        return x
    
    
        
        
        
        """
        x = self.conv1(x, edge_index)
        #x = F.elu(x)
        x = self.conv2(x, edge_index)
        #x = F.elu(x)
        x = self.conv3(x, edge_index)
        #x = F.elu(x)
        x = self.conv4(x, edge_index)
        #x = F.elu(x)
        x = self.lin1(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)"""
        

    
class DotProductLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(DotProductLinkPredictor, self).__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        
    def forward(self, x_i, x_j):
        """
        x = torch.cat([x_i, x_j], dim=-1)
        #x = x_i * x_j
        x = self.lin1(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin2(x)
        #x = torch.sigmoid(x)
        return x.squeeze()"""
        #x = (x_i * x_j).sum(-1) / (torch.norm(x_i) * torch.norm(x_j))
        x = (x_i * x_j).sum(-1)
        return x
        
    
    def reset_parameters(self):
      pass

    

  
    
    

    
    
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
            x = F.elu(x)
            x = self.conv2(x, edge_index)
            x = F.elu(x)
            x = self.conv3(x, edge_index)
            x = F.elu(x)

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





