import torch
from torch_geometric.nn import (
        GAE, GCNConv, GATConv, GATv2Conv, GINConv, SAGEConv,
        Linear, TransformerConv, to_hetero, global_mean_pool, global_add_pool, global_max_pool)
import torch.nn as nn
from torch.nn import Sequential, BatchNorm1d, ReLU, ELU, Dropout
import torch.nn.functional as F
from torch_geometric.nn.pool import TopKPooling, ASAPooling, global_mean_pool, global_max_pool, global_add_pool


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
        x = F.elu(x)
        x = self.conv2(x, edge_index) + self.skip2(x)
        x = F.elu(x)
        x = self.conv3(x, edge_index) + self.skip3(x)
        x = F.elu(x)
        x = self.conv4(x, edge_index) + self.skip4(x)
        #x = self.conv5(x, edge_index) #+ self.skip4(x)
        x = F.elu(x)
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
        #x = self.lin1(x)
        #x = F.elu(x)
        #x = self.lin2(x).squeeze()
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
    def __init__(self, hidden_channels, num_classes, heads=3):
        super().__init__()
        
        
        self.conv1 = GATv2Conv((-1, -1), hidden_channels, add_self_loops=True, heads=heads, edge_dim=3)
        self.lin1 = Linear(hidden_channels*heads, hidden_channels)
        #self.drop1 = Dropout(0.3)
        self.bn1 = BatchNorm1d(hidden_channels)
        self.pool1 = ASAPooling(hidden_channels, ratio=0.8)
        
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, add_self_loops=True, heads=heads, edge_dim=3)
        self.lin2 = Linear(hidden_channels*heads, hidden_channels)
        #self.drop2 = Dropout(0.3)
        self.bn2 = BatchNorm1d(hidden_channels)
        self.pool2 = ASAPooling(hidden_channels, ratio=0.5)
        
        self.conv3 = GATv2Conv(hidden_channels, hidden_channels, add_self_loops=True, heads=heads, edge_dim=3)
        self.lin3 = Linear(hidden_channels*heads, hidden_channels)
        #self.drop3 = Dropout(0.3)
        self.bn3 = BatchNorm1d(hidden_channels)
        self.pool3 = ASAPooling(hidden_channels, ratio=0.2)
        
        self.linear_post1 = Linear(hidden_channels*2, 1024)
        self.bn4 = BatchNorm1d(1024)
        self.linear_post2 = Linear(1024, 1)
        
        

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index) 
        x = self.lin1(x).relu()
        #x = self.bn1(x)
        #x, edge_index, edge_attr, batch, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)
        
        x = self.conv2(x, edge_index) 
        x = self.lin2(x).relu()
        #x = self.bn2(x)
        #x, edge_index, edge_attr, batch, _ = self.pool2(x, edge_index, None, batch)
        
        x2 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)
        
        x = self.conv3(x, edge_index) 
        x = self.lin3(x).relu()
        #x = self.bn3(x)
        #x, edge_index, edge_attr, batch, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)
        
        
        x = x1 + x2 + x3
        x = self.linear_post1(x).relu()
        x = self.bn4(x)
        x = self.linear_post2(x)
        
        return x
        





class Transform_mol(torch.nn.Module):
    def __init__(self, hidden_channels, heads=4, dropout_rate=0.3):
        super().__init__()
    
        # Block1
        self.conv1 = TransformerConv(
                -1, 
                hidden_channels, 
                heads=heads, 
                dropout=0.0,
                beta=True)
        self.lin1 = Linear(hidden_channels*heads, hidden_channels)
        self.bn1 = BatchNorm1d(hidden_channels)
        self.pool1 = TopKPooling(hidden_channels, ratio=0.8)

        # Block2
        self.conv2 = TransformerConv(
                hidden_channels, 
                hidden_channels, 
                heads=heads, 
                dropout=0.0,
                beta=True)
        self.lin2 = Linear(hidden_channels*heads, hidden_channels)
        self.bn2 = BatchNorm1d(hidden_channels)
        self.pool2 = TopKPooling(hidden_channels, ratio=0.5)

        # Block3
        self.conv3 = TransformerConv(
                hidden_channels, 
                hidden_channels, 
                heads=heads, 
                dropout=0.0,
                beta=True)
        self.lin3 = Linear(hidden_channels*heads, hidden_channels)
        self.bn3 = BatchNorm1d(hidden_channels)
        self.pool3 = TopKPooling(hidden_channels, ratio=0.2)

        self.linear_post1 = Linear(hidden_channels*2, 1024)
        self.linear_post2 = Linear(1024, 2)
    
    
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index) 
        x = self.lin1(x).relu()
        x = self.bn1(x)
        x, edge_index, edge_attribut, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([global_max_pool(x, batch), global_add_pool(x, batch)], dim=1)
        
        x = self.conv2(x, edge_index) 
        x = self.lin2(x).relu()
        x = self.bn2(x)
        x, edge_index, edge_attribut, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([global_max_pool(x, batch), global_add_pool(x, batch)], dim=1)
        
        x = self.conv3(x, edge_index) 
        x = self.lin3(x).relu()
        x = self.bn3(x)
        x, edge_index, edge_attribut, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([global_max_pool(x, batch), global_add_pool(x, batch)], dim=1)
        
        x = x1 + x2 + x3
        x = self.linear_post1(x).relu()
        x = self.linear_post2(x)
        
        return x
    

    
    
    
class SAGE_mol(torch.nn.Module):
    def __init__(self, hidden_channels, num_classes):
        super().__init__()
        
        
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.pool1 = TopKPooling(hidden_channels, ratio=0.8)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.pool2 = TopKPooling(hidden_channels, ratio=0.5)
        self.linear_post1 = Linear(hidden_channels*2, 1024)
        self.linear_post2 = Linear(1024, 2)
        

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index) 
        x = torch.relu(x)
        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([global_max_pool(x, batch), global_add_pool(x, batch)], dim=1)
        
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([global_max_pool(x, batch), global_add_pool(x, batch)], dim=1)
        x = x1 + x2
        x = self.linear_post1(x).relu()
        x = self.linear_post2(x)
        
        
        
        
        return x, F.log_softmax(x, dim=1)