import torch
from torch_geometric.nn import (
        GAE, GCNConv, GATConv, GATv2Conv, GINConv, 
        Linear, to_hetero)
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
    def __init__(self, hidden_channels, out_channels, heads=4):
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
    
    
    
    


# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_drug, x_prot, edge_label_index):
        # Convert node embeddings to edge-level representations:
        edge_feat_drug = x_drug[edge_label_index[0]]
        edge_feat_prot = x_prot[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_drug * edge_feat_prot).sum(dim=-1)















def gae_train(train_data, gae_model, optimizer, device):
    gae_model.train()
    optimizer.zero_grad()
    z = gae_model.encode(train_data.x, train_data.edge_index)
    loss = gae_model.recon_loss(z, train_data.pos_edge_label_index.to(device))
    loss.backward(retain_graph=True)
    optimizer.step()
    return float(loss)


@torch.no_grad()
def gae_test(test_data, gae_model):
    gae_model.eval()
    z = gae_model.encode(test_data.x, test_data.edge_index)
    return gae_model.test(z, test_data.pos_edge_label_index, test_data.neg_edge_label_index)

