import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
import torch.nn.functional as F
import tqdm
from torch_geometric.utils import negative_sampling
#from torchmetrics import HingeLoss
from torch_geometric.utils import one_hot
from torch.nn import BCELoss
import torch.nn as nn




def split_data(
        data, val_size, test_size, supervision_ratio,
        edge_types, rev_edge_types,
        neg_sampling_ratio, neg_train_samples=False):
    
    transform = T.RandomLinkSplit(
        num_val=val_size,
        num_test=test_size,
        disjoint_train_ratio=supervision_ratio,
        neg_sampling_ratio=neg_sampling_ratio,
        add_negative_train_samples=neg_train_samples,
        edge_types=edge_types,
        rev_edge_types=rev_edge_types, 
        is_undirected=True)
    train_data, val_data, test_data = transform(data)
    
    return train_data, val_data, test_data





def stratified_split(dataset, train_size, val_size, test_size):
    torch.manual_seed(42)
    dataset = dataset.shuffle()
    pos_ind = []
    neg_ind = []

    # Get the indices of the positive  and negative instances.
    for i, mol in enumerate(dataset):
        if mol.y == 1:
            pos_ind.append(i)
        else:
            neg_ind.append(i)
            
    # Obtain the maximum indices for each of the datasets.
    train_pos_obs = int(train_size * len(pos_ind))
    train_neg_obs = int(train_size * len(neg_ind))
    val_pos_obs = int(val_size * len(pos_ind))
    val_neg_obs = int(val_size * len(neg_ind))
    test_pos_obs = int(test_size * len(pos_ind))
    test_neg_obs = int(test_size * len(neg_ind))
    
    # Get the positive and negative indices for each of the dataset splits.
    train_ind = pos_ind[0 : train_pos_obs] + neg_ind[0 : train_neg_obs]
    val_ind = pos_ind[train_pos_obs : train_pos_obs + val_pos_obs] + (
            neg_ind[train_neg_obs : train_neg_obs + val_neg_obs])
    
    
    test_ind = pos_ind[train_pos_obs + val_pos_obs : ] + (
            neg_ind[train_neg_obs + val_neg_obs : ])
    
    
    # Divide the total dataset into train, test, and validation splits.
    train_data = dataset[train_ind]
    val_data = dataset[val_ind]
    test_data = dataset[test_ind]
    
    return train_data, val_data, test_data





def sample_data(dataset, ratio=0.5, method="down"):

    # Get the indices of the positive  and negative instances.
    pos_ind = []
    neg_ind = []
    for i, mol in enumerate(dataset):
        if mol.y == 1:
            pos_ind.append(i)
        else:
            neg_ind.append(i)
    
    if method == "down":
        sampled_neg_ind = list(np.random.choice(neg_ind, int(len(pos_ind) / ratio), replace=False))
        ind = pos_ind + sampled_neg_ind
    
    sampled_data = dataset[ind]
    
    return sampled_data








def create_minibatches(
        data, edge_types, 
        num_khop_neighbors = [20,10], neg_sampling_ratio=0.0, batch_size=128, shuffle=True):
    # In the first hop, we sample at most 20 neighbors.
    # In the second hop, we sample at most 10 neighbors.
    # In addition, during training, we want to sample negative edges on-the-fly with
    # a ratio of 2:1.
    # We can make use of the `loader.LinkNeighborLoader` from PyG:
    

    # Define seed edges:
    edge_label_index = data[edge_types].edge_label_index
    edge_label = data[edge_types].edge_label
    loader = LinkNeighborLoader(
        data=data,
        num_neighbors=num_khop_neighbors,
        neg_sampling_ratio=neg_sampling_ratio,
        edge_label_index=(edge_types, edge_label_index),
        edge_label=edge_label,
        batch_size=batch_size,
        shuffle=shuffle)
    
    return loader




    
    
    


def train_link_predictor(
        model, predictor, src_node_type, dest_node_type, edge_types,
        loader, optimizer, device, train=True, neg_sampling=False):
    
    #criterion = HingeLoss(task="binary").to(device)
    
    
    # Set the model to training or evaluation mode.
    if train:
        model.train()
        predictor.train()
    else:
        model.eval()
        predictor.eval()
    
    # Initialize epoch-level evaluation metrics to 0.
    total_loss = total_examples = total_auc = 0
    
    # Iterate over each batch in the loader.
    for batch in tqdm.tqdm(loader):
        
        # Reset the optimizer for each batch and send data to the device.
        optimizer.zero_grad()
        batch.to(device)

        # Obtain node embeddings.
        z = model(batch.x_dict, batch.edge_index_dict)
        
        # Retrieve embeddings for the nodes in the specific edge type.
        z_src = z[src_node_type]
        z_dest = z[dest_node_type]

        # Obtain predictions for specific edge type.
        edge_label_index = batch[edge_types].edge_label_index
        
        
        edge_feat_src = z_src[edge_label_index[0]]
        edge_feat_dest = z_dest[edge_label_index[1]]
        edge_preds = predictor(edge_feat_src, edge_feat_dest)
        ground_truth = batch[edge_types].edge_label
        
        # If performing negative sampling...
        """
        if negative_sampling:
            neg_edges = negative_sampling(batch[edge_types].edge_label_index,
                                         num_neg_samples=batch[edge_types].edge_label_index.size(1), method='dense')
            neg_preds = predictor(z[src_node_type][neg_edges[0]], z[dest_node_type][neg_edges[1]])
            edge_preds = torch.concat([edge_preds, neg_preds], dim=-1)
            neg_truth = torch.zeros(len(neg_preds)).to(device)
            ground_truth = torch.concat([ground_truth, neg_truth], dim=-1)
            raise
            #loss = -torch.log(edge_preds + 1e-15).mean() - torch.log(1 - neg_pred + 1e-15).mean()"""
       
        
        # Compute the loss over the batch,
        ####loss = F.binary_cross_entropy_with_logits(edge_preds, ground_truth)
        ####loss = torch.nn.BCELoss(edge_preds, ground_truth)
        loss = F.binary_cross_entropy_with_logits(edge_preds, ground_truth)
        
        
        # If in training mode, backpropogate the gradients to update parameters.
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            optimizer.step()
        
        # Compute the ROC-AUC
        auc = get_auc(ground_truth, edge_preds)
        if auc == "NA":
            print(edge_preds)
            print(ground_truth)
            auc = 0

        # Add the loss / auc for this batch to the total loss / auc for the epoch.
        total_loss += float(loss) * edge_preds.numel()
        total_auc += float(auc) * edge_preds.numel()
        total_examples += edge_preds.numel()
        
        # Free up space on the GPU.
        del batch 
        torch.cuda.empty_cache()
    
    # Compute the loss and auc for the entire epoch.
    epoch_loss = (total_loss / total_examples)
    epoch_auc = (total_auc / total_examples)
    return epoch_loss, epoch_auc
    

        


    
    
    

def train_graph_classifier(model, loader, criterion, optimizer, device, train=True):
    
    # Set the model mode.
    if train:
        model.train()
    else:
        model.eval()
    
    
    # Initialize metrics to 0. 
    total_loss = total_examples = total_correct = total_auc = 0
    
    # Iterate in batches over the data.
    for data in tqdm.tqdm(loader):
        data.to(device)
        optimizer.zero_grad() 
        data.x = data.x.to(torch.float32)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = F.binary_cross_entropy_with_logits(out.squeeze(), data.y.squeeze())
        #loss = criterion(out, labels)  
        
        #loss = BCELoss()(out.squeeze(), data.y.squeeze())
        
        if train:
            loss.backward()
            optimizer.step()
        
        #pred_class = out.argmax(dim=1)
        #correct = (pred_class == data.y.squeeze()).sum()
        auc = get_auc(data.y, out)
        total_loss += float(loss) * len(data)
        if auc != "NA":
            total_auc += float(auc) * len(data)
        
        #total_correct += correct
        total_examples += len(data)
    
    
    epoch_loss = total_loss / total_examples
    #epoch_acc = total_correct / total_examples
    epoch_auc = total_auc / total_examples
    
        
    return epoch_loss, epoch_auc, model


        

        
def get_auc(y, pred_y):
    """Calculate auc.
    """
    
    y = y.cpu().detach().numpy()
    pred_y = pred_y.cpu().detach().numpy()
    try:
        auc = roc_auc_score(y, pred_y)
    except ValueError:
        auc = "NA"
    
    return auc
    
    
   