import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
import torch.nn.functional as F
import tqdm



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
        rev_edge_types=rev_edge_types)
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
        num_khop_neighbors = [20,10], neg_sampling_ratio=2.0, batch_size=128):
    # In the first hop, we sample at most 20 neighbors.
    # In the second hop, we sample at most 10 neighbors.
    # In addition, during training, we want to sample negative edges on-the-fly with
    # a ratio of 2:1.
    # We can make use of the `loader.LinkNeighborLoader` from PyG:
    

    # Define seed edges:
    edge_label_index = data[edge_types].edge_label_index
    edge_label = data[edge_types].edge_label
    train_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=num_khop_neighbors,
        neg_sampling_ratio=neg_sampling_ratio,
        edge_label_index=(edge_types, edge_label_index),
        edge_label=edge_label,
        batch_size=batch_size,
        shuffle=True)
    
    return train_loader




def train_model(model, src_node_type, dest_node_type, edge_types,
                epochs, train_loader, val_data, optimizer, device):
    
    model.train()
    for epoch in range(1, epochs+1):
        total_loss = total_examples = total_auc = val_total_loss = val_total_auc = 0
        for train_batch in tqdm.tqdm(train_loader):
            # Reset the optimizer for each batch.
            optimizer.zero_grad()
            train_batch.to(device)
            
            # Obtain node embeddings.
            z = model(train_batch.x_dict, train_batch.edge_index_dict)
            
            # Retrieve embeddings for the nodes in the specific edge type.
            z_src = z[src_node_type]
            z_dest = z[dest_node_type]
            
            
            edge_label_index = train_batch[edge_types].edge_label_index
            edge_feat_src = z_src[edge_label_index[0]]
            edge_feat_dest = z_dest[edge_label_index[1]]
            edge_preds = (edge_feat_src * edge_feat_dest).sum(dim=-1)
            ground_truth = train_batch[edge_types].edge_label
            loss = F.binary_cross_entropy_with_logits(edge_preds, ground_truth)
            auc = get_auc(ground_truth, edge_preds)
            
            # Update the model parameters
            loss.backward()
            optimizer.step()
            
            # Add the loss / auc for this batch to the total loss / auc for the epoch.
            total_loss += float(loss) * edge_preds.numel()
            total_auc += float(auc) * edge_preds.numel()
            #val_total_loss += float(val_loss) * val_edge_preds.numel()
            #val_total_auc += float(val_auc) * val_edge_preds.numel()
            
            total_examples += edge_preds.numel()
        
        
        model.eval()
        z = model(val_data.x_dict, val_data.edge_index_dict)
    
            
        # Retrieve embeddings for the nodes in the specific edge type.
        z_src = z[src_node_type]
        z_dest = z[dest_node_type]
        val_edge_label_index = val_data[edge_types].edge_label_index
        val_edge_feat_src = z_src[val_edge_label_index[0]]
        val_edge_feat_dest = z_dest[val_edge_label_index[1]]
        val_edge_preds = (val_edge_feat_src * val_edge_feat_dest).sum(dim=-1)
        val_ground_truth = val_data[edge_types].edge_label
        val_loss = F.binary_cross_entropy_with_logits(val_edge_preds, val_ground_truth)
        val_auc = get_auc(val_ground_truth, val_edge_preds)
        
        
        print(f" Epoch: {epoch:03d}")
        print(f" Loss     : {total_loss / total_examples:<10.4f} |     AUC:     {total_auc / total_examples:>10.4f}")
        print(f" Val_Loss : {val_loss:<10.4f} |     Val_AUC: {val_auc:>10.4f}")


        


    
    
    

def train_graph_classifier(model, ohe, loader, criterion, optimizer, train=True):
    
    # Set the model mode.
    if train:
        model.train()
    else:
        model.eval()
    
    # Initialize metrics to 0. 
    total_loss = total_examples = total_correct = total_auc = 0
    
    # Iterate in batches over the data.
    for data in loader: 
        optimizer.zero_grad() 
        data.x = data.x.to(torch.float32)
        out = model(data.x, data.edge_index, data.batch)
        labels = torch.tensor(ohe.transform(data.y).toarray())
        loss = criterion(out, labels)  
        
        if train:
            loss.backward()
            optimizer.step()
        
        pred_class = out.argmax(dim=1)
        correct = (pred_class == data.y.squeeze()).sum()
        auc = get_auc(labels, out)
        total_loss += float(loss) * len(data)
        if auc != "NA":
            total_auc += float(auc) * len(data)
        
        total_correct += correct
        total_examples += len(data)
    
    
    epoch_loss = total_loss / total_examples
    epoch_acc = total_correct / total_examples
    epoch_auc = total_auc / total_examples
    
        
    return epoch_loss, epoch_acc, epoch_auc



def get_predictions(model, loader):
    model.eval()
    total_correct = total_examples = total_auc = 0
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data.x = data.x.to(torch.float32)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1) # Use the class with highest probability.
        print(pred)
        raise
        
        

    
        
        
        
        
def get_auc(y, pred_y):
    """Calculate auc.
    """
    
    y = y.detach().numpy()
    pred_y = pred_y.detach().numpy()
    try:
        auc = roc_auc_score(y, pred_y)
    except ValueError:
        auc = "NA"
    
    return auc
    
    
   