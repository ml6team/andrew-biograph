import numpy as np
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
            
            # Get validation metrics.
            #model.eval()
            #val_edge_label_index = val_batch[edge_types].edge_label_index
            #val_edge_feat_src = z_src[val_edge_label_index[0]]
            #val_edge_feat_dest = z_dest[val_edge_label_index[1]]
            #val_edge_preds = (val_edge_feat_src * val_edge_feat_dest).sum(dim=-1)
            #val_ground_truth = val_batch[edge_types].edge_label
            
            #print(val_ground_truth)
            #raise
            
            #val_loss = F.binary_cross_entropy_with_logits(val_edge_preds, val_ground_truth)
            #val_auc = get_auc(val_ground_truth, val_edge_preds)
            
            
            
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

        
def get_auc(y, pred_y):
    """Calculate auc.
    """
    
    y = y.detach().numpy()
    pred_y = pred_y.detach().numpy()
    auc = roc_auc_score(y, pred_y)
    
    return auc
    
    
   