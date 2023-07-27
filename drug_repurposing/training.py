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




def split_links(
        data, edge_type, rev_edge_type=None, 
        val_size=0.1, test_size=0.1, disjoint_train_ratio=0.0,
        neg_sampling_ratio=1.0, neg_train_samples=False,
        split_labels=False):
    """Split links into train, validation, and test sets.
    
    Parameters
    ----------
    data : torch_geometric.data.Data (or HeteroData)
        Data object containing graph to split.
    edge_type : tuple
        Tuple of form (node1, relation, node2) denoting edge-type to split
        among message-passing / supervision and among train, val, and 
        test sets.
    rev_edge_type : tuple
        Tuple of form (node2, relation, node1) denoting reverse edge-type 
        to split among message-passing / supervision and among train, val, and 
        test sets (default = None).
    val_size : float
        Portion of validation set in total data.
    test_size : float
        Portion of test set in total data.
    disjoint_train_ratio : float
        Portion of training-set links to use as supervision labels and not 
        message passing labels. If 0, all edges of <edge_type> will be used
        for both message-passing an supervison.
    neg_sampling_ratio : float
        Number of negative links to sample for each positive link.
    add_neg_train_samples : bool
        If False, negative-links will only be sampled for validation and test
        sets. Negative-train links can be sampled instead on the fly during 
        training (default=False).
    split_labels : bool
        If True, labels will be split into pos_labels and neg_labels
        (default = False).
        """
    
    # Instantiate RandomLinkSplit object. 
    transform = T.RandomLinkSplit(
        num_val=val_size,
        num_test=test_size,
        disjoint_train_ratio=disjoint_train_ratio,
        neg_sampling_ratio=neg_sampling_ratio,
        add_negative_train_samples=neg_train_samples,
        edge_types=edge_type,
        rev_edge_types=rev_edge_type, 
        is_undirected=True,
        split_labels=split_labels)
    
    # Split links into train, val, and test data.
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
        data, edge_type,
        num_neighbors = None, 
        neg_sampling_ratio=0.0, 
        batch_size=128, shuffle=False):
    """ Create mini-batches for link-prediction. 
    
    Parameters
    ----------
    data : torch_geometric.data.hetero_data.HeteroData
        Graph object to create minibatches for. 
    edge_type : tuple
        Tuple of form (node1, relation, node2) denoting edge-type to split
        among batches.
    num_neighbors : list or dict
        A subgraph will created for each node present in the supervision
        links by sampling up to num_neighbors. If a list is provied, the number of 
        nodes to sample for each khop is provided according to the lists indices. If 
        a dict is provided, subgraphs will be created for each node by sampling the 
        number provided in val for each edge-eype in key.
    neg_sampling_ratio : float
        The number of negative links to sample for each positive
        supervision edge for each batch (default=0.0).
    batch_size : int
        The number of supervision-links to include in each batch (default = 128). 
        This does not include additional links that are sampled on the fly.
    shuffle : bool
        If set to True, the data will be reshuffled at every epoch (default : False).
    
    Notes
    -----
        - Minibatching is useful especially if the entire graph will not fit into RAM. 
        - Each batch will contain <batch_size> supervision-edges of <edge_type>.
        - The neighborhood around the nodes in each supervison_edge wil be sampled 
          according to num_khop_neighbors.
    
    
    """
    
    # Define seed edges:
    edge_label_index = data[edge_type].edge_label_index
    edge_label = data[edge_type].edge_label
            
    # Create LinkNeighborLoader object.
    loader = LinkNeighborLoader(
        data=data,
        num_neighbors=num_neighbors,
        neg_sampling_ratio=neg_sampling_ratio,
        edge_label_index=(edge_type, edge_label_index),
        edge_label=edge_label,
        batch_size=batch_size,
        shuffle=shuffle)
    
    return loader








def train_LinkPredictor(
        model, loader, optimizer, evaluator, 
        device="cpu", train=True, batch_train=False):
    """
    Train or evaluate the LinkPredictor model on the provided data loader.

    Parameters
    ----------
    model : torch.nn.Module
        The LinkPredictor model to train or evaluate.
    loader : torch.utils.data.DataLoader
        The data loader containing batches of graph data for training or evaluation.
    optimizer : torch.optim.Optimizer
        The optimizer used to update the model parameters during training.
    evaluator : function
        A function that computes evaluation metrics based on model predictions.
    device : str, optional
        The device on which the model and data should be loaded (default is "cpu").
    train : bool, optional
        If True, the model is in training mode; if False, the model is in evaluation mode 
        (default is True).
    batch_train : bool, optional
        If True, for each epoch the model will be trained on a series of batches. If False,
        the entire graph will be loaded into memory and the model will be trained on a 
        full_batch each epoch. This will also require generating negative supervision 
        links for training on the fly (default = False).
    
    Returns
    -------
    epoch_loss : float
        The average loss per example for the entire epoch.
    epoch_auc : float
        The area under the ROC curve (AUC) for the entire epoch.
    hits_at : dict
        A dictionary containing hits@k scores for different values of k 
        (top-k accuracy) in the evaluation mode.
    
    Notes
    -----
    - This function performs training or evaluation on a LinkPredictor model. In 
      the training mode, it follows the negative sampling strategy to create 
      negative edges for each positive edge and computes the binary cross-entropy loss
      using logits. Gradients are backpropagated to update the model's parameters. In 
      the evaluation mode, it computes the loss and AUC for the entire epoch and also 
      calculates the hits@k scores using the evaluator function.
    - The function assumes that the LinkPredictor model has a forward method that takes 
      a batch of graph data as input and returns the predicted scores for each edge in 
      the batch. The model should be subclassed from torch.nn.Module.
    - The `evaluator` function should take two arguments: ground truth labels (y_true) 
      and predicted scores (y_pred), and return the hits@k scores for different values 
      of k.
    - During training, this function automatically manages data transfer between the CPU 
      and the specified device and frees up GPU memory to avoid excessive memory consumption.
    """
    
    # Instantiate epoch level metrics.
    total_loss = total_examples = total_auc = 0
    pos_pred = []
    neg_pred = []
    
    # Set the model to training or evaluation mode.
    if train:
        model.train()
    else:
        model.eval()
    
    # Set the iterator depending on whether performing batch-training or not.
    iterator = (loader if batch_train else [0])
       
    # Iterate over each batch in the loader.
    for batch in tqdm.tqdm(iterator):
        if not batch_train:
            batch = loader.clone()
        batch.to(device)
          
        # Reset the optimizer for each batch and send data to the device.
        optimizer.zero_grad()
        
        # If in training mode and full_batch training...
        if train and not batch_train:
            neg_edges = negative_sampling(
                    batch[model.edge_type].edge_label_index,
                    num_nodes = (
                            max(batch[model.edge_type].edge_label_index[0]+1),
                            max(batch[model.edge_type].edge_label_index[1]+1)),
                    num_neg_samples=25000)
        
            neg_labels = torch.zeros(len(neg_edges[0]))
            batch[model.edge_type].edge_label_index = torch.cat(
                    [batch[model.edge_type].edge_label_index, 
                     neg_edges], dim=-1)
            batch[model.edge_type].edge_label = torch.cat(
                    [batch[model.edge_type].edge_label, 
                     neg_labels],dim=-1)
    
        # Obtain node embeddings.
        pred = model(batch)

        # Compute the loss.
        y = batch[model.edge_type].edge_label
        loss = F.binary_cross_entropy_with_logits(pred, y)
        
        # If in training mode, backpropogate the gradients to update parameters.
        if train:
            loss.backward()
            optimizer.step()
        
        # Compute the AUROC.
        auc = get_auc(y, pred)
        
        # Add the loss / auc for this batch to the total loss / auc for the epoch.
        total_loss += float(loss) * pred.numel()
        total_auc += float(auc) * pred.numel()
        total_examples += pred.numel()

        # Record predictions for positive and negative examples.
        pos_ind = y.nonzero()
        neg_ind = (y == 0).nonzero()
        if len(pos_ind) > 1 and len(neg_ind) > 1:
            pos_pred.append(pred[pos_ind].squeeze().detach().cpu().numpy())
            neg_pred.append(pred[neg_ind].squeeze().detach().cpu().numpy())
    
        # Free up space on the GPU.
        del batch 
        torch.cuda.empty_cache()
    
    # Compute the hits@K metrics.
    pos_pred = np.concatenate(pos_pred, axis=0)
    pos_pred = torch.tensor(pos_pred).squeeze()
    neg_pred = np.concatenate(neg_pred, axis=0)
    neg_pred = torch.tensor(neg_pred).squeeze()
    hits_at = hits_at_k(pos_pred, neg_pred, evaluator)

    # Compute the loss and auc for the entire epoch.
    epoch_loss = (total_loss / total_examples)
    epoch_auc = (total_auc / total_examples)
    
    return epoch_loss, epoch_auc, hits_at
        
        
    
        
  
    

def train_GraphClassifier(model, loader, criterion, optimizer, device, evaluator, train=True):
    
    # Set the model mode.
    if train:
        model.train()
    else:
        model.eval()
    
    
    # Initialize metrics to 0. 
    total_loss = total_examples = total_correct = total_auc = 0
    pos_pred = []
    neg_pred = []
    
    # Iterate in batches over the data.
    for data in tqdm.tqdm(loader):
        data.to(device)
        optimizer.zero_grad() 
        data.x = data.x.to(torch.float32)
        pred, _ = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = F.binary_cross_entropy_with_logits(pred.squeeze(), data.y.squeeze())
        
        if train:
            loss.backward()
            optimizer.step()
        
        
        pos_ind = data.y.nonzero()[:, 0]
        neg_ind = (data.y == 0).nonzero()[:, 0]
        if len(pos_ind) > 1:
            pos_pred.append(pred[pos_ind].squeeze().detach().cpu().numpy())
        if len(neg_ind) > 1:
            neg_pred.append(pred[neg_ind].squeeze().detach().cpu().numpy())
        
        
        auc = get_auc(data.y, pred)
        total_loss += float(loss) * len(data)
        if auc != "NA":
            total_auc += float(auc) * len(data)
        
        total_examples += len(data)
    
    
    pos_pred = np.concatenate(pos_pred)
    neg_pred = np.concatenate(neg_pred)
    hits_at = hits_at_k(pos_pred, neg_pred, evaluator)

    epoch_loss = total_loss / total_examples
    epoch_auc = total_auc / total_examples
    
        
    return epoch_loss, epoch_auc, hits_at, model


        

        
def get_auc(y, pred_y):
    """
    Calculate the Area Under the Receiver Operating Characteristic Curve (AUC).

    The AUC is a measure of the ability of a binary classification model to distinguish
    between positive and negative examples. It provides a single scalar value that
    represents the probability that a randomly chosen positive example is ranked higher
    than a randomly chosen negative example.

    Parameters
    ----------
    y : torch.Tensor
        The true labels (ground truth) of the binary classification targets. A 1-D tensor
        of shape (num_samples,) containing binary values (0 or 1).

    pred_y : torch.Tensor
        The predicted logits or probabilities for the binary classification targets. A
        1-D tensor of shape (num_samples,) containing continuous values representing the
        model's confidence for positive class membership. Higher values indicate higher
        confidence of being a positive example.
    
    Returns
    -------
    auc : float
        The computed AUC value, ranging from 0.0 to 1.0. A value of 0.5 indicates a random
        model, while a value close to 1.0 indicates a strong classifier with good
        discrimination ability.
    
    Notes
    -----
    - This function should be used for binary classification problems.
    - This function internally converts the input tensors to NumPy arrays to compute the
      AUC using `roc_auc_score` from the `sklearn.metrics` module.

    Raises
    ------
    ValueError
        If the input tensors have incompatible shapes or data types.
    """
    
    y = y.cpu().detach().numpy()
    pred_y = pred_y.cpu().detach().numpy()
    try:
        auc = roc_auc_score(y, pred_y)
    except ValueError:
        auc = 0.5   
    return auc
    

    
    
def hits_at_k(y_pred_pos, y_pred_neg, evaluator):
    """
    Calculate the Hits@K metric for evaluating GNN performance.

    The Hits@K metric is used to assess the accuracy of link GNN models. It measures how
    often the true positives are ranked among the top K highest
    scored observations when considering both true positive and false negatives.
    
    Parameters
    ----------
    y_pred_pos : torch.Tensor
        A 1-D tensor of shape (num_pos_samples,) containing the predicted scores or logits
        for the true positive links (positive examples) in the link prediction task.

    y_pred_neg : torch.Tensor
        A 1-D tensor of shape (num_neg_samples,) containing the predicted scores or logits
        for the false negative links (negative examples) in the link prediction task.

    evaluator : torch_geometric.nn.Evaluator
        An evaluator object used to compute the Hits@K metric. The evaluator should be
        configured with the desired value of K for Hits@K computation.
    
    Returns
    -------
    dict
        A dictionary with the Hits@K values computed for each specified value of K. The
        keys of the dictionary represent the different K values, and the values represent
        the Hits@K scores.
    
    Notes
    -----
    - The input tensors `y_pred_pos` and `y_pred_neg` should contain the predicted scores
      or logits for positive and negative links, respectively.
    - The evaluator object `evaluator` should be configured with the desired K values for
      computing Hits@K. The evaluator is responsible for ranking and comparing the
      predictions.
    """
    
    # Initialize a dictionary to record the hits@k scores.
    hits_at_dict = {}
    
    # Evaluate the hits@k scores for a range of k values.
    for K in [20, 50, 100]:
        evaluator.K = K
        hits = evaluator.eval(
                {"y_pred_pos" : y_pred_pos,
                "y_pred_neg" : y_pred_neg})
        
        # Configure the dictionary for easy access later.
        hits_at_dict[K] = list(hits.values())[0]
    
    return hits_at_dict

    
   