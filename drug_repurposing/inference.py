import numpy as np
import pandas as pd
import torch


def get_graphClass_results(model, loader, device):
    
    # Instantiate empty lists to record results.
    name_list = []
    id_list = []
    score_list = []
    label_list = []
    
    # Iterate over the dataset.
    model.eval()
    for data in loader:
        
        # Get predictions.
        data.to(device)
        data.x = data.x.to(torch.float32)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        
        # Record names, ids, and labels.
        if hasattr(data, "name"):
            name_list += data.name
        else:
            name_list += ["NA"] * len(data.x)
        
        if hasattr(data, "id"):
            id_list += data.id
        else:
            id_list += ["NA"] * len(data.x)
        
        if data.y is not None:
            label_list += data.y.squeeze().cpu().detach()
        else:
            label_list += ["NA"] * len(data.x)

        score_list += out.squeeze().cpu().detach()
        
        # Delete this
        score_array = np.array(score_list)
        sort_ind = np.argsort(score_array)[::-1]
        
        
    # Sort the lists by the scores from highest to lowest.
    score_array = np.array(score_list)
    sort_ind = np.argsort(score_array)[::-1]
    sorted_ids = np.array(id_list)[sort_ind]
    sorted_names = np.array(name_list)[sort_ind]
    sorted_scores = np.array(score_list)[sort_ind]
    sorted_labels = np.array(label_list)[sort_ind]
    
    results_df = pd.DataFrame(list(zip(sorted_ids, sorted_names, sorted_scores, sorted_labels)),
               columns =["id", "name", "score", "label"])
    results_df["norm_score"] = (
            (results_df["score"] - results_df["score"].mean()) / results_df["score"].std())
    
    return results_df
