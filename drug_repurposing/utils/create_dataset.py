import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import os.path as osp
from tqdm import tqdm
import torch
from ogb.utils import mol
from torch_geometric.data import Data, Dataset, download_url


class MolDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        

    @property
    def raw_file_names(self):
        return "msi_drugs.csv"

    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0])
        return [f"data_{i}.pt" for i in list(self.data.index)]


        return ["data_1.pt"]

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0])
        for idx, row in self.data.iterrows():
            smiles = row["smiles"]
            name = row["activity"]
            label = row["HIV_active"]
            graph = mol.smiles2graph(smiles)
            
            if graph["edge_index"].shape == 0:
                continue

            d = Data()
            d.x = torch.tensor(graph["node_feat"])
            d.edge_index = torch.tensor(graph["edge_index"])
            d.edge_attr = torch.tensor(graph["edge_feat"])
            d.smiles = smiles
            d.y = torch.tensor(label)
            d.name = name
            torch.save(d, os.path.join(self.processed_dir, f"data_{idx}.pt"))

    def len(self):
        return self.data.shape[0]
        #return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data



