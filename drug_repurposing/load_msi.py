"""Loads the multiscale interactome data into a msi object.

Classes
-------
MSI:
"""


import sys
import pandas as pd
import torch
from torch_geometric.data import Data


class MSI():
    """ The multiscale interactome.
    
    Attributes
    -----------
    name : str
        The name identifier of the specific msi object
    mappings: dict
        Dictionary of dictionaries containing mappings between node ids, names, and gids
    """

    
    def __init__(self, name):
        """Initialize the msi.
        
        Parameters
        ----------
        name : str
            The name identifier of the specific msi object
        """
        
        self.name = name
        self.mappings = dict()

        
    def get_mapping(self, mapping_name, dataset, mapping, id_col, offset):
        """ Maps unique entites in a dataframe id column to unique graph ids (gids).
        
        Parameters
        ----------
        mapping_name : str
            Name of dictionary which can be accessed as key in mappings attribute
        dataset : str
            Path to the tsv file to load into dataframe.
        mapping : dict
            Dictionary (can be empty) to load key-value pairs into
        id_col : str
            The column name in the dataset containing ids
        offset : int
            The starting gid to assign as value when mapping ids to gids.
        
        Returns
        -------
        offset : int
            The maximum gid in msi.mappings
        """
        
        # Map ids in id_col to gids starting with offset.
        df = pd.read_csv(dataset, sep="\t")
        i = 0
        for id in df[id_col].unique():
            if id in mapping.keys():
                continue
            else:
                mapping[id] = i + offset
                i += 1
        self.mappings[mapping_name] = mapping
        offset = max(mapping.values()) + 1
        return offset


    def get_name_mapping(self, mapping_name, dataset, key_col, val_col):
        """ Maps ids in 
        
        df = pd.read_csv(dataset, sep="\t")
        mapping = pd.Series(df[val_col].values, index=df[key_col]).to_dict()
        if mapping_name in self.mappings.keys():
            self.mappings[mapping_name].update(mapping)
        else:
            self.mappings[mapping_name] = mapping

    def load_graph(self, features, edge_index, num_nodes):
        self.data = Data(x=features, edge_index=edge_index, num_nodes=num_nodes)
        self.data.validate(raise_on_error=True) 


# Get the edge indices.
def get_edges(dataset, src_mapping, src_col, dest_mapping, dest_col):
    df = pd.read_csv(dataset, sep="\t")
    src_nodes = [src_mapping[src_id] for src_id in df[src_col]]
    dest_nodes = [dest_mapping[dest_id] for dest_id in df[dest_col]]
    edge_index = torch.tensor([src_nodes, dest_nodes])
    return edge_index
    """






def generate_mappings(msi):
    
    
    # Get protein id to gid mappings.
    offset = msi.get_mapping(
            "prot_id_gid_map", "../multiscale_interactome/data/3_protein_to_protein.tsv", 
            {}, "node_1", 0)
    offset = msi.get_mapping(
            "prot_id_gid_map", "../multiscale_interactome/data/3_protein_to_protein.tsv", 
            msi.mappings["prot_id_gid_map"], "node_2", offset)
    offset = msi.get_mapping(
            "prot_id_gid_map", "../multiscale_interactome/data/1_drug_to_protein.tsv", 
            msi.mappings["prot_id_gid_map"], "node_2", offset)
    offset = msi.get_mapping(
            "prot_id_gid_map", "../multiscale_interactome/data/2_indication_to_protein.tsv", 
            msi.mappings["prot_id_gid_map"], "node_2", offset)
    offset = msi.get_mapping(
            "prot_id_gid_map", "../multiscale_interactome/data/4_protein_to_biological_function.tsv", 
            msi.mappings["prot_id_gid_map"], "node_1", offset)

    # Generate drug mapping.
    offset = msi.get_mapping(
            "drug_id_gid_map", "../multiscale_interactome/data/1_drug_to_protein.tsv", 
            {}, "node_1", offset)

    # Generate disease mapping.
    offset = msi.get_mapping(
            "dz_id_gid_map", "../multiscale_interactome/data/2_indication_to_protein.tsv", 
            {}, "node_1", offset)

    # Generate biological function mapping.
    offset = msi.get_mapping(
            "func_id_gid_map", "../multiscale_interactome/data/4_protein_to_biological_function.tsv", 
            {}, "node_2", offset)
    offset = msi.get_mapping(
            "func_id_gid_map", "../multiscale_interactome/data/5_biological_function_to_biological_function.tsv", 
            msi.mappings["func_id_gid_map"], "node_1", offset)
    offset = msi.get_mapping(
            "func_id_gid_map", "../multiscale_interactome/data/5_biological_function_to_biological_function.tsv", 
            msi.mappings["func_id_gid_map"], "node_2", offset)




def generate_edge_index(msi):
    drug_prot_edges = get_edges(
            "../multiscale_interactome/data/1_drug_to_protein.tsv",
            msi.mappings["drug_id_gid_map"], "node_1",
            msi.mappings["prot_id_gid_map"], "node_2")

    dz_prot_edges = get_edges(
            "../multiscale_interactome/data/2_indication_to_protein.tsv",
            msi.mappings["dz_id_gid_map"], "node_1",
            msi.mappings["prot_id_gid_map"], "node_2")

    prot_prot_edges = get_edges(
            "../multiscale_interactome/data/3_protein_to_protein.tsv",
            msi.mappings["prot_id_gid_map"], "node_1",
            msi.mappings["prot_id_gid_map"], "node_2")

    prot_func_edges = get_edges(
            "../multiscale_interactome/data/4_protein_to_biological_function.tsv",
            msi.mappings["prot_id_gid_map"], "node_1",
            msi.mappings["func_id_gid_map"], "node_2")

    func_func_edges = get_edges(
            "../multiscale_interactome/data/5_biological_function_to_biological_function.tsv",
            msi.mappings["func_id_gid_map"], "node_1",
            msi.mappings["func_id_gid_map"], "node_2")

    edge_index = torch.cat((
        drug_prot_edges, 
        dz_prot_edges, 
        prot_prot_edges, 
        prot_func_edges, 
        func_func_edges), 
        dim=1)

    return edge_index



def generate_name_mappings(msi):
    msi.get_name_mapping(
            "prot_id_name_map", "../multiscale_interactome/data/3_protein_to_protein.tsv", "node_1", "node_1_name")
    msi.get_name_mapping(
            "prot_id_name_map", "../multiscale_interactome/data/3_protein_to_protein.tsv", "node_2", "node_2_name")
    msi.get_name_mapping(
            "drug_id_name_map", "../multiscale_interactome/data/1_drug_to_protein.tsv", "node_1", "node_1_name")
    msi.get_name_mapping(
            "dz_id_name_map", "../multiscale_interactome/data/2_indication_to_protein.tsv", "node_1", "node_1_name")
    msi.get_name_mapping(
            "func_id_name_map", "../multiscale_interactome/data/5_biological_function_to_biological_function.tsv", "node_1", "node_1_name")
    msi.get_name_mapping(
            "func_id_name_map", "../multiscale_interactome/data/5_biological_function_to_biological_function.tsv", "node_2", "node_2_name")

def create_msi():
    msi = MSI("drug_repurposing")
    generate_mappings(msi)
    generate_name_mappings(msi)
    edge_index = generate_edge_index(msi)
    num_features = 1
    num_nodes = (
            len(msi.mappings["prot_id_gid_map"]) + 
            len(msi.mappings["drug_id_gid_map"]) + 
            len(msi.mappings["dz_id_gid_map"]) + 
            len(msi.mappings["func_id_gid_map"]))
    features = torch.ones((num_nodes, num_features))
    msi.load_graph(features, edge_index, num_nodes)
    return msi
