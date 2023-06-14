"""Loads the multiscale interactome data into a msi object.

Classes
-------
MSI:
"""


import sys
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, HeteroData
import torch_geometric.transforms as T



class MSI():
    """ The multiscale interactome.
    
    Attributes
    -----------
    name : str
        The name identifier of the specific msi object.
    mappings: dict
        Dictionary of dictionaries containing mappings between node ids, names, and gids
    """

    
    def __init__(self, name, num_features, data_dir, **kwargs):
        """Initialize the msi.
        
        Parameters
        ----------
        name : str
            The name identifier of the specific msi object
        """
    
        self.name = name
        self.num_features = num_features
        self.data_dir = data_dir
        self.mappings = dict()
        
        # Iterate over any provided keyword arguments. 
        self.extra_dz_data = None
        self.exclude_drug_dz_links = False
        for key, val in kwargs.items():
            if key == "extra_dz_data":
                self.extra_dz_data = val
            if key == "exclude_drug_dz_links":
                self.exclude_drug_dz_links = val
        
    def get_mapping(self, mapping_name, dataset, mapping, id_col, offset):
        """ Maps unique entites in a dataframe id column to unique graph ids (gids).
        
        Parameters
        ----------
        mapping_name : str
            Name of dictionary which can be accessed as key in mappings attribute.
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


    def get_name_mapping(self, mapping_name, dataset, id_col, name_col):
        """ Maps unique ids in dataframe id_col to names in name_col.

        Parameters
        ----------
        mapping_name : str
            Name of dictionary which can be accessed as key in mappings attribute.
        dataset : str
            Path to the tsv file to load into dataframe.
        id_col : str
            The column name in the dataset containing ids.
        name_col : str
            The column name in the dataset containing names.
        
        """

        df = pd.read_csv(dataset, sep="\t")
        mapping = pd.Series(df[name_col].values, index=df[id_col]).to_dict()
        if mapping_name in self.mappings.keys():
            self.mappings[mapping_name].update(mapping)
        else:
            self.mappings[mapping_name] = mapping


    def load_graph(self, features, edge_index, num_nodes):
        """ Creates a torch_geometric Data object as an msi attribute.

        features : array
            Array of features the same length as num_nodes.
        edge_index : array
            Array of edges representing pairwise relationships between nodes.
        num_nodes : int
            The number of nodes present in the entire graph. 
        """
        self.data = Data(x=features, edge_index=edge_index, num_nodes=num_nodes)
        
        # Validate that the graph has proper format and has been loaded successfully.
        self.data.validate(raise_on_error=True)
    
    
    def load_hetero_graph(self, 
                          drug_features, dz_features, prot_features, func_features,
                          drug_prot_edges, dz_prot_edges, prot_prot_edges, prot_func_edges, func_func_edges,
                          drug_dz_edges):
        
        # Load nodes with their corresponding types.
        self.hetero_data = HeteroData()
        self.hetero_data["prot"].node_id = torch.tensor(
                list(self.mappings["prot_id_gid_map"].values()), dtype=torch.int32)
        
        self.hetero_data["drug"].node_id = torch.tensor(
                list(self.mappings["drug_id_gid_map"].values()), dtype=torch.int32)
        
        self.hetero_data["dz"].node_id = torch.tensor(
                list(self.mappings["dz_id_gid_map"].values()), dtype=torch.int32)
        
        self.hetero_data["func"].node_id = torch.tensor(
                list(self.mappings["func_id_gid_map"].values()), dtype=torch.int32)
    
        # Load node features.
        self.hetero_data["prot"].x = prot_features
        self.hetero_data["drug"].x = drug_features
        self.hetero_data["dz"].x = dz_features
        self.hetero_data["func"].x = func_features
        
        # Load edges with their corresponding types.
        self.hetero_data["drug", "binds", "prot"].edge_index = drug_prot_edges
        self.hetero_data["dz", "implicates", "prot"].edge_index = dz_prot_edges
        self.hetero_data["prot", "associates", "prot"].edge_index = prot_prot_edges
        self.hetero_data["prot", "partOf", "func"].edge_index = prot_func_edges
        self.hetero_data["func", "partOf", "func"].edge_index = func_func_edges
        self.hetero_data["drug", "treats", "dz"].edge_index = drug_dz_edges
        
        # Make the graph undirected.
        self.hetero_data = T.ToUndirected()(self.hetero_data)
        
        # Validate that the graph has proper format and has been loaded successfully.
        self.hetero_data.validate(raise_on_error=True)
        



def get_edges(dataset, src_mapping, src_col, dest_mapping, dest_col, debug=False):
    """ Get the edges between source and destination nodes in a dataset.

    Parameters
    ----------
    dataset : str
        Path to the tsv file to load into dataframe.
    src_mapping : dict
        Dictionary containing mappings from ids in src_col to unique gids.
    src_col : str
        Name of column in dataset containing source ids.
    dest_mapping : dict
        Dictionary containing mappings from ids in dest_col to unique gids.
    src_col : str
        Name of column in dataset containing destination ids.

    Returns
    -------
    edge_index : torch.Tensor
        Tensor of size (2, source_nodes) showing relationships between nodes (represented by gids).
    """

    df = pd.read_csv(dataset, sep="\t")
    src_nodes = [src_mapping[src_id] for src_id in df[src_col]]
    
    dest_nodes = [dest_mapping[dest_id] for dest_id in df[dest_col]]
    edge_index = torch.tensor([src_nodes, dest_nodes])

    return edge_index






def generate_mappings(msi):
    """ Maps ids in the msi datasets to unique graph ids (gids).

    Parameters
    ----------
    msi : MSI object
        An instantiated msi object.
    """
    
    # Get protein id to gid mappings.
    offset = msi.get_mapping(
            "prot_id_gid_map", f"{msi.data_dir}/3_protein_to_protein.tsv", 
            {}, "node_1", 0)
    offset = msi.get_mapping(
            "prot_id_gid_map", f"{msi.data_dir}/3_protein_to_protein.tsv", 
            msi.mappings["prot_id_gid_map"], "node_2", 0)
    offset = msi.get_mapping(
            "prot_id_gid_map", f"{msi.data_dir}/1_drug_to_protein.tsv", 
            msi.mappings["prot_id_gid_map"], "node_2", offset)
    offset = msi.get_mapping(
            "prot_id_gid_map", f"{msi.data_dir}/2_indication_to_protein.tsv", 
            msi.mappings["prot_id_gid_map"], "node_2", offset)
    offset = msi.get_mapping(
            "prot_id_gid_map", f"{msi.data_dir}/4_protein_to_biological_function.tsv", 
            msi.mappings["prot_id_gid_map"], "node_1", offset)

    # Generate drug mapping.
    offset = msi.get_mapping(
            "drug_id_gid_map", f"{msi.data_dir}/1_drug_to_protein.tsv", 
            {}, "node_1", 0)
    
    if not msi.exclude_drug_dz_links:
        offset = msi.get_mapping(
            "drug_id_gid_map", f"{msi.data_dir}/6_drug_indication_df.tsv", 
            msi.mappings["drug_id_gid_map"], "drug", offset)


    # Generate disease mapping.
    offset = msi.get_mapping(
            "dz_id_gid_map", f"{msi.data_dir}/2_indication_to_protein.tsv", 
            {}, "node_1", 0)
    if not msi.exclude_drug_dz_links:
        offset = msi.get_mapping(
            "dz_id_gid_map", f"{msi.data_dir}/6_drug_indication_df.tsv", 
            msi.mappings["dz_id_gid_map"], "indication", offset)
    if msi.extra_dz_data is not None:
        offset = msi.get_mapping(
            "dz_id_gid_map", msi.extra_dz_data, 
            msi.mappings["dz_id_gid_map"], "node_1", offset)
        
        

    # Generate biological function mapping.
    offset = msi.get_mapping(
            "func_id_gid_map", f"{msi.data_dir}/4_protein_to_biological_function.tsv", 
            {}, "node_2", 0)
    offset = msi.get_mapping(
            "func_id_gid_map", f"{msi.data_dir}/5_biological_function_to_biological_function.tsv", 
            msi.mappings["func_id_gid_map"], "node_1", offset)
    offset = msi.get_mapping(
            "func_id_gid_map", f"{msi.data_dir}/5_biological_function_to_biological_function.tsv", 
            msi.mappings["func_id_gid_map"], "node_2", offset)




def generate_edge_index(msi):
    drug_prot_edges = get_edges(
            f"{msi.data_dir}/1_drug_to_protein.tsv",
            msi.mappings["drug_id_gid_map"], "node_1",
            msi.mappings["prot_id_gid_map"], "node_2")

    dz_prot_edges = get_edges(
            f"{msi.data_dir}/2_indication_to_protein.tsv",
            msi.mappings["dz_id_gid_map"], "node_1",
            msi.mappings["prot_id_gid_map"], "node_2")

    prot_prot_edges = get_edges(
            f"{msi.data_dir}/3_protein_to_protein.tsv",
            msi.mappings["prot_id_gid_map"], "node_1",
            msi.mappings["prot_id_gid_map"], "node_2")

    prot_func_edges = get_edges(
            f"{msi.data_dir}/4_protein_to_biological_function.tsv",
            msi.mappings["prot_id_gid_map"], "node_1",
            msi.mappings["func_id_gid_map"], "node_2")

    func_func_edges = get_edges(
            f"{msi.data_dir}/5_biological_function_to_biological_function.tsv",
            msi.mappings["func_id_gid_map"], "node_1",
            msi.mappings["func_id_gid_map"], "node_2")
    
    # Add in the extra edges from any additionally provided data.
    if msi.extra_dz_data is not None:
        temp = get_edges(
                msi.extra_dz_data, 
                msi.mappings["dz_id_gid_map"], "node_1",
                msi.mappings["prot_id_gid_map"], "node_2",
                debug=True)
        dz_prot_edges = torch.cat((dz_prot_edges, temp), dim=1)
                
    
    # Join all edges together.
    edge_index = torch.cat((
        drug_prot_edges, 
        dz_prot_edges, 
        prot_prot_edges, 
        prot_func_edges, 
        func_func_edges), 
        dim=1)
    
    
    if not msi.exclude_drug_dz_links:
        drug_dz_edges = get_edges(
            "../multiscale_interactome/data/6_drug_indication_df.tsv",
            msi.mappings["drug_id_gid_map"], "drug",
            msi.mappings["dz_id_gid_map"], "indication")
        edge_index = torch.cat((edge_index, drug_dz_edges), dim=1)   
    else:
        drug_dz_edges = None
    
    
    
    return (edge_index, drug_prot_edges, 
            dz_prot_edges, prot_prot_edges, 
            prot_func_edges, func_func_edges, drug_dz_edges)



def generate_name_mappings(msi):
    """ Generates the mappings between node ids and names.

    Parameters
    ----------
    msi : MSI object
        An instantiated msi object.
    """
    
    # Get id-name mappings for all node types in the msi.
    msi.get_name_mapping(
            "prot_id_name_map", f"{msi.data_dir}/3_protein_to_protein.tsv", "node_1", "node_1_name")
    msi.get_name_mapping(
            "prot_id_name_map", f"{msi.data_dir}/3_protein_to_protein.tsv", "node_2", "node_2_name")
    msi.get_name_mapping(
            "drug_id_name_map", f"{msi.data_dir}/1_drug_to_protein.tsv", "node_1", "node_1_name")
    msi.get_name_mapping(
            "dz_id_name_map", f"{msi.data_dir}/2_indication_to_protein.tsv", "node_1", "node_1_name")
    msi.get_name_mapping(
            "func_id_name_map", f"{msi.data_dir}/5_biological_function_to_biological_function.tsv", 
            "node_1", "node_1_name")
    msi.get_name_mapping(
            "func_id_name_map", f"{msi.data_dir}/5_biological_function_to_biological_function.tsv", 
            "node_2", "node_2_name")
    
    # Generate names for any extra data.
    if msi.extra_dz_data:
        msi.get_name_mapping("dz_id_name_map", msi.extra_dz_data, "node_1", "node_1_name")
        

    

def create_msi(name, **kwargs):
    """ Instantiate an msi object with name, mappings, and data attributes.
    
    Parameters
    ----------
    name : str
        The name identifier of the specific msi object.

    Returns
    -------
    msi : MSI object
        MSI objecet contatining the multiscale interactome.
    """
    
    msi = MSI(name, **kwargs)
    generate_mappings(msi)
    generate_name_mappings(msi)
    edge_index, drug_prot_edges, dz_prot_edges, prot_prot_edges, prot_func_edges, func_func_edges, drug_dz_edges = (
            generate_edge_index(msi))
    
    
    total_drug_nodes = len(msi.mappings["drug_id_gid_map"].values())
    total_dz_nodes = len(msi.mappings["dz_id_gid_map"].values())
    total_prot_nodes = len(msi.mappings["prot_id_gid_map"].values())
    total_func_nodes = len(msi.mappings["func_id_gid_map"].values())
    
    
    num_nodes = (
            total_drug_nodes + total_dz_nodes + total_prot_nodes + total_func_nodes)
    features = torch.ones((num_nodes, msi.num_features))
    
    drug_features = torch.ones((total_drug_nodes, msi.num_features))
    dz_features = torch.ones((total_dz_nodes, msi.num_features))
    prot_features = torch.ones((total_prot_nodes, msi.num_features))
    func_features = torch.ones((total_func_nodes, msi.num_features))
  
    
    msi.load_graph(features, edge_index, num_nodes)
    msi.load_hetero_graph(drug_features, dz_features, prot_features, func_features,  
                          drug_prot_edges, dz_prot_edges, prot_prot_edges, prot_func_edges, func_func_edges, 
                          drug_dz_edges)
    
    return msi
