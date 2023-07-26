"""Loads the multiscale interactome data into a msi object.

This module provides functionalities to load and represent multiscale interactome data as
a `MSI` object. The `MSI` object contains both homogeneous and heterogeneous graph data
representing various biological entities and their interactions.

Classes
-------
    * MSI 

Functions
---------
    * get_edges 
    * generate_mappings
    * generate_name_mappings
    * generate_edge_index
    * create msi
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
    data : torch_geometric.data.data.Data
        Data object containing a homogeneous graph.
    hetero_data : torch_geometric.data.hetero_data.HeteroData
        Data object containing a heterogeneous graph.
    num_features : int
        The number of dimensions of the features for nodes.
    mappings: dict
        Dictionary of dictionaries containing mappings between node ids, names, and gids.
    data_dir : str
        Path to the directory containing the data files used to create the graph.
    extra_drug_prot_data : str
        Path to the file containing any extra drug-protein relationships to 
        add to the graph (default = None).
    extra_drug_dz_data : str
        Path to the file containing any extra drug-disease relationships to 
        add to the graph (default = None).
    extra_dz_prot_data : str
        Path to the file containing any extra disease-protein relationships to 
        add to the graph (default = None).
    extra_prot_prot_data : str
        Path to the file containing any extra protein-protein relationships to 
        add to the graph (default = None).
    extra_prot_func_data : str
        Path to the file containing any extra protein-function relationships to 
        add to the graph (default = None).
    extra_func_func_data : str
        Path to the file containing any extra function-function relationships to 
        add to the graph (default = None).
    rest_gids : bool
        If True, gids start from 0 for each node type. If False, gids are entirely 
        unique across the entire graph.
        
    Methods
    -------
    get_mapping
        Set the mapping between node id and graph id (gid).
    get_name_mapping
        Set the mapping between node id and node name. 
    load_graph
        Create a homogeneous graph objcet, accessed via msi.data.
    load_hetero_graph
        Create a heterogeneous graph object, accessed via msi.hetero_data.
    """
    
    
    def __init__(
            self, name, num_features, data_dir, 
            reset_gids=True, **kwargs):
        """Initialize the msi.
        
        Parameters
        ----------
        name : str
            The name identifier of the specific msi object.
        num_features : int
            The number of dimensions of the features for nodes.
        data_dir : str
            Path to the directory containing the data files used to create the graph.
        rest_gids : bool
            If True, gids will start from 0 for each node type. Setting to False makes 
            each gid in the graph unique to a specific node, which is useful for 
            visualization.
        """
        
        # Initialize attributes.
        self.name = name
        self.num_features = num_features
        self.data_dir = data_dir
        self.reset_gids = reset_gids
        self.mappings = dict()
        
        # Iterate over any provided keyword arguments. 
        self.extra_drug_prot_data = None
        self.extra_drug_dz_data = None
        self.extra_dz_prot_data = None
        self.extra_prot_prot_data = None
        self.extra_prot_func_data = None
        self.extra_func_func_data = None
        for key, val in kwargs.items():
            if key == "extra_drug_prot_data":
                self.extra_drug_prot_data = val
            if key == "extra_drug_dz_data":
                self.extra_drug_dz_data = val
            if key == "extra_dz_prot_data":
                self.extra_dz_prot_data = val
            if key == "extra_prot_prot_data":
                self.extra_prot_prot_data = val
            if key == "extra_prot_func_data":
                self.extra_prot_func_data = val
            if key == "extra_func_func_data":
                self.extra_func_func_data = val
            
            
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
        """ Creates a torch_geometric Data object accessed via msi.data.
        
        Parameters
        ----------
        features : array
            Array of features the same length as num_nodes.
        edge_index : array
            Array of edges representing pairwise relationships between nodes.
        num_nodes : int
            The number of nodes present in the entire graph. 
        """
        self.data = Data(x=features, edge_index=edge_index, num_nodes=num_nodes)
        
        # Validate that the graph has proper format and has been loaded successfully.
        self.data.validate(raise_on_error=False)
    
    
    def load_hetero_graph(
            self, drug_features, dz_features, prot_features, func_features,
            drug_prot_edges,  drug_dz_edges, dz_prot_edges, prot_prot_edges, 
            prot_func_edges, func_func_edges):
        """ Creates a torch_geometric HeteroData object accessed via msi.hetero_data.
        
        Parameters
        ----------
        drug_features : torch.Tensor
            Features for drug nodes.
        dz_features : torch.Tensor
            Features for disease nodes.
        prot_features : torch.Tensor
            Features for protein nodes.
        func_features : torch.Tensor
            Features for function nodes.
        drug_prot_edges : torch.Tensor
            Edge index for drug-protein edges.
        drug_dz_edges : torch.Tensor
            Edge index for drug-disease edges.
        dz_prot_edges : torch.Tensor
            Edge index for disease-protein edges.
        prot_prot_edges : torch.Tensor
            Edge index for protein-protein edges.
        prot_func_edges : torch.Tensor
            Edge index for protein-function edges.
        func_func_edges : torch.Tensor
            Edge index for function-function edges.
        """
        
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
        self.hetero_data = T.AddSelfLoops()(self.hetero_data)
        
        # Validate that the graph has proper format and has been loaded successfully.
        self.hetero_data.validate(raise_on_error=False)
        


        
        
        
        

        
        
        
        
def get_edges(dataset, src_mapping, src_col, dest_mapping, dest_col):
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
        Tensor of size [2, source_nodes] showing relationships between nodes (represented by gids).
    """

    df = pd.read_csv(dataset, sep="\t")
    src_nodes = [src_mapping[src_id] for src_id in df[src_col]]
    dest_nodes = [dest_mapping[dest_id] for dest_id in df[dest_col]]
    edge_index = torch.tensor([src_nodes, dest_nodes])
    
    return edge_index




def generate_mappings(msi):
    """ Maps ids in the msi datasets to graph ids (gids).

    Parameters
    ----------
    msi : MSI object
        An instantiated msi object.
    """
    
    # Generate protein id-gid mapping.
    offset = msi.get_mapping(
            "prot_id_gid_map", f"{msi.data_dir}/3_protein_to_protein.tsv", 
            {}, "node_1", 0)
    offset = msi.get_mapping(
            "prot_id_gid_map", f"{msi.data_dir}/3_protein_to_protein.tsv", 
            msi.mappings["prot_id_gid_map"], "node_2", offset)
    offset = msi.get_mapping(
            "prot_id_gid_map", f"{msi.data_dir}/1_drug_to_protein.tsv", 
            msi.mappings["prot_id_gid_map"], "node_2", offset)
    offset = msi.get_mapping(
            "prot_id_gid_map", f"{msi.data_dir}/2_indication_to_protein.tsv", 
            msi.mappings["prot_id_gid_map"], "node_2", offset)
    offset = msi.get_mapping(
            "prot_id_gid_map", f"{msi.data_dir}/4_protein_to_biological_function.tsv", 
            msi.mappings["prot_id_gid_map"], "node_1", offset)
    if msi.extra_prot_prot_data is not None:
        offset = msi.get_mapping(
        "prot_id_gid_map", msi.extra_prot_prot_data, 
        msi.mappings["prot_id_gid_map"], "node_1", offset)
        offset = msi.get_mapping(
        "prot_id_gid_map", msi.extra_prot_prot_data, 
        msi.mappings["prot_id_gid_map"], "node_2", offset)
    if msi.extra_prot_func_data is not None:
        offset = msi.get_mapping(
        "prot_id_gid_map", msi.extra_prot_func_data, 
        msi.mappings["prot_id_gid_map"], "node_1", offset)
    
    
    # Generate drug id-gid mapping.
    offset = (0 if msi.reset_gids == True else offset)
    offset = msi.get_mapping(
            "drug_id_gid_map", f"{msi.data_dir}/1_drug_to_protein.tsv", 
            {}, "node_1", offset)
    offset = msi.get_mapping(
            "drug_id_gid_map", f"{msi.data_dir}/6_drug_indication_df.tsv", 
            msi.mappings["drug_id_gid_map"], "drug", offset)
    if msi.extra_drug_prot_data is not None:
        offset = msi.get_mapping(
        "drug_id_gid_map", msi.extra_drug_prot_data, 
        msi.mappings["drug_id_gid_map"], "node_1", offset)
    if msi.extra_drug_dz_data is not None:
        offset = msi.get_mapping(
            "drug_id_gid_map", msi.extra_drug_dz_data, 
            msi.mappings["drug_id_gid_map"], "drug", offset)
        
    # Generate disease id-gid mapping.
    offset = (0 if msi.reset_gids == True else offset)
    offset = msi.get_mapping(
            "dz_id_gid_map", f"{msi.data_dir}/2_indication_to_protein.tsv", 
            {}, "node_1", offset)
    offset = msi.get_mapping(
        "dz_id_gid_map", f"{msi.data_dir}/6_drug_indication_df.tsv", 
        msi.mappings["dz_id_gid_map"], "indication", offset)
    if msi.extra_dz_prot_data is not None:
        offset = msi.get_mapping(
            "dz_id_gid_map", msi.extra_dz_prot_data, 
            msi.mappings["dz_id_gid_map"], "node_1", offset)
    if msi.extra_drug_dz_data is not None:
        offset = msi.get_mapping(
            "dz_id_gid_map", msi.extra_drug_dz_data, 
            msi.mappings["dz_id_gid_map"], "indication", offset)
        
    # Generate biological function id-gid mapping.
    offset = (0 if msi.reset_gids == True else offset)
    offset = msi.get_mapping(
            "func_id_gid_map", f"{msi.data_dir}/4_protein_to_biological_function.tsv", 
            {}, "node_2", offset)
    offset = msi.get_mapping(
            "func_id_gid_map", f"{msi.data_dir}/5_biological_function_to_biological_function.tsv", 
            msi.mappings["func_id_gid_map"], "node_1", offset)
    offset = msi.get_mapping(
            "func_id_gid_map", f"{msi.data_dir}/5_biological_function_to_biological_function.tsv", 
            msi.mappings["func_id_gid_map"], "node_2", offset)
    if msi.extra_prot_func_data is not None:
        offset = msi.get_mapping(
            "func_id_gid_map", msi.extra_prot_func_data, 
            msi.mappings["func_id_gid_map"], "node_2", offset)
    if msi.extra_func_func_data is not None:
        offset = msi.get_mapping(
            "func_id_gid_map", msi.extra_func_func_data, 
            msi.mappings["func_id_gid_map"], "node_1", offset)
        offset = msi.get_mapping(
            "func_id_gid_map", msi.extra_func_func_data, 
            msi.mappings["func_id_gid_map"], "node_2", offset)


        
        
def generate_edge_index(msi):
    """ Get the edges between nodes saved in the data files.
    
    Parameters
    ----------
    msi : MSI object
        An instantiated msi object.
    
    Returns
    -------
    edge_index : torch.Tensor
        Tensor of size [2, 'number of edges'] containing all edges loaded from files.
    drug_prot_edges : torch.Tensor
        Tensor of size [2, 'number drug-prot edges'] containing drug-protein edges. Drug
        gids are accessed with index 0, and protein gids are accessed via index 1. 
    drug_dz_edges : torch.Tensor
        Tensor of size [2, 'number drug-dz edges'] containing drug-disease edges. Drug
        gids are accessed with index 0, and disease gids are accessed via index 1.
    dz_prot_edges : torch.Tensor
        Tensor of size [2, 'number drug-prot edges'] containing disease-protein edges. 
        Disease gids are accessed with index 0, and protein gids are accessed via index 1.
    prot_prot_edges : torch.Tensor
        Tensor of size [2, 'number prot-prot edges'] containing protein-protein edges. 
        Protein gids are accessed with both index 0 and index 1.
    prot_func_edges : torch.Tensor
        Tensor of size [2, 'number prot-func edges'] containing protein-function edges. 
        Protein gids are accessed with index 0, and function gids are accessed via index 1.
    func_func_edges : torch.Tensor
        Tensor of size [2, 'number func-func edges'] containing function-function edges. 
        Function gids are accessed with both index 0 and index 1.  
    """
    
    # Obtain the edge indices for each edge type.
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
    
    drug_dz_edges = get_edges(
            f"{msi.data_dir}/6_drug_indication_df.tsv",
            msi.mappings["drug_id_gid_map"], "drug",
            msi.mappings["dz_id_gid_map"], "indication")
    
    # Obtain the extra edge indicesfrom any additionally provided data.
    if msi.extra_drug_prot_data is not None:
        temp = get_edges(
                msi.extra_drug_prot_data, 
                msi.mappings["drug_id_gid_map"], "node_1",
                msi.mappings["prot_id_gid_map"], "node_2")
        drug_prot_edges = torch.cat((drug_prot_edges, temp), dim=1)
    
    if msi.extra_drug_dz_data is not None:
        temp = get_edges(
                msi.extra_drug_dz_data, 
                msi.mappings["drug_id_gid_map"], "drug",
                msi.mappings["dz_id_gid_map"], "indication")
        drug_dz_edges = torch.cat((drug_dz_edges, temp), dim=1)
    
    if msi.extra_dz_prot_data is not None:
        temp = get_edges(
                msi.extra_dz_prot_data, 
                msi.mappings["dz_id_gid_map"], "node_1",
                msi.mappings["prot_id_gid_map"], "node_2")
        dz_prot_edges = torch.cat((dz_prot_edges, temp), dim=1)
        
    if msi.extra_prot_prot_data is not None:
        temp = get_edges(
                msi.extra_prot_prot_data, 
                msi.mappings["prot_id_gid_map"], "node_1",
                msi.mappings["prot_id_gid_map"], "node_2")
        prot_prot_edges = torch.cat((prot_prot_edges, temp), dim=1)
    
    if msi.extra_prot_func_data is not None:
        temp = get_edges(
                msi.extra_prot_func_data, 
                msi.mappings["prot_id_gid_map"], "node_1",
                msi.mappings["func_id_gid_map"], "node_2")
        prot_func_edges = torch.cat((prot_func_edges, temp), dim=1)
    
    if msi.extra_func_func_data is not None:
        temp = get_edges(
                msi.extra_func_func_data, 
                msi.mappings["func_id_gid_map"], "node_1",
                msi.mappings["func_id_gid_map"], "node_2")
        func_func_edges = torch.cat((func_func_edges, temp), dim=1)
   
    # Join all edges together.
    edge_index = torch.cat(
            (drug_prot_edges, drug_dz_edges, dz_prot_edges, 
             prot_prot_edges, prot_func_edges, func_func_edges), 
            dim=1)
    
    return (edge_index, drug_prot_edges, drug_dz_edges,
            dz_prot_edges, prot_prot_edges, 
            prot_func_edges, func_func_edges)




def generate_name_mappings(msi):
    """ Generates the mappings between node ids and names.

    Parameters
    ----------
    msi : MSI object
        An instantiated msi object.
    """
    
    # Get id-name mappings for all node types in the msi.
    msi.get_name_mapping(
            "drug_id_name_map", f"{msi.data_dir}/1_drug_to_protein.tsv", "node_1", "node_1_name")
    msi.get_name_mapping(
            "prot_id_name_map", f"{msi.data_dir}/1_drug_to_protein.tsv", "node_2", "node_2_name")
    msi.get_name_mapping(
            "dz_id_name_map", f"{msi.data_dir}/2_indication_to_protein.tsv", "node_1", "node_1_name")
    msi.get_name_mapping(
            "prot_id_name_map", f"{msi.data_dir}/2_indication_to_protein.tsv", "node_2", "node_2_name")
    msi.get_name_mapping(
            "prot_id_name_map", f"{msi.data_dir}/3_protein_to_protein.tsv", "node_1", "node_1_name")
    msi.get_name_mapping(
            "prot_id_name_map", f"{msi.data_dir}/3_protein_to_protein.tsv", "node_2", "node_2_name")
    msi.get_name_mapping(
            "prot_id_name_map", f"{msi.data_dir}/4_protein_to_biological_function.tsv", "node_1", "node_1_name")
    msi.get_name_mapping(
            "func_id_name_map", f"{msi.data_dir}/4_protein_to_biological_function.tsv", "node_2", "node_2_name")
    msi.get_name_mapping(
            "func_id_name_map", f"{msi.data_dir}/5_biological_function_to_biological_function.tsv", 
            "node_1", "node_1_name")
    msi.get_name_mapping(
            "func_id_name_map", f"{msi.data_dir}/5_biological_function_to_biological_function.tsv", 
            "node_2", "node_2_name")
    msi.get_name_mapping(
            "drug_id_name_map", f"{msi.data_dir}/6_drug_indication_df.tsv", "drug", "drug_name")
    msi.get_name_mapping(
            "dz_id_name_map", f"{msi.data_dir}/6_drug_indication_df.tsv", "indication", "indication_name")
    
    # Generate names for any extra data.
    if msi.extra_drug_prot_data:
        msi.get_name_mapping("drug_id_name_map", msi.extra_drug_prot_data, "node_1", "node_1_name")
        msi.get_name_mapping("prot_id_name_map", msi.extra_drug_prot_data, "node_2", "node_2_name")
        
    if msi.extra_drug_dz_data:
        msi.get_name_mapping("drug_id_name_map", msi.extra_drug_dz_data, "drug", "drug_name")
        msi.get_name_mapping("dz_id_name_map", msi.extra_drug_dz_data, "indication", "indication_name")
    
    if msi.extra_dz_prot_data:
        msi.get_name_mapping("dz_id_name_map", msi.extra_dz_prot_data, "node_1", "node_1_name")
        msi.get_name_mapping("prot_id_name_map", msi.extra_dz_prot_data, "node_2", "node_2_name")
    
    if msi.extra_prot_prot_data:
        msi.get_name_mapping("prot_id_name_map", msi.extra_prot_prot_data, "node_1", "node_1_name")
        msi.get_name_mapping("prot_id_name_map", msi.extra_prot_prot_data, "node_2", "node_2_name")
    
    if msi.extra_prot_func_data:
        msi.get_name_mapping("prot_id_name_map", msi.extra_prot_func_data, "node_1", "node_1_name")
        msi.get_name_mapping("func_id_name_map", msi.extra_prot_func_data, "node_2", "node_2_name")
    
    if msi.extra_func_func_data:
        msi.get_name_mapping("func_id_name_map", msi.extra_func_func_data, "node_1", "node_1_name")
        msi.get_name_mapping("func_id_name_map", msi.extra_func_func_data, "node_2", "node_2_name")
    


    
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
    
    # Initialize the msi.
    msi = MSI(name, **kwargs)
    
    # Generate mappings between ids and gids and between ids and names.
    generate_mappings(msi)
    generate_name_mappings(msi)
    
    # Return the edge_index (all edges) and edge_indicies for individual edge types.
    edge_index, drug_prot_edges, drug_dz_edges, dz_prot_edges, prot_prot_edges, prot_func_edges, func_func_edges,  = (
            generate_edge_index(msi))
    
    # Obtain the total number of nodes of each type.
    total_drug_nodes = len(msi.mappings["drug_id_gid_map"].values())
    total_dz_nodes = len(msi.mappings["dz_id_gid_map"].values())
    total_prot_nodes = len(msi.mappings["prot_id_gid_map"].values())
    total_func_nodes = len(msi.mappings["func_id_gid_map"].values())
    
    # Get the total number of nodes and generate features for each node.
    num_nodes = (
            total_drug_nodes + total_dz_nodes + total_prot_nodes + total_func_nodes)
    features = torch.ones((num_nodes, msi.num_features))
    
    # Generate features for each node type.
    drug_features = torch.ones((total_drug_nodes, msi.num_features))
    dz_features = torch.ones((total_dz_nodes, msi.num_features))
    prot_features = torch.ones((total_prot_nodes, msi.num_features))
    func_features = torch.ones((total_func_nodes, msi.num_features))
  
    # Create a homogeneous graph and a heterogeneous graph.
    msi.load_graph(features, edge_index, num_nodes)
    msi.load_hetero_graph(
            drug_features, dz_features, prot_features, func_features,  
            drug_prot_edges, drug_dz_edges, dz_prot_edges, prot_prot_edges, prot_func_edges, 
            func_func_edges)
    
    # Return the msi object.
    return msi
