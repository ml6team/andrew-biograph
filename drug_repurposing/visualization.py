import pandas as pd
import networkx as nx
from pyvis import network as net
from torch_geometric.utils import to_networkx, degree
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
from rdkit import Chem
from rdkit.Chem import Descriptors
import mols2grid
import py3Dmol

def generate_nx_graph(data):
    # Create a directed graph from the data object, then convert it to undirected.
    G = to_networkx(data)  # Do not use the parameter to_undriected. It does not work properly.
    G_undirected = G.to_undirected()
    return G, G_undirected


def generate_khop_subgraph(msi, G, gid, k):
    # Get the neighbors of the drug amphetamine (id=DB00182, name="Amphetamine", gid=17713)
    node_label_dict = {}
    node_type_dict = {}
    node_size_dict = {}
    subgraph_nodes = nx.single_source_shortest_path_length(G, gid, cutoff=k)
    print(f"Total subgraph nodes: {len(subgraph_nodes)}")
    
    # Get the maximum gids of each type of node.
    max_prot_gid = max(msi.mappings["prot_id_gid_map"].values())
    max_drug_gid = max(msi.mappings["drug_id_gid_map"].values())
    max_dz_gid = max(msi.mappings["dz_id_gid_map"].values())
    max_func_gid = max(msi.mappings["func_id_gid_map"].values())
    
    # Get the id_map, name_map, type, and size for each node in the subgraph.
    for node in subgraph_nodes:
        if node <= max_prot_gid:
            id_mapping = msi.mappings["prot_id_gid_map"].copy()
            name_mapping = msi.mappings["prot_id_name_map"].copy()  
            node_type = "protein"
            node_size = "10"
        elif node > max_prot_gid and node <= max_drug_gid:
            id_mapping =  msi.mappings["drug_id_gid_map"].copy()
            name_mapping = msi.mappings["drug_id_name_map"].copy() 
            node_type = "drug"
            node_size = "25"
        elif node > max_drug_gid and node <= max_dz_gid:
            id_mapping = msi.mappings["dz_id_gid_map"].copy()
            name_mapping = msi.mappings["dz_id_name_map"].copy() 
            node_type = "disease"
            node_size = "35"
        else:
            id_mapping = msi.mappings["func_id_gid_map"].copy()
            name_mapping = msi.mappings["func_id_name_map"].copy()
            node_type = "function"
            node_size = "20"
        
        # Assign the node a label, type, and size.
        for id, gid in id_mapping.items():
            if node == gid:
                node_label_dict[gid] = name_mapping[id]
                node_type_dict[gid] = node_type
                node_size_dict[gid] = node_size
    
    # Generate the networkx subraph and assign node attributes.
    subG = G.subgraph(list(subgraph_nodes.keys()))
    for i, node in enumerate(subG.nodes):
        subG.nodes[node]["label"] = node_label_dict[node]
        subG.nodes[node]["group"] = node_type_dict[node]
        subG.nodes[node]["size"] = node_size_dict[node]

    return subG


def generate_pyvis_graph(G):
    pyvis_graph = net.Network(notebook=True)
    pyvis_graph.from_nx(G)
    return pyvis_graph



def draw_molecule(mol_smiles):
    
    mol = Chem.MolFromSmiles(mol_smiles)
    mol = Chem.MolToMolBlock(mol, confId=-1)
    view = py3Dmol.view(width=400,height=400)
    view.removeAllModels()
    
    view.addModel(mol,'sdf')
    view.setStyle({'stick':{}})
    view.setBackgroundColor('0xeeeeee')
    view.zoomTo()
    
    return view.show()



def interactive_mol_grid(
        mol_dict,
        n_cols=3,
        size=(210,130),
        gap=0,
        selection=True,
        border="2px ridge black",
        fontfamily="Times New Roman"):
    
    """ Create interactive molecule grid using mol2grid library.
    
    Parameters
    ----------
    mol_dict : dict
        Dictionary requiring 'name' key and list of molecule names as value and
        'smiles' key with list of smiles string as value.
    n_cols : int 
        Set the number of columns for the viewer (default = 3).
    size : tuple 
        Set the size of each molecule (default = (210,13))
    gap : int
        Set the spacing between the molecules in the grid (default=0).
    selection : bool
        Turn the selection box on or off (default = True).
    border : str
        Set the css border style (default = 2px ridge black).
    fontfamily : str
        Set the css-style font (default = Times New Roman).
    
    Returns
    -------
    viewer : IPython.core.display.HTML
        Interactive HTML object displaying molecules and accompanying information in 
        a grid.
    df : DataFrame
        Dataframe with chemical information for each molecule.
    """
    
    # Load the dictionary into a dataframe.
    df = pd.DataFrame.from_dict(mol_dict)
    
    # Create mol objects from smiles strings.
    df["mol"] = df["SMILES"].apply(Chem.MolFromSmiles)
    
    # Compute extra molecular properties.
    df["MolWt"] = df["mol"].apply(Descriptors.ExactMolWt)
    df["QED"] = df["mol"].apply(Descriptors.qed)
    df["LogP"] = df["mol"].apply(Descriptors.MolLogP)
    df["NumHDonors"] = df["mol"].apply(Descriptors.NumHDonors)
    df["NumHAcceptors"] = df["mol"].apply(Descriptors.NumHAcceptors)
    
    # Reformat the dataframe.
    df.drop(columns=["mol"], inplace=True)
    
    # Create molecule viewer.
    viewer = mols2grid.display(
            df,
            n_cols=3,
            subset=["name", "img"],  # set the fields  displayed on the grid.
            tooltip=["name", "MolWt", "QED"], # set the fields displayed on mouse hover.
            template="pages",
            size=size, # size of each image.
            gap=gap, # spacing between molecules.
            selection=selection, # toggle selection box.
            border=border, # css border style.
            fontfamily=fontfamily) # css font style.
    
    return df, viewer


    
    






def visualize_tsne_embeddings(model, data, title, perplexity=30.0,
                              labeled=False, labels=[]):
    """Visualizes node embeddings in 2D space with t-SNE.
    
    Parameters
    ----------
    model : trained or untrained model
        A trained or untrained model.
    data : Data object
        A data object
    title : str
        Title of the plot
    perplexity : float
        t-SNE hyperparameter for perplexity
        """
    model.eval()
    x = data.x
    z = model.encode(x, data.edge_index)
    ax1, ax2 = zip(*TSNE(n_components=2, learning_rate='auto', perplexity=perplexity,
                       init='random').fit_transform(z.detach().cpu().numpy()))

    fig = px.scatter(x=ax1, y=ax2, color=['r']*17660 + ['g']*1661 + ['b']*840 + ['p']*9798,
                   title=title)

    if labeled:
        for i in labels:
            fig.add_annotation(x=ax1[i], y=ax2[i],
                         text=str(i), showarrow=False)
    fig.show()
    
    

def visualize_pca_embeddings(model, data, title, labeled=False, labels=[]):
    """Visualizes node embeddings in 2D space with PCA (components=2).
    
    Parameters
    ----------
    model : trained or untrained model.
    data : Data object
        A data object.
    title : str
        Title of the plot.
        """
    
    model.eval()
    x = data.x
    z = model.encode(x, data.edge_index)
    pca = PCA(n_components=2)
    components = pca.fit_transform(z.detach().cpu().numpy())
    fig = px.scatter(components, x=0, y=1, color=['r']*17660 + ['g']*1661 + ['b']*840 + ['p']*9798,
                   title=title)
    if labeled:
        for i in labels:
            fig.add_annotation(x=components[:,0][i], y=components[:,1][i],
                         text=str(i), showarrow=False)
    fig.show()