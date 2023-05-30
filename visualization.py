import networkx as nx
from pyvis import network as net
from torch_geometric.utils import to_networkx, degree

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

