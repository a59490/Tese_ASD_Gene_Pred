import igraph as ig
from gprofiler import GProfiler
import leidenalg as la
import pandas as pd
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as plt
import networkx as nx
from netgraph import Graph
from netgraph import InteractiveGraph
import seaborn as sns




d_1=pd.read_csv('d_1.csv',header=None)
d_1.columns=["ensembl_gene_id"]
prot_conv=pd.read_csv('prot_converter.csv')

d_1_prot=pd.merge(d_1,prot_conv,left_on='ensembl_gene_id',right_on='ensb_gene_id',how='inner')
d_1_prot=d_1_prot[['ensembl_gene_id','ensb_prot_id','syb']]

ensb_prot_id_list=d_1_prot['ensb_prot_id'].tolist()

# Load the dataset from the text file
data = pd.read_csv("9606.protein.links.v12.0.txt.gz", sep=" ",compression="gzip")

print("data loaded")

data.columns=["protein1","protein2","weight"]

data["protein1"]=data["protein1"].str.split('.').str[1]
data["protein2"]=data["protein2"].str.split('.').str[1]
data["weight"]=data["weight"]/1000

data=data[data["protein1"].isin(ensb_prot_id_list) & data["protein2"].isin(ensb_prot_id_list)]

print("data filtered")

mygraph = ig.Graph.DataFrame(data, use_vids = False,directed=False)
partition_la = la.find_partition(mygraph, la.ModularityVertexPartition,weights=data["weight"])

print("partition and graph created")
partition_sizes = [len(partition) for partition in partition_la]

partition_list=[] 
for i in range(len(partition_sizes)):
    partition_list.append(set(partition_la[i]))

node_to_community_dict={}
for vertex,cluster_n in zip(mygraph.vs,partition_la.membership):
    node_to_community_dict[vertex["name"]]=cluster_n

edges = []
for prot1,prot2 in zip(data["protein1"].tolist(),data["protein2"].tolist()):
    edges.append((prot1,prot2))

print("edges and nodes created")

num_communities = len(partition_la)

community_to_color = sns.color_palette("Set1", num_communities).as_hex()

node_color = {node: community_to_color[node_to_community_dict[node]] for node in mygraph.vs["name"]}

print("colors assigned")

fig, ax = plt.subplots(figsize=(20, 20))
Graph(edges,
      node_color=node_color, 
      node_edge_width=0.15,     
      edge_width=0.04,        
      edge_alpha=0.25,        
      edge_layout='bundled', # this is where bundling is made possible,
      ax=ax,
      node_size=0.6,
)
print("graph created")


plt.savefig('Community_graph.png')
