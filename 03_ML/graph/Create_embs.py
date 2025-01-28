from grape import Graph
import pandas as pd
import os

from grape import GraphVisualizer
from grape.embedders import FirstOrderLINEEnsmallen
from grape.embedders import SecondOrderLINEEnsmallen

from grape.embedders import Node2VecSkipGramEnsmallen
from grape.embedders import Node2VecCBOWEnsmallen
from grape.embedders import Node2VecGloVeEnsmallen

from grape.embedders import DeepWalkSkipGramEnsmallen
from grape.embedders import DeepWalkCBOWEnsmallen
from grape.embedders import DeepWalkGloVeEnsmallen

from  grape.embedders import WalkletsCBOWEnsmallen
from grape.embedders import WalkletsSkipGramEnsmallen

from grape.embedders import HOPEEnsmallen
from grape.embedders import GLEEEnsmallen

from grape.embedders import DegreeSPINE
from grape.embedders import DegreeWINE

from grape.embedders import LaplacianEigenmapsEnsmallen

from grape.embedders import NodeLabelSPINE
from grape.embedders import NodeLabelWINE
from grape.embedders import RUBICONE
from grape.embedders import RUINE

from grape.embedders import ScoreSPINE
from grape.embedders import ScoreWINE

from grape.embedders import SocioDimEnsmallen
from grape.embedders import StructuredEmbeddingEnsmallen
from grape.embedders import TransEEnsmallen
from grape.embedders import UnstructuredEnsmallen

from grape.embedders import WalkletsCBOWEnsmallen
from grape.embedders import WalkletsGloVeEnsmallen
from grape.embedders import WalkletsSkipGramEnsmallen
from grape.embedders import WeightedSPINE


results_dir = "Results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

embeddings_dir = "Embeddings"
if not os.path.exists(embeddings_dir):
    os.makedirs(embeddings_dir)

clean_embeddings_dir = "Clean_embeddings"
if not os.path.exists(clean_embeddings_dir):
    os.makedirs(clean_embeddings_dir)


csv_file_path = "string_protein_links.txt"

separator = " "  

# Create the graph from the CSV file
string_graph = Graph.from_csv(
    edge_path=csv_file_path,
    edge_list_separator=separator,
    sources_column_number=0,  # protein 1 is in the first column
    destinations_column_number=1,  # Aprotein 2 is in the second column
    weights_column_number=2,  # Assuming weight is in the third column ------------- 
    directed=False,  #  graph is undirected, adjust if necessary
    verbose=True  # Optionally, set verbose to True for more information during loading
)

string_graph.remove_disconnected_nodes()



embedding_list=[("first_order_line", FirstOrderLINEEnsmallen(embedding_size=500)),("second_order_line", SecondOrderLINEEnsmallen(embedding_size=500)),
                ("deepwalk_skipgram", DeepWalkSkipGramEnsmallen(embedding_size=500)),("deepwalk_cbow", DeepWalkCBOWEnsmallen(embedding_size=500)),
                ("deepwalk_glove", DeepWalkGloVeEnsmallen(embedding_size=500)),("node2vec_skipgram", Node2VecSkipGramEnsmallen(embedding_size=500)), 
                ("node2vec_cbow", Node2VecCBOWEnsmallen(embedding_size=500)),("node2vec_glove", Node2VecGloVeEnsmallen(embedding_size=500)),
                ("degree_spine", DegreeSPINE(embedding_size=500)), ("degree_wine", DegreeWINE(embedding_size=500)),
                ("laplacian_eigenmaps", LaplacianEigenmapsEnsmallen(embedding_size=500)),("hope", HOPEEnsmallen(embedding_size=500)),
                ("glee", GLEEEnsmallen(embedding_size=500)),("node_label_spine", NodeLabelSPINE()),
                ("node_label_wine", NodeLabelWINE()),("rubicone", RUBICONE(embedding_size=500)),
                ("ruine", RUINE(embedding_size=500)),("score_spine", ScoreSPINE(embedding_size=500)),
                ("score_wine", ScoreWINE(embedding_size=500)),
                ("socio_dim", SocioDimEnsmallen(embedding_size=500)),("structured_embedding", StructuredEmbeddingEnsmallen(embedding_size=500)),
                ("transe", TransEEnsmallen(embedding_size=500)),("unstructured", UnstructuredEnsmallen(embedding_size=500)),
                ("walklets_cbow", WalkletsCBOWEnsmallen(embedding_size=500)),("walklets_glove", WalkletsGloVeEnsmallen(embedding_size=500)),
                ("walklets_skipgram", WalkletsSkipGramEnsmallen(embedding_size=500)),("weighted_spine", WeightedSPINE(embedding_size=500))
                ]



def make_embedding(embedding_list, graph):
    for emb_name,algorithm in embedding_list:
        try:
            embedding = algorithm.fit_transform(graph)
            if len(embedding.get_all_node_embedding())==2:
                results_df=embedding.get_all_node_embedding()[0]
                results_df.to_csv(f"Embeddings/{emb_name}_embedding.csv")

            else:
                results_df=embedding.get_all_node_embedding()[0] # this is the central tokens, more useful for skipgram
                results_df.to_csv(f"Embeddings/{emb_name}_embedding.csv")
        except:
            print(f"Error with {emb_name}")

def clean_embedding(embedding_list):
    for emb_name,algorithm in embedding_list:
        try:
            results_df=pd.read_csv(f"Embeddings/{emb_name}_embedding.csv")
            converter=pd.read_csv("prot_converter.csv")
            new_emb = pd.merge(results_df, converter, left_on="Unnamed: 0", right_on='ensb_gene_id',how='inner')
            new_emb.to_csv(f"Clean_embeddings/{emb_name}_embedding.csv")
        except:
            print(f"Error with {emb_name}")

make_embedding(embedding_list,string_graph)
clean_embedding(embedding_list)
