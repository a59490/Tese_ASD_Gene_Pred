from grape import Graph
import pandas as pd

from grape import GraphVisualizer
from grape.embedders import FirstOrderLINEEnsmallen
from grape.embedders import SecondOrderLINEEnsmallen

from grape.embedders import Node2VecSkipGramEnsmallen
from grape.embedders import Node2VecCBOWEnsmallen

from grape.embedders import DeepWalkSkipGramEnsmallen
from grape.embedders import DeepWalkCBOWEnsmallen

from  grape.embedders import WalkletsCBOWEnsmallen
from grape.embedders import WalkletsSkipGramEnsmallen

from grape.embedders import HOPEEnsmallen
from grape.embedders import GLEEEnsmallen


csv_file_path = "string_protein_links.txt"

separator = " "  

# Create the graph from the CSV file
string_graph = Graph.from_csv(
    edge_path=csv_file_path,
    edge_list_separator=separator,
    sources_column_number=0,  # protein 1 is in the first column
    destinations_column_number=1,  # Aprotein 2 is in the second column
    weights_column_number=2,  # Assuming weight is in the third column ------------- /1000 check later!!!!!!!!
    directed=False,  #  graph is undirected, adjust if necessary
    verbose=True  # Optionally, set verbose to True for more information during loading
)

string_graph.remove_disconnected_nodes()

embedding_list = [("first_order_line", FirstOrderLINEEnsmallen(embedding_size=500)), ("node2vec_cbow", Node2VecCBOWEnsmallen(embedding_size=500)),
                      ("deepwalk_skipgram", DeepWalkSkipGramEnsmallen(embedding_size=500)),("second_order_line", SecondOrderLINEEnsmallen(embedding_size=500)),
                       ("deepwalk_cbow", DeepWalkCBOWEnsmallen(embedding_size=500)), ("hope", HOPEEnsmallen(embedding_size=500))]

def make_embedding(embedding_list, graph):
    for emb_name,algorithm in embedding_list:
        embedding = algorithm.fit_transform(graph)
        if len(embedding.get_all_node_embedding())==2:
            results_df=embedding.get_all_node_embedding()[0]
            results_df.to_csv(f"Embeddings/{emb_name}_embedding.csv")

        else:
            results_df=embedding.get_all_node_embedding()[0] # this is the central tokens, more useful for skipgram
            results_df.to_csv(f"Embeddings/{emb_name}_embedding.csv")

def clean_embedding(embedding_list):
    for emb_name,algorithm in embedding_list:
        results_df=pd.read_csv(f"Embeddings/{emb_name}_embedding.csv")
        converter=pd.read_csv("prot_converter.csv")
        new_emb = pd.merge(results_df, converter, left_on="Unnamed: 0", right_on='ensb_gene_id',how='inner')
        new_emb.to_csv(f"Clean_embeddings/{emb_name}_embedding.csv")
            

make_embedding(embedding_list,string_graph)
clean_embedding(embedding_list)

"""
embedding_list = [("first_order_line", FirstOrderLINEEnsmallen(embedding_size=200)), ("second_order_line", SecondOrderLINEEnsmallen(embedding_size=200)),
                   ("node2vec_skipgram", Node2VecSkipGramEnsmallen(embedding_size=200)), ("node2vec_cbow", Node2VecCBOWEnsmallen(embedding_size=200)),
                     ("glee", GLEEEnsmallen(embedding_size=200)), ("deepwalk_skipgram", DeepWalkSkipGramEnsmallen(embedding_size=200)),
                       ("deepwalk_cbow", DeepWalkCBOWEnsmallen(embedding_size=200)), ("hope", HOPEEnsmallen(embedding_size=200)),("walklets_cbow", WalkletsCBOWEnsmallen(embedding_size=200))]


"""