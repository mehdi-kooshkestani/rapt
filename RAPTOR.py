from langchain_openai import OpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
import os
# from transformers import AutoModel
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import shutil
import numpy as np
from sentence_transformers import SentenceTransformer
# from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from cleantext import clean
# import matplotlib.pyplot as plt
# import tiktoken
from langchain_community.document_loaders import Docx2txtLoader
from hazm import Normalizer
import re
from langchain_experimental.text_splitter import SemanticChunker
import subprocess
import os
import shutil
import locale #mports Python's locale module, which handles localization (e.g., encoding, number formatting)
import pandas as pd
from sklearn.mixture import GaussianMixture
from typing import Optional
# import numpy as np
import umap
from langchain_community.retrievers.bm25 import BM25Retriever
from typing import Dict, List, Optional, Tuple

RANDOM_SEED = 224  # Fixed seed for reproducibility
OPENAI_API_KEY= None

class JinaV3Embedding:
    def __init__(self):
        self.model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)

    def embed_documents(self, texts):
        return self.model.encode(
            texts,
            task="retrieval.passage",
            prompt_name="retrieval.passage",
            convert_to_numpy=True
        ).tolist()

    def embed_query(self, text):
        return self.model.encode(
            [text],
            task="retrieval.query",
            prompt_name="retrieval.query",
            convert_to_numpy=True
        )[0].tolist()
 

embd=JinaV3Embedding()


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


def cleaning(text):
    text = text.strip()
    text = clean(
        text,
        fix_unicode=True,
        to_ascii=False,
        lower=True,
        no_line_breaks=True,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_numbers=False,
        no_digits=False,
        no_currency_symbols=True,
        no_punct=False,
        replace_with_url="",
        replace_with_email="",
        replace_with_phone_number="",
        replace_with_number="",
        replace_with_digit="0",
        replace_with_currency_symbol="",
    )
    text = cleanhtml(text)
    normalizer = Normalizer()
    text = normalizer.normalize(text)

    # remove emojis, symbols, etc.
    wierd_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u200d"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\u3030"
        u"\ufe0f"
        u"\u2069"
        u"\u2066"
        u"\u2068"
        u"\u2067"
        "]+", flags=re.UNICODE)
    text = wierd_pattern.sub(r'', text)
    text = re.sub("#", "", text)
    text = re.sub("\s+", " ", text)
    return text


def load_model(model_name="gpt-3.5-turbo-instruct", max_tokens=500):
    # load_dotenv()
    # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    llm = OpenAI(
        model=model_name,
        api_key=OPENAI_API_KEY,
        max_tokens=max_tokens,
        temperature=0.7
    )
    return llm


def has_nvidia_gpu():
    """Returns True if NVIDIA GPU is available and nvidia-smi exists."""
    return shutil.which("nvidia-smi") is not None


def getpreferredencoding(do_setlocale = True):
  return "UTF-8"


def global_cluster_embeddings(embeddings: np.ndarray,dim: int,n_neighbors: Optional[int] = None,metric: str = "cosine") -> np.ndarray:
    """
    Perform global dimensionality reduction on the embeddings using UMAP.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - dim: The target dimensionality for the reduced space.
    - n_neighbors: Optional; the number of neighbors to consider for each point.
                   If not provided, it defaults to the square root of the number of embeddings.
    - metric: The distance metric to use for UMAP.

    Returns:
    - A numpy array of the embeddings reduced to the specified dimensionality.
    """
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    return umap.UMAP(n_neighbors=n_neighbors, n_components=dim, metric=metric).fit_transform(embeddings)


def local_cluster_embeddings(embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine") -> np.ndarray:
    """
    Perform local dimensionality reduction on the embeddings using UMAP, typically after global clustering.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - dim: The target dimensionality for the reduced space.
    - num_neighbors: The number of neighbors to consider for each point.
    - metric: The distance metric to use for UMAP.

    Returns:
    - A numpy array of the embeddings reduced to the specified dimensionality.
    """
    return umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    """
    Determine the optimal number of clusters using the Bayesian Information Criterion (BIC) with a Gaussian Mixture Model.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - max_clusters: The maximum number of clusters to consider.
    - random_state: Seed for reproducibility.

    Returns:
    - An integer representing the optimal number of clusters found.
    """
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    return n_clusters[np.argmin(bics)]


def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    """
    Cluster embeddings using a Gaussian Mixture Model (GMM) based on a probability threshold.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - threshold: The probability threshold for assigning an embedding to a cluster.
    - random_state: Seed for reproducibility.

    Returns:
    - A tuple containing the cluster labels and the number of clusters determined.
    """
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def perform_clustering(
    embeddings: np.ndarray,
    dim: int,
    threshold: float,
) -> List[np.ndarray]:
    """
    Perform clustering on the embeddings by first reducing their dimensionality globally, then clustering
    using a Gaussian Mixture Model, and finally performing local clustering within each global cluster.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - dim: The target dimensionality for UMAP reduction.
    - threshold: The probability threshold for assigning an embedding to a cluster in GMM.

    Returns:
    - A list of numpy arrays, where each array contains the cluster IDs for each embedding.
    """
    if len(embeddings) <= dim + 1:
        # Avoid clustering when there's insufficient data
        return [np.array([0]) for _ in range(len(embeddings))]

    # Global dimensionality reduction
    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)
    # Global clustering
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold
    )

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    # Iterate through each global cluster to perform local clustering
    for i in range(n_global_clusters):
        # Extract embeddings belonging to the current global cluster
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]

        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            # Handle small clusters with direct assignment
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            # Local dimensionality reduction and clustering
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold
            )

        # Assign local cluster IDs, adjusting for total clusters already processed
        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    return all_local_clusters


def embed(texts):
    """
    Generate embeddings for a list of text documents.

    This function assumes the existence of an `embd` object with a method `embed_documents`
    that takes a list of texts and returns their embeddings.

    Parameters:
    - texts: List[str], a list of text documents to be embedded.

    Returns:
    - numpy.ndarray: An array of embeddings for the given text documents.
    """
    text_embeddings = embd.embed_documents(texts)
    text_embeddings_np = np.array(text_embeddings)
    return text_embeddings_np


def embed_cluster_texts(texts):
    """
    Embeds a list of texts and clusters them, returning a DataFrame with texts, their embeddings, and cluster labels.

    This function combines embedding generation and clustering into a single step. It assumes the existence
    of a previously defined `perform_clustering` function that performs clustering on the embeddings.

    Parameters:
    - texts: List[str], a list of text documents to be processed.

    Returns:
    - pandas.DataFrame: A DataFrame containing the original texts, their embeddings, and the assigned cluster labels.
    """
    text_embeddings_np = embed(texts)  # Generate embeddings
    cluster_labels = perform_clustering(
        text_embeddings_np, 10, 0.1
    )  # Perform clustering on the embeddings
    df = pd.DataFrame()  # Initialize a DataFrame to store the results
    df["text"] = texts  # Store original texts
    df["embd"] = list(text_embeddings_np)  # Store embeddings as a list in the DataFrame
    df["cluster"] = cluster_labels  # Store cluster labels
    return df


def fmt_txt(df: pd.DataFrame) -> str:
    """
    Formats the text documents in a DataFrame into a single string.

    Parameters:
    - df: DataFrame containing the 'text' column with text documents to format.

    Returns:
    - A single string where all text documents are joined by a specific delimiter.
    """
    unique_txt = df["text"].tolist()
    return "--- --- \n --- --- ".join(unique_txt)


def embed_cluster_summarize_texts(
    texts: List[str], level: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Embeds, clusters, and summarizes a list of texts. This function first generates embeddings for the texts,
    clusters them based on similarity, expands the cluster assignments for easier processing, and then summarizes
    the content within each cluster.

    Parameters:
    - texts: A list of text documents to be processed.
    - level: An integer parameter that could define the depth or detail of processing.

    Returns:
    - Tuple containing two DataFrames:
      1. The first DataFrame (`df_clusters`) includes the original texts, their embeddings, and cluster assignments.
      2. The second DataFrame (`df_summary`) contains summaries for each cluster, the specified level of detail,
         and the cluster identifiers.
    """

    # Embed and cluster the texts, resulting in a DataFrame with 'text', 'embd', and 'cluster' columns
    df_clusters = embed_cluster_texts(texts)

    # Prepare to expand the DataFrame for easier manipulation of clusters
    expanded_list = []

    # Expand DataFrame entries to document-cluster pairings for straightforward processing
    for index, row in df_clusters.iterrows():
        for cluster in row["cluster"]:
            expanded_list.append(
                {"text": row["text"], "embd": row["embd"], "cluster": cluster}
            )

    # Create a new DataFrame from the expanded list
    expanded_df = pd.DataFrame(expanded_list)

    # Retrieve unique cluster identifiers for processing
    all_clusters = expanded_df["cluster"].unique()

    print(f"--Generated {len(all_clusters)} clusters--")
    model=load_model(model_name="gpt-3.5-turbo-instruct", max_tokens=500)
    # Summarization
    template = """متن زیر را خلاصه کن:\n\n
    {context}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model | StrOutputParser()

    # Format text within each cluster for summarization
    summaries = []
    for i in all_clusters:
        df_cluster = expanded_df[expanded_df["cluster"] == i]
        formatted_txt = fmt_txt(df_cluster)
        summaries.append(chain.invoke({"context": formatted_txt}))

    # Create a DataFrame to store summaries with their corresponding cluster and level
    df_summary = pd.DataFrame(
        {
            "summaries": summaries,
            "level": [level] * len(summaries),
            "cluster": list(all_clusters),
        }
    )

    return df_clusters, df_summary


def recursive_embed_cluster_summarize(
    texts: List[str], level: int = 1, n_levels: int = 3
    ) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Recursively embeds, clusters, and summarizes texts up to a specified level or until
    the number of unique clusters becomes 1, storing the results at each level.

    Parameters:
    - texts: List[str], texts to be processed.
    - level: int, current recursion level (starts at 1).
    - n_levels: int, maximum depth of recursion.

    Returns:
    - Dict[int, Tuple[pd.DataFrame, pd.DataFrame]], a dictionary where keys are the recursion
      levels and values are tuples containing the clusters DataFrame and summaries DataFrame at that level.
    """
    results = {}  # Dictionary to store results at each level

    # Perform embedding, clustering, and summarization for the current level
    df_clusters, df_summary = embed_cluster_summarize_texts(texts, level)

    # Store the results of the current level
    results[level] = (df_clusters, df_summary)

    # Determine if further recursion is possible and meaningful
    unique_clusters = df_summary["cluster"].nunique()
    if level < n_levels and unique_clusters > 1:
        # Use summaries as the input texts for the next level of recursion
        new_texts = df_summary["summaries"].tolist()
        next_level_results = recursive_embed_cluster_summarize(
            new_texts, level + 1, n_levels
        )

        # Merge the results from the next level into the current results dictionary
        results.update(next_level_results)

    return results


def query_chunks(query: str, retriever_25, retriever_sum):  # retriever_sum returnes 7 chunks and retriever_25 returns 4 chunks
    results_25 = retriever_25.get_relevant_documents(query)
    results_25_list=[doc.page_content for doc in results_25]
    results_sum = retriever_sum.invoke(query)
    results_sum_list = [doc.page_content for doc in results_sum]
    # results_25_list.extend(results_sum_list)
    for i in range(3):
        index = 2 + 3 * i
        results_sum_list.insert(index, results_25_list[i])
    results_sum_list[10] = results_25_list[3]   
    return results_sum_list


def load_file_semantic_chunking_and_embedding_saving(file_path):
    """
    Args:
        file_path (str): The path to the file to be processed. The file is expected to be in `.docx` format.
    returns:

    """
    loader = Docx2txtLoader(file_path)
    documents = loader.load()
    # View content
    # for i, doc in enumerate(documents):
    #     print(f"Document {i+1}:\n {doc.page_content}")
    # print(f"Total number of documents: {len(documents)}") 
    file_text=documents[0].page_content
    paragraphs = [para.strip() for para in file_text.split('\n\n') if para.strip()]
    # print(f"Number of paragraphs: {len(paragraphs)}")

    paragraphs = [cleaning(para) for para in paragraphs]
    paragraphs=[para.replace("\u200c", " ") for para in paragraphs]
    paragraphs=[para.replace("\u202b", "") for para in paragraphs] # is a control character used to change text direction in bidirectional text
    paragraphs=[para.replace("\u202a", " ") for para in paragraphs]

    # for i, para in enumerate(paragraphs):
    #     print(f"Paragraph {i+1}")
    #     print(para)

    
    cleaned_docs = [
        Document(page_content=para) for para in paragraphs
    ]

    embedding = JinaV3Embedding()

    text_splitter = SemanticChunker(embedding, breakpoint_threshold_type="gradient", breakpoint_threshold_amount=50)
    chunks = text_splitter.split_documents(cleaned_docs)

    # print(f"\n🧩 Total Chunks Created: {len(chunks)}")

    #print all chunks
    # for i, chunk in enumerate(chunks):
    #     print(f"Chunk {i+1}:\n{chunk.page_content}\n")
    
    global_embeddings=[embedding.embed_query(ch.page_content) for ch in chunks]
    # print(np.array(global_embeddings).shape)  
    
    if has_nvidia_gpu():
        # print("✅ NVIDIA GPU detected. Installing llama-cpp-python with CUDA support...")
        cuda_availablity=" NVIDIA GPU detected. Installing llama-cpp-python with CUDA support..."
        # Set environment variables
        os.environ["CMAKE_ARGS"] = "-DGGML_CUDA=on"
        os.environ["FORCE_CMAKE"] = "1"

        # Install the package with GPU support
        subprocess.run(["pip", "install", "-qU", "llama-cpp-python"], check=True)

    else:
        # print("⚠️ No NVIDIA GPU detected. Skipping CUDA install or using CPU fallback.")
        cuda_availablity= "⚠️ No NVIDIA GPU detected. Skipping CUDA install or using CPU fallback."
        # Optionally install CPU version:
        subprocess.run(["pip", "install", "-qU", "llama-cpp-python"], check=True)

    locale.getpreferredencoding = getpreferredencoding
    chunks_texts=[ch.page_content for ch in chunks]
    dim = 2
    # global_embeddings_reduced = reduce_cluster_embeddings(global_embeddings, dim)
    
    # embd=JinaV3Embedding()
    # embd= embedding

    leaf_texts = chunks_texts
    results = recursive_embed_cluster_summarize(leaf_texts, level=1, n_levels=3)    #results is that complex dictionary
    

    """gen=results[1]
    print(type(gen))
    print(f"length:{len(gen)}")
    print(gen)
    print("-----------------------------------")
    dat=gen[1]
    print(type(dat))
    print(f"length:{len(dat)}")
    print(dat)
    for index,row in dat.iterrows():
        print(row["summaries"])

    gen=results[2]
    print(type(gen))
    print(gen)
    print("-----------------------------------")
    dat=gen[1]
    print(type(dat))
    print(dat)
    for index,row in dat.iterrows():
        print(row["summaries"])
    
    gen=results[3]
    print(type(gen))
    print(gen)
    print("-----------------------------------")
    dat=gen[1]
    print(type(dat))
    print(dat)
    for index,row in dat.iterrows():
        print(row["summaries"])"""

    # Initialize all_texts with leaf_texts
    all_texts = leaf_texts.copy()

    # Iterate through the results to extract summaries from each level and add them to all_texts
    for level in sorted(results.keys()):
        # Extract summaries from the current level's DataFrame
        summaries = results[level][1]["summaries"].tolist()
        # Extend all_texts with the summaries from the current level
        all_texts.extend(summaries)
        
    #Final Summaries extracted
    all_chunks_number=len(all_texts)
    # print(f"number of total chunks consisting main and summaries:{len(all_texts)}") #195+38+8+1
    # print("--------------------------------------")
    # for i,at in enumerate(all_texts):
    #     print(f"{i}th chunk:\n{at}")
    
    CHROMA_PATH = "./chroma_final"
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    vector_sum = Chroma.from_texts(
        texts=all_texts,
        embedding=embd,
        persist_directory=CHROMA_PATH
    )
    vector_sum.persist()

    retriever_sum = vector_sum.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 7}
    )
    
    all_texts_docs = [Document(page_content=text) for text in all_texts]
    retriever_25 = BM25Retriever.from_documents(all_texts_docs)

    return(retriever_25, retriever_sum, cuda_availablity, paragraphs, chunks_texts, all_texts, all_chunks_number)
    


# r_25, r_sum, c_availablity, pragphs, chks_ts, all_ts, all_cks_nuer = load_file_semantic_chunking_and_embedding_saving("final_text.docx")

# print(f"cuda availability: {c_availablity}")
# print(f"number of paragraphs: {len(pragphs)}")
# print(f"number of chunks: {len(chks_ts)}")
# print(f"number of all chunks consisting main and summaries: {all_cks_nuer}")
# print("all chunks texts:")
# for i, at in enumerate(all_ts):
#     print(f"{i}th chunk:\n{at}")
# print("------------------------------------------------------")



# query="وضعیت مذهب تشیع در زمان صفویه چگونه بود؟"
# for i,ch in enumerate(query_chunks(query,r_25, r_sum)):
#     print(f"chunk:{i}\n{ch}\n")
# print("------------------------------------------------------")
# query="وضعیت کشور در دوران قاجار چگونه بود؟"
# for i,ch in enumerate(query_chunks(query,r_25, r_sum)):
#     print(f"chunk:{i}\n{ch}\n")
# print("------------------------------------------------------")   
# query="تاثیر چای سبز بر روی لاغری چیست؟"
# for i,ch in enumerate(query_chunks(query,r_25, r_sum)):
#     print(f"chunk:{i}\n{ch}\n")


# text="سلام بر تو ای "
# normalizer = Normalizer()
# text = normalizer.normalize(text)
# print(text)