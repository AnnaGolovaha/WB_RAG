from langchain_community.document_loaders import DataFrameLoader

from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from langchain_community.vectorstores import FAISS


class Retrieval:
    def __init__(self, knowledge_base, embedding_model_name):
        self.loader = DataFrameLoader(knowledge_base, page_content_column="chunk")
        self.documents = self.loader.load()

        self.embeddings_function = HuggingFaceEmbeddings(
            model_name=embedding_model_name
        )

    def faiss_func(self, faiss_top_k) -> None:
        faiss = FAISS.from_documents(self.documents, self.embeddings_function)
        return faiss.as_retriever(search_kwargs={"k": faiss_top_k})

    def BM25_func(self, bm25_top_k) -> None:
        bm25_retriever = BM25Retriever.from_documents(self.documents)
        bm25_retriever.k = bm25_top_k
        return bm25_retriever

    def ensemble(self, bm25_retriever, faiss_retriever):
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
        )
        return ensemble_retriever