from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


class Reranker:
    def __init__(self, reranker_model_name, ensemble_retriever, compressor_top_n):
        self.model_reranker = HuggingFaceCrossEncoder(model_name=reranker_model_name)

        self.compressor = CrossEncoderReranker(
            model=self.model_reranker, top_n=compressor_top_n
        )
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor, base_retriever=ensemble_retriever
        )