from pathlib import Path
import yaml
from pydantic import BaseSettings

from src.preprocessing import DataPreprocessor
from src.retrieval import Retrieval
from src.reranker import Reranker
from src.generator import Generator


class Params(BaseSettings):
    knowledge_base_path: Path
    qa_pairs_path: Path
    faiss_top_k: int
    bm25_top_k: int
    embedding_model_name: str
    reranker_model_name: str
    compressor_top_n: int
    # generator_model_name: str


def process_query(query: str) -> str:
    with Path.cwd().joinpath("params.yaml").open() as p:
        yaml_obj = yaml.safe_load(p)
        params = Params.parse_obj(yaml_obj)

    database = DataPreprocessor(
        knowledge_base_path=params.knowledge_base_path,
        qa_pairs_path=params.qa_pairs_path,
    )
    database.filter()

    retrieval = Retrieval(
        knowledge_base=database.knowledge_base,
        embedding_model_name=params.embedding_model_name,
    )
    ensemble_retriever = retrieval.ensemble(
        retrieval.BM25_func(params.bm25_top_k), retrieval.faiss_func(params.faiss_top_k)
    )

    reranker = Reranker(
        ensemble_retriever=ensemble_retriever,
        reranker_model_name=params.reranker_model_name,
        compressor_top_n=params.compressor_top_n,
    )

    generator = Generator(
        # generator_model_name=params.generator_model_name,
        compression_retriever=reranker.compression_retriever
    )

    result = generator.chain.invoke({"query": query})
    answer = result["result"].split("Ответ:")[1].strip()
    return answer


if __name__ == "__main__":
    query = "query: Какой рейтинг при открытии ПВЗ?"
    result = process_query(query)
    print(result)
