from pathlib import Path
import pandas as pd
import re

from pydantic import BaseSettings
import yaml
import emoji


class DataPreprocessor:
    def __init__(self, knowledge_base_path: Path, qa_pairs_path: Path):
        self.knowledge_base = pd.read_excel(knowledge_base_path)
        self.qa_pairs = pd.read_excel(qa_pairs_path)

    def clean_text(self, text: str) -> str:
        cleaned_text = text.replace("\\n", " ")
        cleaned_text = cleaned_text.replace("\n", " ")
        cleaned_text = cleaned_text.replace("__________", ",")
        cleaned_text = re.sub(r"<[^>]+>", "", cleaned_text)
        cleaned_text = emoji.replace_emoji(cleaned_text, replace="")
        return cleaned_text

    def clean_text_column(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        df[column_name] = df[column_name].apply(self.clean_text)
        return df

    def filter(self) -> None:
        self.knowledge_base.drop_duplicates(
            subset=["chunk", "document_id"], inplace=True
        )
        assert (
                self.knowledge_base.duplicated(subset=["chunk", "document_id"]).sum() == 0
        )
        self.knowledge_base["part_id"] = self.knowledge_base["part_id"].fillna(0).astype(int)
        self.knowledge_base = self.clean_text_column(
            df=self.knowledge_base, column_name="chunk"
        )
        self.qa_pairs.drop_duplicates(subset=["question", "answer"], inplace=True)
        assert self.qa_pairs.duplicated(subset=["question", "answer"]).sum() == 0
        self.qa_pairs = self.clean_text_column(df=self.qa_pairs, column_name="question")
        self.qa_pairs = self.clean_text_column(df=self.qa_pairs, column_name="answer")
