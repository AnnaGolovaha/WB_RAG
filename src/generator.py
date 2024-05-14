from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain import HuggingFaceHub, HuggingFacePipeline, PromptTemplate

from langchain.chains import RetrievalQA


class Generator:
    def __init__(self, compression_retriever):
        self.generator_model_name = 'IlyaGusev/rugpt_large_turbo_instructed'
        self.model = AutoModelForCausalLM.from_pretrained(self.generator_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.generator_model_name)
        self.compression_retriever = compression_retriever
        self.prompt = """Ты - сотрудник техподдержки. Используй фрагменты контекста ниже, чтобы ответить на вопрос в конце. Не используй информацию, которой нет во фрагментах ниже.
        Если ты не знаешь ответ, просто скажи, что не знаешь. Используй одно предложение только один раз. Выбирай наиболее релевантные фрагменты по вопросу.

        Контекст:{context}

        Вопрос: {question}
        Ответ:
        """

        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"], template=self.prompt
        )
        self.hf_pipeline = self.init_hf_pipeline(
            model=self.model, tokenizer=self.tokenizer
        )
        self.chain = self.init_chain(
            hf_pipeline=self.hf_pipeline,
            compression_retriever=self.compression_retriever,
        )

    def init_hf_pipeline(self, model, tokenizer):
        llm = pipeline(
            task="text-generation",  # задание для модели
            model=model,  # сама модель
            tokenizer=tokenizer,
            max_new_tokens=500,  # Максимальное количество токенов для генерации
            repetition_penalty=1.6,  # штраф за повторение токенов, чтобы использовались более разнообразные слова при ответе
            model_kwargs={
                "load_in_8bit": False,  # возможность использовать квантованную модель
                "max_length": 1000,  # максимальная длина послед-ти
                "do_sample": True,  # это нужно для использования top_k и top_p
                "temperature": 0.8,  # креативность модели
                "top_k": 5,
                "top_p": 1,
            },
        )
        return HuggingFacePipeline(pipeline=llm)

    def init_chain(self, hf_pipeline, compression_retriever):

        qa_with_sources_chain_reranker = RetrievalQA.from_chain_type(
            llm=hf_pipeline,
            chain_type="stuff",
            retriever=compression_retriever,
            chain_type_kwargs={"prompt": self.prompt_template},
            return_source_documents=True,
        )
        return qa_with_sources_chain_reranker