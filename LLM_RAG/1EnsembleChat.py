import torch
import knowledgebase.split
from textsplitter.ChineseRecursiveTextSplitter import ChineseRecursiveTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever, SVMRetriever, TFIDFRetriever, KNNRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.generation.utils import GenerationConfig


# 配置项
model_path = "../models/Baichuan2-7B-Chat"
# model_path = "../models/Qwen-7B-Chat"
embed_path = "../models/bge-large-zh-v1.5"

# 加载embedding模型
embeddings = HuggingFaceEmbeddings(
    model_name=embed_path,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)
# 加载LLM
tokenizer = AutoTokenizer.from_pretrained(model_path,
                                          device_map="auto",
                                          use_fast=False,
                                          trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

model.generation_config = GenerationConfig.from_pretrained(model_path)

pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
)

llm = HuggingFacePipeline(pipeline=pipeline)

def ensemblechat(query):
    # 加载文件
    loaders = knowledgebase.split.load_files_to_knowledge_base('../database/txt')
    # loaders = knowledgebase.split.load_classified_files_to_knowledge_base('../database/txt', query)
    # text_splitter = RecursiveCharacterTextSplitter(separators=['。'], chunk_size=512, chunk_overlap=32)
    text_splitter = ChineseRecursiveTextSplitter(
        keep_separator=True,
        is_separator_regex=True,
        chunk_size=512,
        chunk_overlap=32
    )
    texts_chunks_list = []
    for loader in loaders:
        documents = loader.load_and_split()
        texts_chunks = text_splitter.split_documents(documents)
        texts_chunks_list.extend(texts_chunks)
    texts_list = [text.page_content for text in texts_chunks_list]

    # 构造emsemble retriever
    bm25_retriever = BM25Retriever.from_texts(texts_list)
    bm25_retriever.k = 2

    knn_retriever = KNNRetriever.from_texts(texts_list,embeddings)
    knn_retriever.k = 2

    # tfidf_retriever = TFIDFRetriever.from_texts(texts_list)
    # tfidf_retriever.k = 2

    # svm_retriever = SVMRetriever.from_texts(texts_list,embeddings)
    # svm_retriever.k = 2

    vectorstore = Chroma.from_documents(texts_chunks_list, embeddings, persist_directory="db")
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    # ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5])
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, knn_retriever, chroma_retriever], weights=[0.5, 0.3, 0.2])


    # 提示模板
    template = """你是一名智能助手，请根据下面提供的信息更加全面且条理清晰地回答问题。

    已知内容：
    {context}

    问题：
    {question}
    """
    prompt = PromptTemplate.from_template(template)
    chain = (
            {"question": RunnablePassthrough(), "context": ensemble_retriever}
            | prompt
            | llm
            | StrOutputParser()
    )
    # 查询验证
    # query = "老年人经常进行体力活动，是否会减缓衰老或降低患病概率？"
    result = chain.invoke(query)
    print('-------------------------------')
    print(result)
    return result

if __name__ == "__main__":
    query = "高血压患者哪些食物要少吃？"
    result = ensemblechat(query)