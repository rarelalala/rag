from ragas.metrics import faithfulness, answer_relevancy, context_relevancy, context_recall
from ragas.langchain.evalchain import RagasEvaluatorChain
from langsmith import Client
from langchain.smith import RunEvalConfig, run_on_dataset
from langsmith.utils import LangSmithError
import torch
import knowledgebase.split
from langchain.storage import InMemoryStore
from textsplitter.ChineseRecursiveTextSplitter import ChineseRecursiveTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever, SVMRetriever, TFIDFRetriever, KNNRetriever
from langchain.retrievers import EnsembleRetriever, ParentDocumentRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.document_transformers import LongContextReorder
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.chains import LLMChain, StuffDocumentsChain
from transformers.generation.utils import GenerationConfig
from langchain.chains.conversation.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory

import os
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="elder-rag-chat"

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
                                          trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

# model.generation_config = GenerationConfig.from_pretrained(model_path)

pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
)

llm = HuggingFacePipeline(pipeline=pipeline)
# memory = ConversationBufferWindowMemory(k=2)
# memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=40)

# 加载文件
loaders = knowledgebase.split.load_files_to_knowledge_base('../database/txt')
# loaders = knowledgebase.split.load_classified_files_to_knowledge_base('../database/txt', query)
# text_splitter = RecursiveCharacterTextSplitter(separators=['。'], chunk_size=512, chunk_overlap=32)
text_splitter = ChineseRecursiveTextSplitter(
    keep_separator=True,
    is_separator_regex=True,
    chunk_size=512,
    chunk_overlap=16
)

texts_chunks_list = []
documents = []
for loader in loaders:
    document = loader.load_and_split()
    documents.extend(document)
    texts_chunks = text_splitter.split_documents(document)
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

# faiss_vectorstore = FAISS.from_texts(texts_chunks_list, embeddings)
# faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})

vectorstore = Chroma.from_documents(texts_chunks_list, embeddings, persist_directory="db")
chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
# chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 2}, search_type="mmr")

vectorstore = Chroma(
    collection_name="split_parents",
    embedding_function=embeddings
)
parent_store = InMemoryStore()
parent_splitter = ChineseRecursiveTextSplitter(
    keep_separator=True,
    is_separator_regex=True,
    chunk_size=256,
    chunk_overlap=0
)
child_splitter = ChineseRecursiveTextSplitter(
    keep_separator=True,
    is_separator_regex=True,
    chunk_size=128,
    chunk_overlap=16
)
parent_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=parent_store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
parent_retriever.add_documents(documents)

# 存入向量库，创建retriever
ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, knn_retriever, chroma_retriever], weights=[0.5, 0.4, 0.1])
    
# 文档重排序
# docs = ensemble_retriever.get_relevant_documents(query)

# 相关性小的文档放在中间，相关性大的文档放在首尾两端
# reordering = LongContextReorder()
# reordered_docs = reordering.transform_documents(docs)

# 构造提示模板
document_prompt = PromptTemplate(
input_variables=["page_content"], template="{page_content}"
)
document_variable_name = "context"

template = """你是一名智能助手，请根据下面提供的信息更加全面且条理清晰地回答问题。

已知内容：
{context}

问题：
{question}

答案：
"""
prompt = PromptTemplate.from_template(template)
chain = (
        {"question": RunnablePassthrough(), "context": ensemble_retriever}
        | prompt
        | llm
        | StrOutputParser()
)

client = Client()
dataset_name = "elderly test"

# create evaluation chains
faithfulness_chain = RagasEvaluatorChain(metric=faithfulness)
answer_rel_chain = RagasEvaluatorChain(metric=answer_relevancy)
context_rel_chain = RagasEvaluatorChain(metric=context_relevancy)
context_recall_chain = RagasEvaluatorChain(metric=context_recall)

evaluation_config = RunEvalConfig(
    custom_evaluators=[
        faithfulness_chain,
        answer_rel_chain,
        context_rel_chain,
        context_recall_chain,
    ],
    prediction_key="result",
)

result = run_on_dataset(
    client,
    dataset_name,
    chain,
    evaluation=evaluation_config,
    input_mapper=lambda x: x,
)