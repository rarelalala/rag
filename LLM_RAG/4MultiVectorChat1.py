import uuid
import torch
import knowledgebase.split
from dataset.mydataset import MyDataset
from torch.utils.data import DataLoader
from langchain_core.outputs.chat_generation import ChatGeneration
from langchain_core.messages.base import BaseMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from textsplitter.ChineseRecursiveTextSplitter import ChineseRecursiveTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain


def full_documents_func(docs):
    # 方法1 分割文档，生成更小的组块
    child_text_splitter = ChineseRecursiveTextSplitter(
        keep_separator=True,
        is_separator_regex=True,
        chunk_size=512,
        chunk_overlap=32
    )
    sub_docs = []
    for i, doc in enumerate(docs):
        _id = doc_ids[i]
        _sub_docs = child_text_splitter.split_documents([doc])
        for _doc in _sub_docs:
            _doc.metadata[id_key] = _id
        sub_docs.extend(_sub_docs)
    return sub_docs


def summaries_func(docs):
    # 方法2 生成摘要
    chain = (
            {"doc": lambda x: x.page_content}
            | PromptTemplate.from_template("总结下列文档:\n\n{doc}")
            | llm
            | StrOutputParser()
    )
    summaries = chain.batch(docs, {"max_concurrency": 5})
    summary_docs = [Document(page_content=s, metadata={id_key: doc_ids[i]}) for i, s in enumerate(summaries)]
    return summary_docs


def hypo_questions_func(docs):
    # 方法3 生成假设性的问题
    functions = [
        {
            "name": "hypothetical_questions",
            "description": "生成假设性问题",
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                    },
                },
                "required": ["questions"]
            }
        }
    ]
    chain = (
            {"doc": lambda x: x.text}
            # {"doc": lambda x: x}
            | PromptTemplate.from_template("生成3个假设问题，下面的文档可以用来回答这些问题:\n\n{doc}")
            | llm.bind(functions=functions, function_call={"name": "hypothetical_questions"})
            | JsonKeyOutputFunctionsParser(key_name="questions")
    )
    print('hypothetical_questions')
    hypothetical_questions = []
    print(docs[0])
    hypothetical_questions = chain.batch(docs, {"max_concurrency": 5})
    question_docs = []
    for i, question_list in enumerate(hypothetical_questions):
        print(i)
        question_docs.extend([Document(page_content=s, metadata={id_key: doc_ids[i]}) for s in question_list])
    return question_docs


def get_docs(func_num, docs):
    if func_num == 1:
        return full_documents_func(docs)
    elif func_num == 2:
        return summaries_func(docs)
    elif func_num == 3:
        return hypo_questions_func(docs)
    else:
        return []


# 配置项
model_path = "../models/Baichuan2-7B-Chat"
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

pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
)

llm = HuggingFacePipeline(pipeline=pipeline)
query = "老年人经常进行体力活动，是否会减缓衰老或降低患病概率？"
# sorted_categories = classify_text(query)
# sorted_categories = classify_question(query)
# 加载文件
loaders = knowledgebase.split.load_classified_files_to_knowledge_base('../database/txt', query)
# text_splitter = RecursiveCharacterTextSplitter(separators=["。"], chunk_size=1024, chunk_overlap=64)
text_splitter = ChineseRecursiveTextSplitter(
    keep_separator=True,
    is_separator_regex=True,
    chunk_size=1024,
    chunk_overlap=64
)
texts_chunks_list = []
documents = []
for loader in loaders:
    document = loader.load_and_split()
    documents.extend(document)
    texts_chunks = text_splitter.split_documents(document)
    texts_chunks_list.extend(texts_chunks)
# texts_list = [text.page_content for text in texts_chunks_list]

texts_generation = []
for chunk in texts_chunks_list:
    text = chunk.page_content  # 假设文档中的文本是生成的文本
    metadata = chunk.metadata  # 假设文档的元数据包含生成信息
    message = BaseMessage(content=text, additional_kwargs=metadata)
    # 创建 Generation 对象
    generation = ChatGeneration(text=text, message=message)
    texts_generation.append(generation)

# dataset = MyDataset(texts_chunks_list)
# 创建 DataLoader 对象
# dataloader = DataLoader(dataset, batch_size=5)

# multi_vector_retriever设置
# 用于索引子chunk的向量存储
vectorstore = Chroma(collection_name="hypo-questions", embedding_function=embeddings)
# 父文档的存储层
store = InMemoryByteStore()
id_key = "doc_id"

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)
doc_ids = [str(uuid.uuid4()) for _ in texts_chunks_list]

func_num = 3
candidate_docs = get_docs(func_num, texts_generation)

retriever.vectorstore.add_documents(candidate_docs)
retriever.docstore.mset(list(zip(doc_ids, texts_chunks_list)))
# 验证检索结果
sub_docs = vectorstore.similarity_search(query)[0].page_content
print(sub_docs)

retrieved_docs = retriever.get_relevant_documents(query)[0].page_content
print(retrieved_docs)
# 提示模板
template = """你是一名智能助手，可以根据上下文回答用户的问题。

已知内容：
{context}

问题：
{question}
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])
# 构造chain验证
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(context=retrieved_docs, question=query)
print("Answer:\n", result)
