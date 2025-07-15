import torch
import knowledgebase.split
from textsplitter.ChineseRecursiveTextSplitter import ChineseRecursiveTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever, SVMRetriever, TFIDFRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import PromptTemplate
from langchain_community.document_transformers import LongContextReorder
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain, StuffDocumentsChain


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
def longcontextchat(query):
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
    bm25_retriever.k = 5

    tfidf_retriever = TFIDFRetriever.from_texts(texts_list)
    tfidf_retriever.k = 5

    vectorstore = Chroma.from_documents(texts_chunks_list, embeddings, persist_directory="db")
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, tfidf_retriever], weights=[0.5, 0.5])
    # 存入向量库，创建retriever

    # 文档重排序
    # query = "老年人经常进行体力活动，是否会减缓衰老或降低患病概率？"
    docs = ensemble_retriever.get_relevant_documents(query)

    # 相关性小的文档放在中间，相关性大的文档放在首尾两端
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(docs)

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
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # 初始化chain并测试
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=document_prompt,
        document_variable_name=document_variable_name,
    )
    result = chain.run(input_documents=reordered_docs, question=query)
    print(result)
    return result

if __name__ == "__main__":
    query = "老年人经常进行体力活动，是否会减缓衰老或降低患病概率？"
    result = longcontextchat(query)