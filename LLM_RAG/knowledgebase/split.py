import os
import documentloaders.main
import knowledgebase.classify_text
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain_community.document_loaders import TextLoader


# 文件解析函数
def parse_file(file_path):
    if file_path.endswith('.docx'):
        file = documentloaders.main.RapidOCRDocLoader(file_path)
    elif file_path.endswith('.pptx'):
        file = documentloaders.main.RapidOCRPPTLoader(file_path)
    elif file_path.endswith('.pdf'):
        # file = PyPDFLoader(file_path)
        file = documentloaders.main.RapidOCRPDFLoader(file_path)
    elif file_path.endswith('.img'):
        file = documentloaders.main.RapidOCRLoader(file_path)
    elif file_path.endswith('.txt'):
        file = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file format")
    return file


# 加载文件到知识库
def load_files_to_knowledge_base(directory):
    # folders = ['互联网', '社交', '健康', '心理', '医疗', '退休']
    folders = ['健康']
    knowledge_base = []
    for folder in folders:
        folder_path = os.path.join(directory, folder)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            # print(file_path)
            if file_path.endswith('.txt'):
                text = parse_file(file_path)
                knowledge_base.append(text)
    return knowledge_base


# 加载文件到知识库
def load_classified_files_to_knowledge_base(directory, query):
    # sorted_categories = knowledgebase.classify_text.classify_text(query)
    sorted_categories = knowledgebase.classify_text.classify_question(query)
    folders = ['互联网', '社交', '健康', '心理', '医疗', '退休']
    knowledge_base = []
    print(sorted_categories)
    for category in sorted_categories:
        if category in folders:
            folder_path = os.path.join(directory, category)
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if file_path.endswith('.txt'):
                    text = parse_file(file_path)
                    knowledge_base.append(text)
    return knowledge_base

