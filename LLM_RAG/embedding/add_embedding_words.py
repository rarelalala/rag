import os
import torch
from datetime import datetime
from safetensors.torch import save_model
from sentence_transformers import SentenceTransformer


def get_keyword_embedding(bert_model, tokenizer, key_words):
    tokenizer_output = tokenizer(key_words, return_tensors="pt", padding=True, truncation=True)
    input_ids = tokenizer_output['input_ids']
    input_ids = input_ids[:, 1:-1]

    keyword_embedding = bert_model.embeddings.word_embeddings(input_ids)
    keyword_embedding = torch.mean(keyword_embedding, 1)
    return keyword_embedding


def add_keyword_to_model(model_name='bge-large-zh-v1.5', keyword_file: str = "keywords.txt", output_model_path: str = None):
    key_words = []
    with open(keyword_file, "r") as f:
        for line in f:
            key_words.append(line.strip())

    st_model = SentenceTransformer(model_name)
    key_words_len = len(key_words)
    word_embedding_model = st_model._first_module()
    bert_model = word_embedding_model.auto_model
    tokenizer = word_embedding_model.tokenizer
    key_words_embedding = get_keyword_embedding(bert_model, tokenizer, key_words)

    embedding_weight = bert_model.embeddings.word_embeddings.weight
    embedding_weight_len = len(embedding_weight)
    tokenizer.add_tokens(key_words)
    bert_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=32)
    embedding_weight = bert_model.embeddings.word_embeddings.weight
    with torch.no_grad():
        embedding_weight[embedding_weight_len:embedding_weight_len + key_words_len, :] = key_words_embedding

    if output_model_path:
        os.makedirs(output_model_path, exist_ok=True)
        word_embedding_model.save(output_model_path)
        safetensors_file = os.path.join(output_model_path, "model.safetensors")
        metadata = {'format': 'pt'}
        save_model(bert_model, safetensors_file, metadata)
        print("save model to {}".format(output_model_path))


def add_keyword_to_embedding_model(keyword_file: str = "keywords.txt"):
    model_name = "../models/bge-large-zh-v1.5"
    model_parent_directory = os.path.dirname(model_name)
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_model_name = "{}_Merge_Keywords_{}".format('bge-large-zh-v1.5', current_time)
    output_model_path = os.path.join(model_parent_directory, output_model_name)
    add_keyword_to_model(model_name, keyword_file, output_model_path)


if __name__ == '__main__':
    add_keyword_to_embedding_model("../LLM_RAG/embedding/keywords.txt")