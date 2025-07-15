import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

model_path = "../models/Baichuan2-7B-Chat"
print('model_path=', model_path)


if __name__ == '__main__':
    messages = []
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained(model_path)
    messages.append({"role": "user", "content": "面对死亡如何进行自我调节？"})
    response = model.chat(tokenizer, messages)
    print(response)
