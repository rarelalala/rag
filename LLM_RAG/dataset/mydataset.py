from torch.utils.data import Dataset, DataLoader
from langchain_core.outputs.generation import Generation

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        document = self.data[idx]
        # 假设从文档中提取生成的文本和相关信息
        text = document.page_content  # 假设文档中的文本是生成的文本
        generation_info = document.metadata  # 假设文档的元数据包含生成信息
        # 创建 Generation 对象
        generation = Generation(text=text, generation_info=generation_info)
        return [generation]