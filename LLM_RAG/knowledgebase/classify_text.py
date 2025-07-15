import os
import jieba
from fuzzywuzzy import process


# 定义关键词列表，用于各个类别的匹配
fold_categories = {
    '互联网': ['互联网', '网络', '网站', '在线', '数字化', '数字鸿沟', '手机'],
    '社交': ['社交', '工作', '就业', '教育', '朋友', '自理', '收入', '消费', '生活', '社会'],
    '健康': ['健康', '体力', '健身', '营养', '锻炼', '保健', '生理', '病情', '共病', '身体', '衰弱', '衰老'],
    '心理': ['心理', '心理健康', '心理咨询', '情感', '心理医生', '抑郁', '失眠', '焦虑', '认知', '满意度', '幸福感'],
    '医疗': ['医疗', '医生', '医院', '诊所', '健康护理', '体检', '社会福利'],
    '退休': ['退休', '养老', '退休金', '储蓄', '投资']
}


# 将输入文本分类到对应类别
def classify_text(input_text):
    # 对输入文本进行分词
    words = jieba.cut(input_text)

    # 计算每个类别的得分
    scores = {category: 0 for category in fold_categories}
    for word in words:
        for category, keywords in fold_categories.items():
            if word in keywords:
                scores[category] += 1

    # 找到得分排名前两位的类别
    sorted_categories = sorted(scores, key=scores.get, reverse=True)

    if scores[sorted_categories[0]] == 0:
        return sorted_categories

    # 如果得分排名前两位的类别得分之差小于等于1，且它们都不为0，则将文本分到这两个类别中
    if scores[sorted_categories[0]] - scores[sorted_categories[1]] <= 1 and scores[sorted_categories[1]] != 0:
        return sorted_categories[:2]
    else:
        return [sorted_categories[0]]  # 否则只返回得分最高的类别


def classify_question(question):
    folder_path = "../database/txt"

    # 获取文件夹下所有文件夹
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

    # 用于存储最佳匹配的文件夹及其相似度得分
    best_folder = None
    best_score = 0

    # 遍历每个子文件夹
    for folder in subfolders:
        file_names = [f.name for f in os.scandir(folder) if f.is_file()]

        # 计算当前文件夹下所有文件名与问题的相似度得分
        match = process.extractOne(question, file_names)

        # 检查是否有匹配项
        if match is not None:
            matched_file, score = match

            # 更新最佳匹配文件夹
            if score > best_score:
                best_score = score
                best_folder = os.path.basename(folder)

    if best_folder:
        print("最匹配的文件夹:", best_folder)
        return [best_folder]
    else:
        print("未找到匹配的文件夹.")
        return ['互联网', '社交', '健康', '心理', '医疗', '退休']

