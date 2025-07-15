import re
import os


def starts_with_number(line):
    # 阿拉伯数字的正则表达式
    arabic_number_pattern = r'^\d'
    # 汉字数字的正则表达式
    chinese_number_pattern = r'^[一二三四五六七八九十]'
    # 括号包围的汉字数字的正则表达式
    parenthesized_chinese_number_pattern = r'^\（[一二三四五六七八九十]\）'
    # 汉字“第”的正则表达式
    character_pattern = r'^第'

    # 检查是否以阿拉伯数字开头
    if re.match(arabic_number_pattern, line):
        return True
        # 检查是否以汉字数字开头
    elif re.match(chinese_number_pattern, line):
        return True
        # 检查是否以括号包围的汉字数字开头
    elif re.match(parenthesized_chinese_number_pattern, line):
        return True
    # 检查是否以“第”字开头
    elif re.match(character_pattern, line):
        return True
    else:
        return False


def remove_citations(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    # 使用正则表达式匹配并替换标注符号
    clean_text = re.sub(r'\[[\d]+\]', '', text)
    clean_text2 = re.sub(r'［[\d]+］', '', clean_text)
    cleaned_text = re.sub(r'〔[\d]+〕', '', clean_text2)
    # new_text= cleaned_text.replace(' ', '')

    # 将修改后的文本保存回原来的文本文件中
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(cleaned_text)

    print("标注符号已成功删除，并已保存到原始文件中。")


def remove_newlines_except_titles_and_periods(file_path):
    # 打开文件，读取内容
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 初始化一个变量，用于标记是否处于标题行
    in_title = False

    # 初始化一个变量，用于存储处理后的文本内容
    modified_content = ""

    # 遍历文件的每一行
    for line in lines:
        # 判断是否是标题行（以数字加一段描述性文字开头）
        if starts_with_number(line):
            in_title = True
        else:
            in_title = False

        # 如果不是标题行，则删除除句号之外的换行符
        if not in_title:
            if line.rstrip('\n').endswith(('。', '？', '！')):
                modified_content += line
            else:
                modified_content += line.rstrip('\n')
        else:
            modified_content += line

    # 将修改后的内容写回文件
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(modified_content)

    print("除标题和以句号结尾的换行符外的所有换行符已成功删除，并已保存到原始文件中。")


def get_path(directory):
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)


# 主程序
def main():
    txt_file_path = "E:\\AAA\\database\\txt\\草稿.txt"  # 替换为你的文本文件路径
    directory = "E:\\AAA\\database\\test"
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        remove_newlines_except_titles_and_periods(file_path)
        remove_citations(file_path)


if __name__ == "__main__":
    main()
