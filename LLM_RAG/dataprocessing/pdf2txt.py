import os
import PyPDF2


def pdf_to_text(pdf_file_path, txt_file_path):
    # 打开 PDF 文件
    with open(pdf_file_path, 'rb') as pdf_file:
        # 创建一个 PDF 解析器对象
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # 初始化一个变量，用于存储提取的文本内容
        text_content = ""

        # 遍历 PDF 的每一页
        for page_num in range(len(pdf_reader.pages)):
            # 获取当前页面的内容
            page = pdf_reader.pages[page_num]
            text_content += page.extract_text()

        # 写入提取的文本内容到文本文件中
        with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text_content)


def batch_convert_pdf_to_txt(pdf_folder_path):
    # 遍历文件夹中的每个 PDF 文件
    for pdf_file_name in os.listdir(pdf_folder_path):
        if pdf_file_name.endswith('.pdf'):
            # 构建 PDF 文件的完整路径
            pdf_file_path = os.path.join(pdf_folder_path, pdf_file_name)
            # 构建对应的 TXT 文件路径
            txt_file_path = os.path.splitext(pdf_file_path)[0] + '.txt'
            # 转换 PDF 文件为 TXT 文件
            pdf_to_text(pdf_file_path, txt_file_path)

    print("所有 PDF 文件已成功转换为 TXT 文件。")


# 主程序
def main():
    pdf_folder_path = "E:\\AAA\\database\\pdf\\健康"  # 替换为包含 PDF 文件的文件夹路径
    batch_convert_pdf_to_txt(pdf_folder_path)


if __name__ == "__main__":
    main()
