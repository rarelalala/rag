
import chatmodel
import time

def extract_content_between_substrings(full_string, start_substring, end_substring):
    # Find the index of the start substring
    start_index = full_string.find(start_substring)
    
    # If the start substring is not found, return None
    if start_index == -1:
        return None
    
    # Find the index of the end substring after the start index
    end_index = full_string.find(end_substring, start_index + len(start_substring))
    
    # If the end substring is not found after the start index, return None
    if end_index == -1:
        return None
    
    # Extract the content between the start and end substrings
    content_between_substrings = full_string[start_index + len(start_substring):end_index]
    
    return content_between_substrings

def extract_content_after_substring(full_string, substring):
    # Find the index of the substring
    index = full_string.find(substring)
    
    # If the substring is not found, return None
    if index == -1:
        return None
    
    # Extract the content after the substring
    content_after_substring = full_string[index + len(substring):]
    
    return content_after_substring

def read_questions_and_truths_from_file(file_path):
    questions = []
    truths = []

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

        # Iterate through the lines, considering every odd line as question and every even line as answer
        for i in range(0, len(lines), 2):
            question = lines[i].strip()  # Strip to remove leading/trailing whitespaces
            truth = lines[i + 1].strip()  # Similarly, strip to remove leading/trailing whitespaces

            if question.endswith('？'):
                questions.append(question)
                truths.append([truth])
                print(questions)

    return questions, truths

def save_to_file(filename, content):
    with open(filename, "a") as file:
        file.write(content + "\n" + "************\n")

file_path = "question_and_truth.txt"  # Replace this with the path to your txt file
questions, ground_truths = read_questions_and_truths_from_file(file_path)
print(ground_truths)
 
# Inference
for query in questions:
    start_time = time.time()
    response = chatmodel.get_response(query)

    end_time = time.time()
    # 计算运行时长（以秒为单位）
    duration = end_time - start_time
    print("代码执行时长（秒）：", duration)
    
    answer = extract_content_after_substring(response, "答案：")
    print(answer)
    context = extract_content_between_substrings(response, "已知内容：", "问题：")
    print(context)
    # save_to_file("qwen-result.txt", answer)
    # save_to_file("context.txt", context)
 