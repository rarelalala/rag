from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
import chatmodel
import os

os.environ["OPENAI_API_KEY"] = "sk-regiGkMh6PqOgTbrcbHybmoFBPqwlPMNq5uY4dFT1LB3Nt9T"


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

file_path = "question_and_truth.txt"  # Replace this with the path to your txt file
questions, ground_truths = read_questions_and_truths_from_file(file_path)
print(ground_truths)

answers = []
contexts = []
 
# Inference
for query in questions:
    response = chatmodel.get_response(query)
    answer = extract_content_after_substring(response, "答案：")
    print(answer)
    context = extract_content_between_substrings(response, "已知内容：", "问题：")
    print(context)
    answers.append(answer)
    contexts.append([context])
 
# To dict
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truths": ground_truths
}
 
# Convert dict to dataset
dataset = Dataset.from_dict(data)

result = evaluate(
    dataset = dataset, 
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
)
 
df = result.to_pandas()
df
