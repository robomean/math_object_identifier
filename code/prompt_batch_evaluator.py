import glob
import logging
import numpy
import os

import requests
from tqdm import tqdm


TOKEN = os.getenv('SOY_TOKEN')
MODEL = "gpt-4-turbo-2024-04-09"


def load_content_from_files(text_dir='../dataset/texts', object_dir='../dataset/objects', answer_dir='../dataset/answers'):
    text_files = sorted(glob.glob(f"{text_dir}/*.txt"), key=lambda x: int(x.split("/")[-1].split(".")[0]))
    texts = [open(file, 'r', encoding='utf-8').read() for file in text_files]

    objects_content = []
    for i in range(1, len(texts) + 1):
        object_files = sorted(glob.glob(f"{object_dir}/{i}_*.txt"), key=lambda x: (int(x.split("/")[-1].split("_")[0]), int(x.split("/")[-1].split("_")[1].split(".")[0])))
        objects = [open(obj_file, 'r', encoding='utf-8').read() for obj_file in object_files]
        objects_content.append(objects)

    answers_content = []
    for i in range(1, len(texts) + 1):
        answer_files = sorted(glob.glob(f"{answer_dir}/{i}_*.txt"), key=lambda x: (int(x.split("/")[-1].split("_")[0]), int(x.split("/")[-1].split("_")[1].split(".")[0])))
        answers = [open(ans_file, 'r', encoding='utf-8').read() for ans_file in answer_files]
        answers_content.append(answers)

    return texts, objects_content, answers_content

def generate_prompt(text, obj):
    return f"Write ОК"  # simple prompt for fast testing

def prepare_and_send_requests(model, texts, objects_content, a):
    model_responses = []

    for text, obj_list in tqdm(list(zip(texts, objects_content)), desc="Processing texts"):
        for obj in tqdm(obj_list, desc="Object loop", leave=False):
            prompt = generate_prompt(text, obj)
            headers = {
                "Authorization": f"OAuth {TOKEN}",
            }
            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0
            }

            response = requests.post('http://soyproxy.yandex-team.ru/proxy/openai/v1/chat/completions', json=data, headers=headers, timeout=300)
            if response.status_code == 200:
                model_responses.append(response.json()['response']['choices'][0]['message']['content'])
            else:
                logging.error(f"Error: {response.status_code} {response.json()['response']['error']['message']}")

    return model_responses

def get_embedding(text):
    headers = {
        'Authorization': f'OAuth {TOKEN}',
    }
    data = {
        'input': text,
        'model': 'text-embedding-ada-002',
    }

    response = requests.post('http://soyproxy.yandex-team.ru/proxy/openai/v1/embeddings', headers=headers, json=data)
    if response.status_code == 200:
        embedding = response.json()['response']['data'][0]['embedding']
        return embedding
    else:
        print(f"Error: Unable to fetch the embedding. Status Code: {response.status_code}")
        return None

def cosine_similarity(a, b):
    a = numpy.array(a)
    b = numpy.array(b)
    return numpy.dot(a, b) / (numpy.linalg.norm(a) * numpy.linalg.norm(b))

def calculate_similarity(text1, text2):
    embedding1 = get_embedding(text1)
    embedding2 = get_embedding(text2)
    
    if embedding1 is not None and embedding2 is not None:
        return cosine_similarity(embedding1, embedding2)
    else:
        print("Could not calculate cosine similarity due to an error in fetching embeddings.")

def print_metrics(model_responses, answers_content):
    total_similarity = 0
    total_count = 0
    no_info_answer_count = 0
    no_info_model_right_count = 0
    no_info_model_wrong_count = 0
    info_count = 0
    info_similarity_total = 0

    flat_answers = [answer for sublist in answers_content for answer in sublist]

    for model_response, correct_answer in tqdm(zip(model_responses, flat_answers), total=len(model_responses), desc="Evaluating Responses"):
        if "No info about this object in text" in correct_answer:
            no_info_answer_count += 1
            if "No info about this object in text" in model_response:
                no_info_model_right_count += 1
        elif "No info about this object in text" in model_response:
            no_info_model_wrong_count += 1
        similarity_score = calculate_similarity(model_response, correct_answer)
        if similarity_score is not None:
            total_similarity += similarity_score
            total_count += 1
            if "No info about this object in text" not in correct_answer:
                info_count += 1
                info_similarity_total += similarity_score
        else:
            print("Warning: A similarity score could not be calculated for one or more pairs.")

    if total_count > 0:
        mean_similarity = total_similarity / total_count
        print(f"Mean quality: {mean_similarity:.4f}")
    else:
        print("Unable to calculate mean similarity due to lack of valid similarity scores.")
    
    if info_count > 0:
        info_mean_similarity = info_similarity_total / info_count
        print(f"Mean quality for responses with information: {info_mean_similarity:.4f}")
    else:
        print("No information-based responses for calculating metrics.")

    print(f"Files with 'No info about this object in text': {no_info_answer_count}")
    print(f"Wrong 'No info about this object in text': {no_info_model_wrong_count}")
    print(f"Right 'No info about this object in text': {no_info_model_right_count}")

if __name__ == "__main__":
    texts, objects_content, answers_content = load_content_from_files()
    model_responses = prepare_and_send_requests(MODEL, texts, objects_content, answers_content)
    print_metrics(model_responses, answers_content)