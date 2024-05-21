import glob
import logging
import numpy
import os

import requests
from tqdm import tqdm

TOKEN = os.getenv('SOY_TOKEN')
MODEL = "gpt-4-turbo-2024-04-09"

def load_content_from_files(text_dir='../dataset/texts', object_dir='../dataset/objects', answer_dir='../dataset/answers'):
    texts = [open(file, 'r', encoding='utf-8').read() for file in sorted(glob.glob(f"{text_dir}/*.txt"))]

    objects = [sorted(glob.glob(f"{object_dir}/{i+1}_*.txt")) for i in range(len(texts))]
    objects_content = [[open(obj_file, 'r', encoding='utf-8').read() for obj_file in obj_list] for obj_list in objects]

    answers = [sorted(glob.glob(f"{answer_dir}/{i+1}_*.txt")) for i in range(len(texts))]
    answers_content = [[open(ans_file, 'r', encoding='utf-8').read() for ans_file in ans_list] for ans_list in answers]

    return texts, objects_content, answers_content

def generate_prompt(text, obj):
    return f"Напиши ОК"  # simple prompt for fast testing

def prepare_and_send_requests(model, texts, objects_content):
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
        'model': 'text-embedding-ada-002',  # the only available embedding model
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

    flat_answers = [answer for sublist in answers_content for answer in sublist]

    for model_response, correct_answer in tqdm(zip(model_responses, flat_answers), total=len(model_responses), desc="Evaluating Responses"):
        similarity_score = calculate_similarity(model_response, correct_answer)
        print(model_response, correct_answer, similarity_score)
        if similarity_score is not None:
            total_similarity += similarity_score
            total_count += 1
        else:
            print("Warning: A similarity score could not be calculated for one or more pairs.")

    if total_count > 0:
        mean_similarity = total_similarity / total_count
        print(f"Mean quality: {mean_similarity:.4f}")
    else:
        print("Unable to calculate mean similarity due to lack of valid similarity scores.")

if __name__ == "__main__":
    texts, objects_content, answers_content = load_content_from_files()
    model_responses = prepare_and_send_requests(MODEL, texts, objects_content)
    print(model_responses)
    print(answers_content)
    print_metrics(model_responses, answers_content)