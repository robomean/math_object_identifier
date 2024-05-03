import glob
import logging
import os

import requests
from tqdm import tqdm


HEADERS = {"Authorization": f"OAuth {os.getenv('SOY_TOKEN')}"}
MODEL = "gpt-4-turbo-2024-04-09"

def generate_prompt(text, obj):
    return f"Привет, вот {text}. Вот {obj}. Напиши \"ОК\", если прочитал."  # simple prompt for fast testing

def load_content_from_files(text_dir='../dataset/texts', object_dir='../dataset/objects'):
    texts = [open(file, 'r', encoding='utf-8').read() for file in sorted(glob.glob(f"{text_dir}/*.txt"))]
    objects = [sorted(glob.glob(f"{object_dir}/{i+1}_*.txt")) for i in range(len(texts))]
    objects_content = [[open(obj_file, 'r', encoding='utf-8').read() for obj_file in obj_list] for obj_list in objects]
    return texts, objects_content

def prepare_and_send_requests(model):
    texts, objects_content = load_content_from_files()

    for text, obj_list in tqdm(list(zip(texts, objects_content)), desc="Processing texts"):
        for obj in tqdm(obj_list, desc="Object loop", leave=False):
            prompt = generate_prompt(text, obj)
            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            }

            response = requests.post('http://soyproxy.yandex-team.ru/proxy/openai/v1/chat/completions', json=data, headers=HEADERS, timeout=300)
            if response.status_code == 200:
                response_text = response.json()['response']['choices'][0]['message']['content']
            else:
                logging.error(f"Error: {response.status_code} {response.json()['response']['error']['message']}")


if __name__ == "__main__":
    prepare_and_send_requests(MODEL)