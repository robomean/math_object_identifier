import numpy
import os
import requests


def get_embedding(text, token):
    headers = {
        'Authorization': f'OAuth {token}',
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

def cosine_similarity(vec_a, vec_b):
    vec_a = numpy.array(vec_a)
    vec_b = numpy.array(vec_b)
    return numpy.dot(vec_a, vec_b) / (numpy.linalg.norm(vec_a) * numpy.linalg.norm(vec_b))

def main(text1, text2, token):
    embedding1 = get_embedding(text1, token)
    embedding2 = get_embedding(text2, token)
    
    if embedding1 is not None and embedding2 is not None:
        similarity = cosine_similarity(embedding1, embedding2)
        print(f"Cosine Similarity: '{:.4f}'.format(similarity)")
    else:
        print("Could not calculate cosine similarity due to an error in fetching embeddings.")

if __name__ == "__main__":
    text1 = """1. a set of parking functions on [n].
2. having cardinality |Pn| = (n+1)^(n-1).
"""
    text2 = """1. Set of parking functions on [n]
2. |P_n| = (n + 1) ^ (n − 1)"""
    
    token = os.getenv('SOY_TOKEN')
    
    main(text1, text2, token)