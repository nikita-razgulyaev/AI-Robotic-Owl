import requests
import os
from dotenv import load_dotenv

load_dotenv()  # без этого os.getenv() не увидит переменные из .env

# Токен берётся из переменной окружения (huggingface.co/settings/tokens,
# право доступа "Make calls to Inference Providers")
key = os.getenv('HF_TOKEN')
if not key:
    raise ValueError('HF_TOKEN не задан. Установи переменную окружения.')

payload = {
    'model': 'meta-llama/Llama-3.3-70B-Instruct:fastest',
    'messages': [
        {'role': 'system', 'content': 'Ты — Сорен, амбарная сова.'},
        {'role': 'user', 'content': 'Привет!'}
    ],
    'temperature': 0.6,
    'max_tokens': 256
}

resp = requests.post(
    'https://router.huggingface.co/v1/chat/completions',
    headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
    json=payload,
    timeout=30
)
print(f'Status: {resp.status_code}')
print(f'Body: {resp.text[:500]}')