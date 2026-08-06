import requests
import os

# Токен берётся из переменной окружения
key = os.getenv('GITHUB_MODELS_KEY')
if not key:
    raise ValueError('GITHUB_MODELS_KEY не задан. Установи переменную окружения.')

payload = {
    'model': 'gpt-4o-mini',
    'messages': [
        {'role': 'system', 'content': 'Ты — Сорен, амбарная сова.'},
        {'role': 'user', 'content': 'Привет!'}
    ],
    'temperature': 0.6,
    'max_tokens': 256
}

resp = requests.post(
    'https://models.inference.ai.azure.com/chat/completions',
    headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
    json=payload,
    timeout=30
)
print(f'Status: {resp.status_code}')
print(f'Body: {resp.text[:500]}')
