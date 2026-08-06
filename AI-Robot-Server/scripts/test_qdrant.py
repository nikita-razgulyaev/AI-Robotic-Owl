"""Тест Qdrant — создание коллекций и загрузка RAG-чанков"""
import json
import os
from pathlib import Path

print("=" * 60)
print("ТЕСТ QDRANT + SENTENCE-TRANSFORMERS")
print("=" * 60)

# 1. Проверка Qdrant
print("\n1. Проверка Qdrant Client...")
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    print("   ✅ Qdrant Client импортирован")
except Exception as e:
    print(f"   ❌ Ошибка: {e}")
    exit(1)

# 2. Подключение (локальный, файловый)
print("\n2. Подключение к Qdrant (локальный режим)...")
try:
    client = QdrantClient(path="./qdrant_storage")
    print("   ✅ Qdrant подключён (локальный)")
except Exception as e:
    print(f"   ❌ Ошибка: {e}")
    exit(1)

# 3. Создание коллекций
print("\n3. Создание коллекций...")
collections = [
    "soren_emotional_moments",
    "soren_dialogue_turns",
    "soren_facts",
    "soren_rag_chunks"
]

for name in collections:
    try:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print(f"   ✅ Создана: {name}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"   ℹ️  Уже существует: {name}")
        else:
            print(f"   ❌ Ошибка: {name} - {e}")

# 4. Проверка Sentence-Transformers
print("\n4. Загрузка эмбеддинг-модели...")
try:
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    print("   ✅ Модель загружена")
    print(f"   Размер вектора: {encoder.get_sentence_embedding_dimension()}")
except Exception as e:
    print(f"   ❌ Ошибка: {e}")
    exit(1)

# 5. Загрузка RAG-чанков
print("\n5. Загрузка RAG-чанков...")
rag_file = Path("character/Soren_rag_chunks.jsonl")
if not rag_file.exists():
    print(f"   ❌ Файл не найден: {rag_file}")
    exit(1)

chunks = []
with open(rag_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            chunks.append(json.loads(line))

print(f"   Загружено чанков: {len(chunks)}")

# 6. Векторизация и загрузка
print("\n6. Векторизация и загрузка в Qdrant...")
points = []
for i, chunk in enumerate(chunks):
    text = chunk.get("text", "")
    embedding = encoder.encode(text).tolist()
    
    points.append(PointStruct(
        id=i,
        vector=embedding,
        payload={
            "text": text,
            "tags": chunk.get("tags", []),
            "weight": chunk.get("weight", ""),
            "source": chunk.get("source", ""),
            "category": chunk.get("id", "").split("_")[1] if "_" in chunk.get("id", "") else "general"
        }
    ))
    
    if (i + 1) % 10 == 0:
        print(f"   Обработано: {i + 1}/{len(chunks)}")

# Загружаем пачками
BATCH_SIZE = 50
for i in range(0, len(points), BATCH_SIZE):
    batch = points[i:i + BATCH_SIZE]
    client.upsert(
        collection_name="soren_rag_chunks",
        points=batch
    )
    print(f"   Загружено: {min(i + BATCH_SIZE, len(points))}/{len(points)}")

print(f"\n   ✅ Все {len(points)} чанков загружены!")

# 7. Тест поиска
print("\n7. Тест семантического поиска...")
test_queries = [
    "Кто такой Клудд?",
    "Расскажи про Сант-Эголиус",
    "Что такое серебряная душа?",
    "Кто такая Гильфи?"
]

for query in test_queries:
    print(f"\n   Запрос: '{query}'")
    query_vector = encoder.encode(query).tolist()
    
    results = client.search(
        collection_name="soren_rag_chunks",
        query_vector=query_vector,
        limit=2
    )
    
    for j, res in enumerate(results):
        text_preview = res.payload["text"][:80] + "..."
        print(f"   [{j+1}] Сходство: {res.score:.3f} | {text_preview}")

print("\n" + "=" * 60)
print("✅ ТЕСТ ПРОЙДЕН УСПЕШНО!")
print("=" * 60)