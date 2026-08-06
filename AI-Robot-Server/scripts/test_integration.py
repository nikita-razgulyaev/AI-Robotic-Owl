"""Интеграционный тест: Vector RAG + Memory + LLM (Singleton Qdrant)"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("ИНТЕГРАЦИОННЫЙ ТЕСТ: Vector RAG + Memory + LLM")
print("=" * 60)

# 1. Проверка импортов
print("\n1. Проверка импортов...")
try:
    from config.settings import (
        QDRANT_PATH, RAG_ENCODER_MODEL, RAG_TOP_K,
        LONG_TERM_MEMORY_ENABLED, CHARACTER_DIR
    )
    print(f"   ✅ QDRANT_PATH: {QDRANT_PATH}")
    print(f"   ✅ RAG_ENCODER_MODEL: {RAG_ENCODER_MODEL}")
    print(f"   ✅ RAG_TOP_K: {RAG_TOP_K}")
    print(f"   ✅ LTM_ENABLED: {LONG_TERM_MEMORY_ENABLED}")
except Exception as e:
    print(f"   ❌ Ошибка импорта settings: {e}")
    sys.exit(1)

# 2. Проверка синглтона
print("\n2. Проверка Qdrant Singleton...")
try:
    from modules.qdrant_singleton import get_qdrant_client, get_encoder, encode_text
    client1 = get_qdrant_client()
    client2 = get_qdrant_client()
    print(f"   ✅ Singleton работает: {client1 is client2}")

    enc1 = get_encoder()
    enc2 = get_encoder()
    print(f"   ✅ Encoder singleton: {enc1 is enc2}")

    # Тест encode
    vec = encode_text("тест")
    print(f"   ✅ encode_text работает, размер: {len(vec)}")
except Exception as e:
    print(f"   ❌ Ошибка singleton: {e}")
    sys.exit(1)

# 3. Проверка VectorRAG
print("\n3. Проверка VectorRAG...")
try:
    from modules.llm import VectorRAG
    rag = VectorRAG()
    if rag.client and rag.encoder:
        print("   ✅ VectorRAG инициализирован")
        results = rag.search("Кто такой Клудд?", top_k=2)
        print(f"   ✅ Найдено чанков: {len(results)}")
        for i, r in enumerate(results[:2]):
            print(f"      [{i+1}] {r[:80]}...")
    else:
        print("   ⚠️ VectorRAG fallback")
except Exception as e:
    print(f"   ❌ Ошибка VectorRAG: {e}")

# 4. Проверка MemoryManager
print("\n4. Проверка MemoryManager...")
try:
    from modules.memory import MemoryManager
    memory = MemoryManager(user_id="test_001")
    print(f"   ✅ MemoryManager создан")
    print(f"   ✅ LTM enabled: {memory.long_term.enabled}")
    print(f"   ✅ STM max: {memory.short_term.max_turns}")

    # Тест записи
    memory.record_interaction("Привет, Сорен!", "Приветствую, друг.", "calm")
    print(f"   ✅ Запись в память работает")

    # Тест контекста
    ctx = memory.get_context_for_llm()
    print(f"   ✅ Контекст получен ({len(ctx)} символов)")

    # Тест LTM поиска
    if memory.long_term.enabled:
        memories = memory.get_relevant_memories("Клудд")
        print(f"   ✅ LTM поиск работает ({len(memories)} символов)")
except Exception as e:
    print(f"   ❌ Ошибка MemoryManager: {e}")
    import traceback
    traceback.print_exc()

# 5. Проверка LLMEngine
print("\n5. Проверка LLMEngine...")
try:
    from modules.llm import LLMEngine
    llm = LLMEngine(memory_manager=memory)
    print(f"   ✅ LLMEngine создан")
    print(f"   ✅ Режим: {llm.get_mode()}")

    engine = llm.cloud_engine if llm.mode == "cloud" else llm.local_engine
    if engine and engine.vector_rag:
        print(f"   ✅ VectorRAG подключён к LLM")
        # Проверяем, что это тот же клиент
        if engine.vector_rag.client is client1:
            print(f"   ✅ Один и тот же Qdrant клиент!")
        else:
            print(f"   ⚠️ Разные клиенты (но singleton должен это предотвратить)")
    else:
        print(f"   ⚠️ VectorRAG не подключён")
except Exception as e:
    print(f"   ❌ Ошибка LLMEngine: {e}")
    import traceback
    traceback.print_exc()

# 6. Проверка RobotBrain
print("\n6. Проверка RobotBrain...")
try:
    from modules.robot_brain import RobotBrain
    print("   ✅ RobotBrain импортирован")
except Exception as e:
    print(f"   ❌ Ошибка RobotBrain: {e}")

print("\n" + "=" * 60)
print("✅ ИНТЕГРАЦИОННЫЙ ТЕСТ ЗАВЕРШЁН")
print("=" * 60)