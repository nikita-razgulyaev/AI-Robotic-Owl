"""LLM модуль — Сорен (local llama.cpp / cloud Hugging Face Inference Providers) с Vector RAG, памятью и LoRA"""
import logging
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from config.settings import (
    LLM_MODE, LLM_MODEL_PATH, LLM_N_CTX, LLM_N_THREADS, LLM_TEMPERATURE, LLM_REPEAT_PENALTY,
    CHARACTER_DIR, RAG_TOP_K, CLOUD_API_KEY, CLOUD_MODEL_NAME,
    LLM_LOCAL_MODELS, LLM_CLOUD_MODELS,
    LORA_ENABLED, LORA_PATH, LORA_SCALE, FAST_MODE
)
from modules.qdrant_singleton import get_qdrant_client, get_encoder, encode_text
from modules import memory_flags

logger = logging.getLogger(__name__)


# === VECTOR RAG через Qdrant + Sentence-Transformers ===

class VectorRAG:
    """Векторный RAG через Qdrant — семантический поиск по чанкам"""

    def __init__(self):
        self.client = None
        self.encoder = None
        self._init()

    def _init(self):
        try:
            self.client = get_qdrant_client()
            self.encoder = get_encoder()
            logger.info("✅ VectorRAG инициализирован (shared Qdrant)")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации VectorRAG: {e}")
            self.client = None
            self.encoder = None

    def search(self, query: str, top_k: int = 3) -> List[str]:
        """Семантический поиск по RAG-чанкам"""
        if self.client is None:
            logger.warning("VectorRAG недоступен, возвращаем пустой результат")
            return []

        try:
            query_vector = encode_text(query)
            results = self.client.search(
                collection_name="soren_rag_chunks",
                query_vector=query_vector,
                limit=top_k
            )
            texts = [r.payload["text"] for r in results]
            logger.info(f"🔍 VectorRAG: найдено {len(texts)} чанков для запроса")
            return texts
        except Exception as e:
            logger.error(f"Ошибка поиска в VectorRAG: {e}")
            return []

    def search_with_scores(self, query: str, top_k: int = 3) -> List[Dict]:
        """Поиск с оценками сходства"""
        if self.client is None:
            return []

        try:
            query_vector = encode_text(query)
            results = self.client.search(
                collection_name="soren_rag_chunks",
                query_vector=query_vector,
                limit=top_k
            )
            return [
                {
                    "text": r.payload["text"],
                    "score": round(r.score, 3),
                    "tags": r.payload.get("tags", []),
                    "source": r.payload.get("source", "")
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Ошибка поиска в VectorRAG: {e}")
            return []


# === Fallback keyword-based RAG ===

class KeywordRAG:
    """Простой keyword-based RAG — fallback"""

    def __init__(self, chunks_path: Path):
        self.chunks: List[Dict] = []
        self._load(chunks_path)

    def _load(self, path: Path):
        if not path.exists():
            logger.warning(f"RAG файл не найден: {path}")
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.chunks.append(json.loads(line))
            logger.info(f"Keyword RAG загружено: {len(self.chunks)} чанков")
        except Exception as e:
            logger.error(f"Ошибка загрузки RAG: {e}")

    def search(self, query: str, top_k: int = 3) -> List[str]:
        query_lower = query.lower()
        keywords = set(re.findall(r'[\w\-]+', query_lower))

        scored = []
        for chunk in self.chunks:
            score = 0
            text_lower = chunk["text"].lower()
            tags = [t.lower() for t in chunk.get("tags", [])]

            for kw in keywords:
                if kw in tags:
                    score += 10
                if kw in text_lower:
                    score += 3

            triggers = {
                "клудд": ["клудд", "брат", "металлический", "предательство"],
                "гильфи": ["гильфи", "друг", "подруга", "сычик"],
                "эзилриб": ["эзилриб", "учитель", "наставник"],
                "пеллиппер": ["пеллиппер", "любовь", "сердце"],
                "сант": ["сант-эголиус", "плен", "эголиус"],
                "древо": ["древо", "га'хул", "стражи"],
                "чистые": ["чистые", "крупинки", "враг"],
            }
            for trigger_word, related in triggers.items():
                if trigger_word in query_lower:
                    for rel in related:
                        if rel in text_lower or rel in tags:
                            score += 5

            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk["text"] for _, chunk in scored[:top_k]]


# === Облачный LLM через Hugging Face Inference Providers (OpenAI-совместимый роутер) ===

class CloudLLMEngine:
    """Облачный LLM через Hugging Face Inference Providers — роутер сам выбирает
    самого быстрого провайдера (Cerebras/Groq/и др.) под нужную модель"""

    def __init__(self, memory_manager=None, model_name: str = None):
        self.api_key = CLOUD_API_KEY
        self.model = model_name or CLOUD_MODEL_NAME
        self.base_url = "https://router.huggingface.co/v1/chat/completions"
        self.conversation_history = []
        self.system_prompt = ""
        self.vector_rag = None
        self.keyword_rag = None
        self.emotion_keywords = {}
        self.memory = memory_manager

        self._load_system_prompt()
        self._load_rag()
        self._load_emotion_keywords()

        if not self.api_key:
            logger.warning("HF_TOKEN не задан! Облачный LLM не будет работать.")
        if not self.model:
            logger.warning("llm.cloud_model не задан в config.yaml! Облачный LLM не будет работать.")
        logger.info(f"☁️ Облачный LLM (Hugging Face Inference Providers) инициализирован: {self.model}")

    def _load_system_prompt(self):
        # fast_mode: короткий промпт (character/Soren_short.txt) — без полного канона,
        # меньше токенов на обработку промпта -> быстрее ответ. Иначе — полный Soren.txt.
        prompt_filename = "Soren_short.txt" if FAST_MODE else "Soren.txt"
        prompt_path = CHARACTER_DIR / prompt_filename
        if prompt_path.exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read()
            logger.info(f"System prompt загружен ({prompt_filename}): {len(self.system_prompt)} chars")
        else:
            self.system_prompt = "Ты — Сорен, амбарная сова, Главный Страж. Отвечай мудро и сдержанно."

    def _load_rag(self):
        # Грузим тяжёлые ресурсы (Qdrant, sentence-transformers энкодер, файл
        # канона) только если уровень "rag" включён по умолчанию (обычно = not
        # fast_mode). Если выключен, self.vector_rag/keyword_rag остаются None —
        # _ensure_rag_loaded() поднимет их лениво, если уровень включат из
        # панели уже во время работы сервера.
        if memory_flags.get_flags().get("rag", not FAST_MODE):
            self._ensure_rag_loaded()
        else:
            self.vector_rag = None
            self.keyword_rag = None

    def _ensure_rag_loaded(self):
        """Ленивая инициализация RAG-ресурсов при включении уровня 'rag' из панели"""
        if self.vector_rag is not None or self.keyword_rag is not None:
            return
        self.vector_rag = VectorRAG()
        rag_path = CHARACTER_DIR / "Soren_rag_chunks.json"
        if not rag_path.exists():
            rag_path = CHARACTER_DIR / "Soren_rag_chunks.jsonl"
        self.keyword_rag = KeywordRAG(rag_path)

    def _load_emotion_keywords(self):
        emotions_path = CHARACTER_DIR / "Soren_emotions.json"
        if emotions_path.exists():
            try:
                with open(emotions_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for emotion_name, emotion_data in data.get("emotions", {}).items():
                    self.emotion_keywords[emotion_name] = emotion_data.get("triggered_by", [])
                logger.info(f"Эмоции загружены: {list(self.emotion_keywords.keys())}")
            except Exception as e:
                logger.error(f"Ошибка загрузки эмоций: {e}")

    def _detect_emotion(self, text: str) -> str:
        text_lower = text.lower()
        emotion_scores = {}
        for emotion, triggers in self.emotion_keywords.items():
            score = 0
            for trigger in triggers:
                if trigger.lower() in text_lower:
                    score += 1
            if score > 0:
                emotion_scores[emotion] = score

        if emotion_scores:
            return max(emotion_scores, key=emotion_scores.get)

        if any(w in text_lower for w in ["клудд", "пепел", "потерял", "погиб", "тень", "скорбь"]):
            return "sad"
        if any(w in text_lower for w in ["огонь", "коготь", "буря", "гнев", "убить", "враг"]):
            return "angry"
        if any(w in text_lower for w in ["пеллиппер", "любовь", "тепло", "гнездо", "доверие", "сердце"]):
            return "loving"
        if any(w in text_lower for w in ["должен", "вперёд", "защищать", "битва", "миссия"]):
            return "determined"
        if any(w in text_lower for w in ["что...", "не может быть", "удивлён", "вспышка"]):
            return "surprised"
        if any(w in text_lower for w in ["устал", "отдохнуть", "пепел", "закат", "позже"]):
            return "tired"

        return "calm"

    def _build_messages(self, user_message: str, vision_context: str = "", memory_context: str = "") -> List[Dict]:
        rag_chunks = []
        if memory_flags.get_flags().get("rag", True):
            rag_chunks = self.vector_rag.search(user_message, top_k=RAG_TOP_K) if self.vector_rag else []
            if not rag_chunks and self.keyword_rag:
                rag_chunks = self.keyword_rag.search(user_message, top_k=RAG_TOP_K)

        rag_text = "\n".join(rag_chunks) if rag_chunks else ""

        system_content = self.system_prompt
        if rag_text:
            system_content += f"\n\nРелевантные воспоминания из канона:\n{rag_text}"
        if vision_context:
            system_content += f"\n\n[Ты видишь: {vision_context}]"
        if memory_context:
            system_content += f"\n\n[Память о пользователе:\n{memory_context}]"

        messages = [{"role": "system", "content": system_content}]

        for msg in self.conversation_history[-5:]:
            messages.append(msg)

        messages.append({"role": "user", "content": user_message})
        return messages

    def generate(self, user_message: str, vision_context: str = "", memory_context: str = "") -> dict:
        if not self.api_key:
            return {
                "text": "HF_TOKEN не задан. Получи токен на https://huggingface.co/settings/tokens (с правом Make calls to Inference Providers) и добавь в .env",
                "action": None,
                "emotion": "calm"
            }

        try:
            import requests

            messages = self._build_messages(user_message, vision_context, memory_context)

            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": LLM_TEMPERATURE,
                "max_tokens": 256
            }

            json_body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            logger.info(f"HF Inference Providers запрос: model={self.model}, messages={len(messages)}")

            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json; charset=utf-8"
                },
                data=json_body,
                timeout=60
            )

            logger.info(f"HF Inference Providers ответ: status={response.status_code}")

            if response.status_code == 200:
                data = response.json()
                response_text = data["choices"][0]["message"]["content"].strip()
            else:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
                except:
                    error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                logger.error(f"HF Inference Providers ошибка: {error_msg}")
                return {
                    "text": f"*(пауза)* ... Прости, друг. Ветер сменил направление. Повтори, пожалуйста. (Ошибка: {error_msg})",
                    "action": None,
                    "emotion": "calm"
                }

            response_text = re.sub(r'^(Сорен:|Assistant:|AI:)', '', response_text).strip()
            response_text = re.sub(r'\*.*?\*', '', response_text)

            action = None
            action_match = re.search(r'\[ACTION:(\w+)\]', response_text)
            if action_match:
                action = action_match.group(1)
                response_text = re.sub(r'\[ACTION:\w+\]', '', response_text).strip()

            emotion = self._detect_emotion(response_text)

            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": response_text})

            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

            return {"text": response_text, "action": action, "emotion": emotion}

        except Exception as e:
            logger.error(f"Ошибка облачного LLM: {e}")
            return {
                "text": "*(пауза)* ... Прости, друг. Мысли улетели далеко. Повтори, пожалуйста.",
                "action": None,
                "emotion": "calm"
            }

    def clear_history(self):
        self.conversation_history = []
        logger.info("История диалога очищена")


# === Local LLM через llama.cpp ===

class LocalLLMEngine:
    """Локальный LLM через llama.cpp с поддержкой LoRA"""

    def __init__(self, memory_manager=None, model_path: Path = None):
        self.model = None
        self.model_path = Path(model_path) if model_path else LLM_MODEL_PATH
        self.conversation_history = []
        self.system_prompt = ""
        self.vector_rag = None
        self.keyword_rag = None
        self.emotion_keywords = {}
        self.memory = memory_manager

        self._load_system_prompt()
        self._load_rag()
        self._load_emotion_keywords()
        self._load_model()

    def _load_system_prompt(self):
        # fast_mode: короткий промпт (character/Soren_short.txt) — без полного канона,
        # меньше токенов на обработку промпта -> быстрее ответ. Иначе — полный Soren.txt.
        prompt_filename = "Soren_short.txt" if FAST_MODE else "Soren.txt"
        prompt_path = CHARACTER_DIR / prompt_filename
        if prompt_path.exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read()
            logger.info(f"System prompt загружен ({prompt_filename}): {len(self.system_prompt)} chars")
        else:
            self.system_prompt = "Ты — Сорен, амбарная сова, Главный Страж. Отвечай мудро и сдержанно."

    def _load_rag(self):
        # Грузим тяжёлые ресурсы (Qdrant, sentence-transformers энкодер, файл
        # канона) только если уровень "rag" включён по умолчанию (обычно = not
        # fast_mode). Если выключен, self.vector_rag/keyword_rag остаются None —
        # _ensure_rag_loaded() поднимет их лениво, если уровень включат из
        # панели уже во время работы сервера.
        if memory_flags.get_flags().get("rag", not FAST_MODE):
            self._ensure_rag_loaded()
        else:
            self.vector_rag = None
            self.keyword_rag = None

    def _ensure_rag_loaded(self):
        """Ленивая инициализация RAG-ресурсов при включении уровня 'rag' из панели"""
        if self.vector_rag is not None or self.keyword_rag is not None:
            return
        self.vector_rag = VectorRAG()
        rag_path = CHARACTER_DIR / "Soren_rag_chunks.json"
        if not rag_path.exists():
            rag_path = CHARACTER_DIR / "Soren_rag_chunks.jsonl"
        self.keyword_rag = KeywordRAG(rag_path)

    def _load_emotion_keywords(self):
        emotions_path = CHARACTER_DIR / "Soren_emotions.json"
        if emotions_path.exists():
            try:
                with open(emotions_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for emotion_name, emotion_data in data.get("emotions", {}).items():
                    self.emotion_keywords[emotion_name] = emotion_data.get("triggered_by", [])
                logger.info(f"Эмоции загружены: {list(self.emotion_keywords.keys())}")
            except Exception as e:
                logger.error(f"Ошибка загрузки эмоций: {e}")

    def _load_model(self):
        if self.model_path is None:
            logger.error(
                "LLM_MODEL_PATH не задан (llm.local_model_path отсутствует в config.yaml) — "
                "локальная модель не может быть загружена."
            )
            self.model = None
            return

        logger.info(f"Загрузка LLM: {self.model_path}")
        if not self.model_path.exists():
            logger.error(f"Модель LLM не найдена: {self.model_path}")
            return
        try:
            from llama_cpp import Llama

            kwargs = {
                "model_path": str(self.model_path),
                "n_ctx": LLM_N_CTX,
                "n_threads": LLM_N_THREADS,
                "verbose": False
            }

            if LORA_ENABLED and LORA_PATH.exists():
                kwargs["lora_path"] = str(LORA_PATH)
                kwargs["lora_scale"] = LORA_SCALE
                logger.info(f"LoRA загружен: {LORA_PATH} (scale={LORA_SCALE})")
            elif LORA_ENABLED:
                logger.warning(f"LoRA файл не найден: {LORA_PATH}")

            self.model = Llama(**kwargs)
            logger.info("LLM загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            self.model = None

    def _detect_emotion(self, text: str) -> str:
        text_lower = text.lower()
        emotion_scores = {}
        for emotion, triggers in self.emotion_keywords.items():
            score = 0
            for trigger in triggers:
                if trigger.lower() in text_lower:
                    score += 1
            if score > 0:
                emotion_scores[emotion] = score

        if emotion_scores:
            return max(emotion_scores, key=emotion_scores.get)

        if any(w in text_lower for w in ["клудд", "пепел", "потерял", "погиб", "тень", "скорбь"]):
            return "sad"
        if any(w in text_lower for w in ["огонь", "коготь", "буря", "гнев", "убить", "враг"]):
            return "angry"
        if any(w in text_lower for w in ["пеллиппер", "любовь", "тепло", "гнездо", "доверие", "сердце"]):
            return "loving"
        if any(w in text_lower for w in ["должен", "вперёд", "защищать", "битва", "миссия"]):
            return "determined"
        if any(w in text_lower for w in ["что...", "не может быть", "удивлён", "вспышка"]):
            return "surprised"
        if any(w in text_lower for w in ["устал", "отдохнуть", "пепел", "закат", "позже"]):
            return "tired"

        return "calm"

    def _build_prompt(self, user_message: str, vision_context: str = "", memory_context: str = "") -> List[Dict]:
        rag_chunks = []
        if memory_flags.get_flags().get("rag", True):
            rag_chunks = self.vector_rag.search(user_message, top_k=RAG_TOP_K) if self.vector_rag else []
            if not rag_chunks and self.keyword_rag:
                rag_chunks = self.keyword_rag.search(user_message, top_k=RAG_TOP_K)

        rag_text = "\n".join(rag_chunks) if rag_chunks else ""

        messages = []
        system_content = self.system_prompt
        if rag_text:
            system_content += f"\n\nРелевантные воспоминания из канона:\n{rag_text}"
        if vision_context:
            system_content += f"\n\n[Ты видишь: {vision_context}]"
        if memory_context:
            system_content += f"\n\n[Память о пользователе:\n{memory_context}]"

        messages.append({"role": "system", "content": system_content})

        for msg in self.conversation_history[-10:]:
            messages.append(msg)

        messages.append({"role": "user", "content": user_message})
        return messages

    def generate(self, user_message: str, vision_context: str = "", memory_context: str = "") -> dict:
        if self.model is None:
            return {
                "text": "Модель LLM не загружена. Проверь логи и настройки.",
                "action": None,
                "emotion": "calm"
            }

        try:
            messages = self._build_prompt(user_message, vision_context, memory_context)

            output = self.model.create_chat_completion(
                messages=messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=256,
                repeat_penalty=LLM_REPEAT_PENALTY,
                stop=["</s>", "Пользователь:", "User:"],
            )

            response_text = output["choices"][0]["message"]["content"].strip()
            response_text = re.sub(r'^(Сорен:|Assistant:|AI:)', '', response_text).strip()
            response_text = re.sub(r'\*.*?\*', '', response_text)

            action = None
            action_match = re.search(r'\[ACTION:(\w+)\]', response_text)
            if action_match:
                action = action_match.group(1)
                response_text = re.sub(r'\[ACTION:\w+\]', '', response_text).strip()

            emotion = self._detect_emotion(response_text)

            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": response_text})

            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

            return {"text": response_text, "action": action, "emotion": emotion}

        except Exception as e:
            logger.error(f"Ошибка LLM: {e}")
            return {
                "text": "*(пауза)* ... Прости, друг. Мысли улетели далеко, словно перо на ветру. Повтори, пожалуйста.",
                "action": None,
                "emotion": "calm"
            }

    def clear_history(self):
        self.conversation_history = []
        logger.info("История диалога очищена")


# === Универсальный LLM движок ===

class LLMEngine:
    """Универсальный LLM движок с переключением local/cloud и памятью"""

    def __init__(self, memory_manager=None):
        self.mode = LLM_MODE
        self.local_engine = None
        self.cloud_engine = None
        self.memory = memory_manager
        # Текущая выбранная модель (id из config.yaml llm.local_models/cloud_models).
        # Если текущий путь/имя модели не совпадает ни с одним элементом
        # куратированного списка (например, задан вручную в config.yaml),
        # используем сам путь/имя как id — панель тогда просто не подсветит
        # ни один пункт списка активным, но всё продолжит работать.
        self.local_model_id = self._find_local_model_id(str(LLM_MODEL_PATH) if LLM_MODEL_PATH else None)
        self.cloud_model_id = self._find_cloud_model_id(CLOUD_MODEL_NAME)

        if self.mode == "cloud":
            self.cloud_engine = CloudLLMEngine(memory_manager)
            logger.info("☁️ LLM режим: ОБЛАЧНЫЙ (Hugging Face Inference Providers)")
        else:
            self.local_engine = LocalLLMEngine(memory_manager)
            logger.info("💻 LLM режим: ЛОКАЛЬНЫЙ (llama.cpp + LoRA)")

    @staticmethod
    def _find_local_model_id(path_str: Optional[str]) -> Optional[str]:
        if not path_str:
            return None
        for entry in LLM_LOCAL_MODELS:
            if entry.get("path") and Path(entry["path"]).name == Path(path_str).name:
                return entry.get("id")
        return path_str

    @staticmethod
    def _find_cloud_model_id(model_name: Optional[str]) -> Optional[str]:
        if not model_name:
            return None
        for entry in LLM_CLOUD_MODELS:
            if entry.get("model") == model_name:
                return entry.get("id")
        return model_name

    def set_mode(self, mode: str):
        if mode not in ["local", "cloud"]:
            logger.warning(f"Неверный режим LLM: {mode}. Используем 'local'")
            mode = "local"

        self.mode = mode
        if mode == "cloud":
            if self.cloud_engine is None:
                self.cloud_engine = CloudLLMEngine(self.memory)
            self.local_engine = None
            logger.info("☁️ LLM переключён на ОБЛАЧНЫЙ")
        else:
            if self.local_engine is None:
                self.local_engine = LocalLLMEngine(self.memory)
            self.cloud_engine = None
            logger.info("💻 LLM переключён на ЛОКАЛЬНЫЙ")

    def get_mode(self) -> str:
        return self.mode

    def get_current_model_id(self) -> Optional[str]:
        return self.local_model_id if self.mode == "local" else self.cloud_model_id

    def set_model(self, model_id: str) -> bool:
        """Переключает конкретную модель по id из config.yaml (llm.local_models
        или llm.cloud_models) и сразу перезагружает соответствующий движок —
        без рестарта сервера. Список, в котором нашёлся id, определяет и режим
        (local/cloud), поэтому переключение модели может заодно переключить режим."""
        for entry in LLM_LOCAL_MODELS:
            if entry.get("id") == model_id:
                logger.info(f"💻 Загружаю локальную модель '{model_id}' ({entry.get('path')})…")
                self.local_engine = LocalLLMEngine(self.memory, model_path=entry.get("path"))
                self.cloud_engine = None
                self.mode = "local"
                self.local_model_id = model_id
                return True

        for entry in LLM_CLOUD_MODELS:
            if entry.get("id") == model_id:
                logger.info(f"☁️ Переключаюсь на облачную модель '{model_id}' ({entry.get('model')})…")
                self.cloud_engine = CloudLLMEngine(self.memory, model_name=entry.get("model"))
                self.local_engine = None
                self.mode = "cloud"
                self.cloud_model_id = model_id
                return True

        logger.warning(f"Неизвестный id модели LLM: {model_id}")
        return False

    def ensure_rag_loaded(self):
        """Лениво поднимает RAG-ресурсы (Qdrant + энкодер + канон) активного
        движка — вызывается при включении уровня 'rag' из веб-панели, если
        сервер стартовал с выключенным RAG (fast_mode или ручной выбор)"""
        engine = self.cloud_engine if self.mode == "cloud" else self.local_engine
        if engine is not None:
            engine._ensure_rag_loaded()

    def generate(self, user_message: str, vision_context: str = "", memory_context: Optional[str] = None) -> dict:
        # Если вызывающий код (RobotBrain._build_memory_context) уже собрал контекст
        # памяти (включая релевантные долгосрочные воспоминания из Qdrant) — используем
        # его. Иначе (например, при прямом вызове LLMEngine в тестах) строим сами,
        # но это более простой вариант — без семантического поиска по LTM.
        if memory_context is None and self.memory:
            memory_context = self.memory.get_context_for_llm()
        memory_context = memory_context or ""

        if self.mode == "cloud" and self.cloud_engine:
            result = self.cloud_engine.generate(user_message, vision_context, memory_context)
        elif self.local_engine:
            result = self.local_engine.generate(user_message, vision_context, memory_context)
        else:
            return {"text": "LLM движок не инициализирован.", "action": None, "emotion": "calm"}

        if self.memory:
            self.memory.record_interaction(user_message, result["text"], result["emotion"])

        return result

    def clear_history(self):
        if self.cloud_engine:
            self.cloud_engine.clear_history()
        if self.local_engine:
            self.local_engine.clear_history()