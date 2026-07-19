"""LLM модуль - Сорен (local llama.cpp / cloud GitHub Models) с RAG и persona"""
import logging
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from config.settings import (
    LLM_MODE, LLM_MODEL_PATH, LLM_N_CTX, LLM_N_THREADS, LLM_TEMPERATURE,
    CHARACTER_DIR, RAG_TOP_K, GITHUB_MODELS_KEY, GITHUB_MODELS_NAME
)

logger = logging.getLogger(__name__)


class GitHubModelsLLMEngine:
    """Облачный LLM через GitHub Models API (бесплатный, без VPN)"""

    def __init__(self):
        self.api_key = GITHUB_MODELS_KEY
        self.model = GITHUB_MODELS_NAME
        self.base_url = "https://models.inference.ai.azure.com/chat/completions"
        self.conversation_history = []
        self.system_prompt = ""
        self.rag = None
        self.emotion_keywords = {}

        self._load_system_prompt()
        self._load_rag()
        self._load_emotion_keywords()

        if not self.api_key:
            logger.warning("GITHUB_MODELS_KEY не задан! Облачный LLM не будет работать.")
        logger.info(f"☁️ Облачный LLM (GitHub Models) инициализирован: {self.model}")

    def _load_system_prompt(self):
        prompt_path = CHARACTER_DIR / "Soren.txt"
        if prompt_path.exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read()
            logger.info(f"System prompt загружен: {len(self.system_prompt)} chars")
        else:
            self.system_prompt = "Ты — Сорен, амбарная сова, Главный Страж. Отвечай мудро и сдержанно."

    def _load_rag(self):
        rag_path = CHARACTER_DIR / "Soren_rag_chunks.jsonl"
        self.rag = RAGIndex(rag_path)

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

    def _build_messages(self, user_message: str, vision_context: str = "") -> List[Dict]:
        rag_chunks = self.rag.search(user_message, top_k=RAG_TOP_K) if self.rag else []
        rag_text = "\n".join(rag_chunks) if rag_chunks else ""

        system_content = self.system_prompt
        if rag_text:
            system_content += f"\n\nРелевантные воспоминания:\n{rag_text}"
        if vision_context:
            system_content += f"\n\n[Ты видишь: {vision_context}]"

        messages = [{"role": "system", "content": system_content}]

        for msg in self.conversation_history[-5:]:
            messages.append(msg)

        messages.append({"role": "user", "content": user_message})
        return messages

    def generate(self, user_message: str, vision_context: str = "") -> dict:
        if not self.api_key:
            return {
                "text": "GITHUB_MODELS_KEY не задан. Получи токен на https://github.com/settings/tokens и добавь в .env",
                "action": None,
                "emotion": "calm"
            }

        try:
            import requests

            messages = self._build_messages(user_message, vision_context)

            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": LLM_TEMPERATURE,
                "max_tokens": 256
            }

            json_body = json.dumps(payload, ensure_ascii=False).encode('utf-8')

            logger.info(f"GitHub Models запрос: model={self.model}, messages={len(messages)}")

            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json; charset=utf-8"
                },
                data=json_body,
                timeout=60
            )

            logger.info(f"GitHub Models ответ: status={response.status_code}")

            if response.status_code == 200:
                data = response.json()
                response_text = data["choices"][0]["message"]["content"].strip()
            else:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
                except:
                    error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                logger.error(f"GitHub Models ошибка: {error_msg}")
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
            logger.error(f"Ошибка облачного LLM (GitHub Models): {e}")
            import traceback
            traceback.print_exc()
            return {
                "text": "*(пауза)* ... Прости, друг. Мысли улетели далеко. Повтори, пожалуйста.",
                "action": None,
                "emotion": "calm"
            }

    def clear_history(self):
        self.conversation_history = []
        logger.info("История диалога очищена")


class RAGIndex:
    """Простой keyword-based RAG по чанкам"""

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
            logger.info(f"RAG загружено: {len(self.chunks)} чанков")
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


class LocalLLMEngine:
    """Локальный LLM через llama-cpp-python"""

    def __init__(self):
        self.model = None
        self.conversation_history = []
        self.system_prompt = ""
        self.rag = None
        self.emotion_keywords = {}

        self._load_system_prompt()
        self._load_rag()
        self._load_emotion_keywords()
        self._load_model()

    def _load_system_prompt(self):
        prompt_path = CHARACTER_DIR / "Soren.txt"
        if prompt_path.exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read()
            logger.info(f"System prompt загружен: {len(self.system_prompt)} chars")
        else:
            self.system_prompt = "Ты — Сорен, амбарная сова, Главный Страж. Отвечай мудро и сдержанно."

    def _load_rag(self):
        rag_path = CHARACTER_DIR / "Soren_rag_chunks.jsonl"
        self.rag = RAGIndex(rag_path)

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
        logger.info(f"Загрузка LLM: {LLM_MODEL_PATH}")
        if not LLM_MODEL_PATH.exists():
            logger.error(f"Модель LLM не найдена: {LLM_MODEL_PATH}")
            return
        try:
            from llama_cpp import Llama
            self.model = Llama(
                model_path=str(LLM_MODEL_PATH),
                n_ctx=LLM_N_CTX,
                n_threads=LLM_N_THREADS,
                verbose=False
            )
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

    def _build_prompt(self, user_message: str, vision_context: str = "") -> List[Dict]:
        rag_chunks = self.rag.search(user_message, top_k=RAG_TOP_K) if self.rag else []
        rag_text = "\n".join(rag_chunks) if rag_chunks else ""

        messages = []
        system_content = self.system_prompt
        if rag_text:
            system_content += f"\n\nРелевантные воспоминания для контекста:\n{rag_text}"
        if vision_context:
            system_content += f"\n\n[Ты видишь: {vision_context}]"

        messages.append({"role": "system", "content": system_content})

        for msg in self.conversation_history[-10:]:
            messages.append(msg)

        messages.append({"role": "user", "content": user_message})
        return messages

    def generate(self, user_message: str, vision_context: str = "") -> dict:
        if self.model is None:
            return {
                "text": "Модель LLM не загружена. Проверь логи и настройки.",
                "action": None,
                "emotion": "calm"
            }

        try:
            messages = self._build_prompt(user_message, vision_context)

            output = self.model.create_chat_completion(
                messages=messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=256,
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


class LLMEngine:
    """Универсальный LLM движок с переключением local/cloud"""

    def __init__(self):
        self.mode = LLM_MODE
        self.local_engine = None
        self.cloud_engine = None

        if self.mode == "cloud":
            self.cloud_engine = GitHubModelsLLMEngine()
            logger.info("☁️ LLM режим: ОБЛАЧНЫЙ (GitHub Models)")
        else:
            self.local_engine = LocalLLMEngine()
            logger.info("💻 LLM режим: ЛОКАЛЬНЫЙ (llama.cpp)")

    def set_mode(self, mode: str):
        """Переключает режим LLM"""
        if mode not in ["local", "cloud"]:
            logger.warning(f"Неверный режим LLM: {mode}. Используем 'local'")
            mode = "local"

        self.mode = mode
        if mode == "cloud":
            if self.cloud_engine is None:
                self.cloud_engine = GitHubModelsLLMEngine()
            self.local_engine = None
            logger.info("☁️ LLM переключён на ОБЛАЧНЫЙ (GitHub Models)")
        else:
            if self.local_engine is None:
                self.local_engine = LocalLLMEngine()
            self.cloud_engine = None
            logger.info("💻 LLM переключён на ЛОКАЛЬНЫЙ")

    def get_mode(self) -> str:
        return self.mode

    def generate(self, user_message: str, vision_context: str = "") -> dict:
        if self.mode == "cloud" and self.cloud_engine:
            return self.cloud_engine.generate(user_message, vision_context)
        elif self.local_engine:
            return self.local_engine.generate(user_message, vision_context)
        else:
            return {
                "text": "LLM движок не инициализирован.",
                "action": None,
                "emotion": "calm"
            }

    def clear_history(self):
        if self.cloud_engine:
            self.cloud_engine.clear_history()
        if self.local_engine:
            self.local_engine.clear_history()