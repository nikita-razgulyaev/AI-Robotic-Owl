"""
Авторизация панели мониторинга — простая cookie-сессия на общем пароле.

Не требует внешних зависимостей: подпись токена — HMAC-SHA256 на стандартной
библиотеке (hmac/hashlib), без JWT-библиотек. Токен = "<unix_expiry>.<hmac>".
Пароль один общий (это личный робот, а не сервис на много пользователей) —
см. PANEL_PASSWORD в config/settings.py. Если пароль не задан — авторизация
выключена (обратная совместимость с текущим поведением по умолчанию).
"""
import hmac
import hashlib
import json
import time
from typing import Optional

from config.settings import SESSION_SECRET


def create_session_token(remember: bool, short_max_age_sec: int, remember_max_age_sec: int) -> str:
    """Создаёт подписанный токен сессии с истечением через short/remember секунд"""
    max_age = remember_max_age_sec if remember else short_max_age_sec
    expiry = int(time.time()) + max_age
    payload = str(expiry)
    signature = hmac.new(SESSION_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{signature}"


def verify_session_token(token: Optional[str]) -> bool:
    """Проверяет подпись и срок действия токена сессии"""
    if not token or "." not in token:
        return False

    payload, _, signature = token.partition(".")
    expected_signature = hmac.new(SESSION_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()

    # compare_digest — защита от timing-атак при сравнении подписи
    if not hmac.compare_digest(signature, expected_signature):
        return False

    try:
        expiry = int(payload)
    except ValueError:
        return False

    return time.time() < expiry


def is_valid_device_ping(text: str, device_key: str) -> bool:
    """Проверяет, что первое сообщение — легитимный ping от ESP32.
    Если device_key не настроен (пусто) — доверяем любому ping (как раньше,
    обратная совместимость со старой прошивкой без DEVICE_KEY)."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False

    if data.get("type") != "ping":
        return False

    if not device_key:
        return True

    return data.get("device_key") == device_key
