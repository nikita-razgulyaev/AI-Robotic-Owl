"""Определяет, ночь сейчас или день, для заданных координат — используется
подсветкой-ночником в подставке (см. modules/backlight_state.py). Расчёт
полностью офлайн через astral (астрономические формулы), без обращения к
внешним API погоды/времени — координаты и часовой пояс задаются в
config.yaml (секция backlight), по умолчанию Вологда."""
import logging
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from astral import LocationInfo
from astral.sun import sun

logger = logging.getLogger(__name__)


def is_night(latitude: float, longitude: float, tz_name: str, now: Optional[datetime] = None) -> bool:
    """True, если сейчас между заходом и восходом солнца (тёмное время суток)
    для заданных координат/часового пояса.

    Не требует данных на "завтра": ночь — это либо ДО сегодняшнего восхода
    (продолжение вчерашней ночи), либо ПОСЛЕ сегодняшнего заката."""
    try:
        tz = ZoneInfo(tz_name)
        now = now.astimezone(tz) if now else datetime.now(tz)
        loc = LocationInfo(latitude=latitude, longitude=longitude)
        today_sun = sun(loc.observer, date=now.date(), tzinfo=tz)

        if now < today_sun["sunrise"]:
            return True
        if now >= today_sun["sunset"]:
            return True
        return False
    except Exception as e:
        # Полярный день/ночь на экстремальных широтах может уронить расчёт
        # (astral кидает ValueError, если солнце не восходит/не заходит) —
        # в этом случае считаем "не ночь", чтобы не залипнуть в вечно горящей
        # или вечно потушенной подсветке по ошибке.
        logger.error(f"Не удалось посчитать закат/рассвет ({latitude},{longitude},{tz_name}): {e}")
        return False
