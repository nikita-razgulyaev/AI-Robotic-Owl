# Состояния и события — общий словарь

Единый словарь для поведения, локомоции, эмоций и UI. Не дублируйте имена в других модулях — ссылайтесь сюда.

## 1. Опора (датчики лапа / крыло)

Источник: `{"type":"sensor","paw":bool,"wing":bool}` с ESP32 → `modules/behavior.py`.

| Состояние (`SupportState`) | Лапа | Крыло | Смысл |
|---|---|---|---|
| `unknown` | — | — | Данных ещё не было |
| `standing` | 1 | 0 | Стоит на опоре |
| `held` | 0 | 1 | Держат в руках на весу |
| `held_touch` | 1 | 1 | Держат, но лапа касается опоры |
| `none` | 0 | 0 | Опоры нет (полёт / падение / прыжок) |

| Событие | Переход | Смысл |
|---|---|---|
| `support_lost` | standing → none | Ушёл с опоры (падение, если не self-flight) |
| `released` | held/held_touch → none | Отпустили в воздухе |
| `grabbed` | none/standing → held/held_touch | Взяли в руки |
| `touchdown` | none → standing/held_touch | Касание опоры |
| `support_regained` | held/held_touch → standing | Поставили на опору |

Флаг **self-flight**: после команды `takeoff` потеря опоры не считается падением (таймаут `BEHAVIOR_SELF_FLIGHT_TIMEOUT_SEC`).

Правила реакций: `character/behavior/rules.yaml` (hot-reload).

## 2. Режим управления локомоцией

`modules/control_mode.py`

| Режим | Геймпад | Панель walk/fly/… | Автономный планировщик (будущее) |
|---|---|---|---|
| `manual` | да | да | нет |
| `autonomous` | нет | да (override) | да |

Переключение:
- панель, вкладка **Движение**;
- WS/команда `{type:"set_control_mode","mode":"manual"|"autonomous"}`;
- голос: триггеры в `quick_answers.json` (`control_mode_manual` / `control_mode_autonomous`).

## 3. Фазы локомоции (логические)

Не enum в коде пока — ориентиры для чеклистов и будущих датчиков IMU:

| Фаза | Описание |
|---|---|
| `idle_stand` | Стоит, сервы у нейтрали / calm |
| `walk_cycle` | Идёт `build_walk` |
| `turn_ground` | Поворот на земле `build_turn` |
| `takeoff` | `build_takeoff` |
| `flight_cruise` | `build_flight_forward` / взмахи |
| `turn_air` | Поворот в полёте (тот же `build_turn` + кинематика хвоста) |
| `landing` | `build_landing` |
| `emergency` | `emergency_stop` → 90° |

## 4. Эмоциональные оси / метки

Используются в `Soren_emotions.json`, quick_answers, LLM:

`calm`, `curious`, `loving`, `playful`, `excited`, `confident`, `sad`, `alert`, …

Эмоция → поза серв через `emotion_engine.get_servo_angles(emotion)`.

## 5. Приоритеты арбитра (`modules/arbiter.py`)

| Priority | Кто | Может перебить |
|---|---|---|
| SAFETY | emergency_stop, падение | всё |
| PANEL | ручные команды панели | idle / environment |
| GAMEPAD | геймпад | idle / environment |
| ENVIRONMENT | behavior rules | idle |
| IDLE | idle_thoughts, micro_motion | — |

## 6. Где править

| Что | Файл |
|---|---|
| Состояния/события опоры | `modules/behavior.py` |
| Правила реакций | `character/behavior/rules.yaml` |
| Режим управления | `modules/control_mode.py` |
| Кинематика шага/взмаха | `modules/motion_primitives.py` |
| Голосовые переключатели | `character/quick_answers.json` |
