# -*- coding: utf-8 -*-
"""User-interface localisation: English (default), Russian, Ukrainian."""

from __future__ import annotations

UI_LANGS = ["en", "ru", "uk"]
UI_NAMES = {"en": "English", "ru": "Русский", "uk": "Українська"}
DEFAULT_UI = "en"

STRINGS = {
    # ---------------- window and sections ----------------
    "app.title": {
        "en": "Vigenere Cracker",
        "ru": "Дешифратор Виженера",
        "uk": "Дешифратор Віженера",
    },
    "sec.cipher": {"en": "Ciphertext", "ru": "Шифртекст", "uk": "Шифротекст"},
    "sec.params": {"en": "Attack settings", "ru": "Параметры атаки",
                   "uk": "Параметри атаки"},
    "sec.run": {"en": "Run", "ru": "Выполнение", "uk": "Виконання"},
    "sec.results": {
        "en": "Candidates (double-click to copy the text)",
        "ru": "Кандидаты (двойной клик — копировать текст)",
        "uk": "Кандидати (подвійний клік — копіювати текст)",
    },
    "sec.log": {"en": "Log", "ru": "Журнал", "uk": "Журнал"},

    # ---------------- fields ----------------
    "lbl.uilang": {"en": "Interface:", "ru": "Интерфейс:", "uk": "Інтерфейс:"},
    "lbl.textlang": {"en": "Text language:", "ru": "Язык текста:", "uk": "Мова тексту:"},
    "lbl.alphabet": {"en": "Alphabet:", "ru": "Алфавит:", "uk": "Абетка:"},
    "lbl.variant": {"en": "Cipher variant:", "ru": "Вариант шифра:",
                    "uk": "Варіант шифру:"},
    "lbl.mode": {"en": "Mode:", "ru": "Режим:", "uk": "Режим:"},
    "lbl.keylen": {"en": "Key length: from", "ru": "Длина ключа: от",
                   "uk": "Довжина ключа: від"},
    "lbl.to": {"en": "to", "ru": "до", "uk": "до"},
    "lbl.device": {"en": "Compute:", "ru": "Вычислитель:", "uk": "Обчислювач:"},
    "lbl.procs": {"en": "Processes:", "ru": "Процессов:", "uk": "Процесів:"},
    "lbl.restarts": {"en": "Restarts (heuristic):", "ru": "Рестартов (эвристика):",
                     "uk": "Рестартів (евристика):"},
    "lbl.penalty": {"en": "Key-length penalty:", "ru": "Штраф за длину ключа:",
                    "uk": "Штраф за довжину ключа:"},
    "lbl.penalty.hint": {
        "en": "(stops long keys from winning by overfitting)",
        "ru": "(борьба с «подгонкой» длинным ключом)",
        "uk": "(боротьба з «підгонкою» довгим ключем)",
    },
    "lbl.reversed": {
        "en": "also reversed words in the dictionary",
        "ru": "в словаре также обратные слова",
        "uk": "у словнику також зворотні слова",
    },
    "lbl.letters": {"en": "letters: %d", "ru": "букв: %d", "uk": "літер: %d"},
    "lbl.manual": {"en": "Check a key manually:", "ru": "Проверить ключ вручную:",
                   "uk": "Перевірити ключ вручну:"},
    "lbl.nogpu": {"en": "no CUDA", "ru": "нет CUDA", "uk": "немає CUDA"},

    # ---------------- buttons ----------------
    "btn.paste": {"en": "Paste", "ru": "Вставить", "uk": "Вставити"},
    "btn.clear": {"en": "Clear", "ru": "Очистить", "uk": "Очистити"},
    "btn.start": {"en": "▶  Start", "ru": "▶  Начать", "uk": "▶  Почати"},
    "btn.stop": {"en": "■  Stop", "ru": "■  Стоп", "uk": "■  Стоп"},
    "btn.export": {"en": "Export CSV", "ru": "Экспорт CSV", "uk": "Експорт CSV"},

    # ---------------- results table ----------------
    "col.n": {"en": "#", "ru": "#", "uk": "#"},
    "col.key": {"en": "Key", "ru": "Ключ", "uk": "Ключ"},
    "col.alpha": {"en": "Alph.", "ru": "Алф.", "uk": "Абет."},
    "col.score": {"en": "Score", "ru": "Оценка", "uk": "Оцінка"},
    "col.ng": {"en": "n-gram/char", "ru": "н-грамм/симв", "uk": "н-грам/симв"},
    "col.words": {"en": "Words", "ru": "Слов", "uk": "Слів"},
    "col.plain": {"en": "Decryption", "ru": "Расшифровка", "uk": "Розшифровка"},

    # ---------------- attack modes ----------------
    "mode.auto": {
        "en": "Auto: strategy by text length",
        "ru": "Авто: стратегия по длине текста",
        "uk": "Авто: стратегія за довжиною тексту",
    },
    "mode.freq": {
        "en": "Frequency analysis (long text)",
        "ru": "Частотный анализ (длинный текст)",
        "uk": "Частотний аналіз (довгий текст)",
    },
    "mode.brute": {
        "en": "Brute force (short key)",
        "ru": "Полный перебор (короткий ключ)",
        "uk": "Повний перебір (короткий ключ)",
    },
    "mode.dict": {
        "en": "Dictionary attack (key is a word)",
        "ru": "Словарная атака (ключ — слово)",
        "uk": "Словникова атака (ключ — слово)",
    },
    "mode.hill": {
        "en": "Heuristic / hill-climbing",
        "ru": "Эвристика / hill-climbing",
        "uk": "Евристика / hill-climbing",
    },

    # ---------------- cipher variants ----------------
    "cip.vigenere": {"en": "Vigenere (c = p + k)", "ru": "Виженер (c = p + k)",
                     "uk": "Віженер (c = p + k)"},
    "cip.beaufort": {"en": "Beaufort (c = k - p)", "ru": "Бофор (c = k - p)",
                     "uk": "Бофор (c = k - p)"},
    "cip.variant": {"en": "Variant Beaufort (c = p - k)",
                    "ru": "Виженер-вариант (c = p - k)",
                    "uk": "Віженер-варіант (c = p - k)"},

    # ---------------- alphabet ----------------
    "alpha.auto": {"en": "Auto: check all variants", "ru": "Авто: проверить все варианты",
                   "uk": "Авто: перевірити всі варіанти"},
    "alpha.one": {"en": "%s letters", "ru": "%s букв", "uk": "%s літер"},

    # ---------------- status line ----------------
    "st.ready": {"en": "Ready.", "ru": "Готов к работе.", "uk": "Готовий до роботи."},
    "st.starting": {"en": "Starting…", "ru": "Запуск…", "uk": "Запуск…"},
    "st.stopping": {"en": "Stopping…", "ru": "Останавливаю…", "uk": "Зупиняю…"},
    "st.stopped": {"en": "Stopped by user.", "ru": "Остановлено пользователем.",
                   "uk": "Зупинено користувачем."},
    "st.done": {"en": "Done in %s.", "ru": "Готово за %s.", "uk": "Готово за %s."},
    "st.failed": {"en": "Failed — see the log.", "ru": "Завершено с ошибкой — см. журнал.",
                  "uk": "Завершено з помилкою — див. журнал."},
    "st.stats": {
        "en": "%5.1f%%   keys: %s / %s   speed: %s k/s   elapsed: %s   left: %s",
        "ru": "%5.1f%%   ключей: %s / %s   скорость: %s кл/с   прошло: %s   осталось: %s",
        "uk": "%5.1f%%   ключів: %s / %s   швидкість: %s кл/с   минуло: %s   лишилось: %s",
    },

    # ---------------- key-space hints ----------------
    "hint.space": {
        "en": "Key space: %s  ≈ %s  (%s, ~%s keys/s)",
        "ru": "Пространство ключей: %s  ≈ %s  (%s, ~%s кл/с)",
        "uk": "Простір ключів: %s  ≈ %s  (%s, ~%s кл/с)",
    },
    "hint.short": {
        "en": "Short text (%d < %d letters) → brute force + dictionary + analysis",
        "ru": "Текст короткий (%d < %d букв) → перебор + словарь + анализ",
        "uk": "Текст короткий (%d < %d літер) → перебір + словник + аналіз",
    },
    "hint.long": {
        "en": "Long text (%d letters) → the key is computed by frequency analysis, "
              "no brute force needed",
        "ru": "Текст длинный (%d букв) → ключ вычисляется частотным анализом, "
              "перебор не нужен",
        "uk": "Текст довгий (%d літер) → ключ обчислюється частотним аналізом, "
              "перебір не потрібен",
    },
    "hint.freq": {
        "en": "The key is computed column by column: work grows as L, not as 32^L",
        "ru": "Ключ вычисляется по столбцам: работа растёт как L, а не как 32^L",
        "uk": "Ключ обчислюється по стовпцях: робота зростає як L, а не як 32^L",
    },

    # ---------------- dialog messages ----------------
    "msg.needtext": {
        "en": "Enter a ciphertext (at least 3 letters of the chosen language).",
        "ru": "Введите шифртекст (минимум 3 буквы выбранного языка).",
        "uk": "Введіть шифротекст (щонайменше 3 літери обраної мови).",
    },
    "msg.modelwait": {
        "en": "The language model is still being built, please wait a few seconds.",
        "ru": "Языковая модель ещё строится, подождите пару секунд.",
        "uk": "Мовна модель ще будується, зачекайте кілька секунд.",
    },
    "msg.numbers": {"en": "Check the numeric fields.", "ru": "Проверьте числовые поля.",
                    "uk": "Перевірте числові поля."},
    "msg.bigspace": {
        "en": "Key space: %s variants, about %s of work.\n"
              "For long keys the Frequency analysis mode is much faster.\n\nContinue?",
        "ru": "Пространство ключей: %s вариантов, примерно %s работы.\n"
              "Для длинных ключей быстрее режим «Частотный анализ».\n\nПродолжить?",
        "uk": "Простір ключів: %s варіантів, приблизно %s роботи.\n"
              "Для довгих ключів швидший режим «Частотний аналіз».\n\nПродовжити?",
    },
    "msg.noresults": {"en": "No results to export.", "ru": "Нет результатов для экспорта.",
                      "uk": "Немає результатів для експорту."},
    "msg.nodata": {
        "en": "Language data for %s is not downloaded.\nRun: python download_data.py %s\n"
              "Without it the quality on short texts is much worse. Continue anyway?",
        "ru": "Языковые данные для «%s» не скачаны.\nЗапустите: python download_data.py %s\n"
              "Без них качество на коротких текстах заметно хуже. Всё равно продолжить?",
        "uk": "Мовні дані для «%s» не завантажені.\nЗапустіть: python download_data.py %s\n"
              "Без них якість на коротких текстах помітно гірша. Все одно продовжити?",
    },

    # ---------------- log (interface side) ----------------
    "log.buildmodel": {
        "en": "Building the language model (%s)…",
        "ru": "Строю языковую модель (%s)…",
        "uk": "Будую мовну модель (%s)…",
    },
    "log.modelready": {
        "en": "Model \"%s\" ready in %.1f s: %s words in the dictionary.",
        "ru": "Модель «%s» готова за %.1f c: %s слов в словаре.",
        "uk": "Модель «%s» готова за %.1f с: %s слів у словнику.",
    },
    "log.cuda": {"en": "CUDA ready: %s (warm-up %.1f s)",
                 "ru": "CUDA готова: %s (прогрев %.1f c)",
                 "uk": "CUDA готова: %s (прогрів %.1f с)"},
    "log.start": {"en": "Start: mode=%s, key %d..%d, %s",
                  "ru": "Старт: режим=%s, ключ %d..%d, %s",
                  "uk": "Старт: режим=%s, ключ %d..%d, %s"},
    "log.copied": {"en": "Copied: %s", "ru": "Скопировано: %s", "uk": "Скопійовано: %s"},
    "log.exported": {"en": "Exported: %s", "ru": "Экспортировано: %s",
                     "uk": "Експортовано: %s"},
    "log.best": {"en": "Best: key \"%s\" → %s", "ru": "Лучший: ключ «%s» → %s",
                 "uk": "Найкращий: ключ «%s» → %s"},
    "log.dictfull": {
        "en": "Key dictionary: %s words (full)",
        "ru": "Словарь ключей: %s слов (полный)",
        "uk": "Словник ключів: %s слів (повний)",
    },
    "log.dictfreq": {
        "en": "Key dictionary: %s frequent words (the full one is in Dictionary attack mode)",
        "ru": "Словарь ключей: %s частых слов (полный — в режиме «Словарная атака»)",
        "uk": "Словник ключів: %s частих слів (повний — у режимі «Словникова атака»)",
    },

    # ---------------- log (engine side) ----------------
    "eng.short": {
        "en": "Text too short: at least 3 letters of the alphabet are needed.",
        "ru": "Слишком короткий текст: нужно минимум 3 буквы алфавита.",
        "uk": "Занадто короткий текст: потрібно щонайменше 3 літери абетки.",
    },
    "eng.info": {
        "en": "Letters in ciphertext: %d | cipher: %s | alphabets: %s",
        "ru": "Букв в шифртексте: %d | вариант: %s | алфавиты: %s",
        "uk": "Літер у шифротексті: %d | варіант: %s | абетки: %s",
    },
    "eng.ic": {
        "en": "Index of coincidence (~%.3f = natural language): ",
        "ru": "Индекс совпадений (~%.3f = естественный язык): ",
        "uk": "Індекс збігів (~%.3f = природна мова): ",
    },
    "eng.kasiski": {
        "en": "Kasiski examination (repeated n-grams) suggests key length: ",
        "ru": "Метод Касиски (повторы n-грамм) даёт длину ключа: ",
        "uk": "Метод Касіскі (повтори n-грам) дає довжину ключа: ",
    },
    "eng.keytoolong": {
        "en": "WARNING: key length ≥ text length — with such a key any \"meaningful\" "
              "text fits, the result is not trustworthy.",
        "ru": "ВНИМАНИЕ: длина ключа ≥ длины текста — при таком ключе подходит любой "
              "«осмысленный» текст, результат недостоверен.",
        "uk": "УВАГА: довжина ключа ≥ довжини тексту — з таким ключем підходить "
              "будь-який «осмислений» текст, результат недостовірний.",
    },
    "eng.strategy": {"en": "Strategy: ", "ru": "Стратегия: ", "uk": "Стратегія: "},
    "eng.ph.brute": {"en": "brute force", "ru": "полный перебор", "uk": "повний перебір"},
    "eng.ph.freq": {"en": "frequency analysis", "ru": "частотный анализ",
                    "uk": "частотний аналіз"},
    "eng.ph.dict": {"en": "dictionary", "ru": "словарь", "uk": "словник"},
    "eng.ph.hill": {"en": "heuristic", "ru": "эвристика", "uk": "евристика"},
    "eng.dictprep": {
        "en": "%sKey dictionary: %s (prepared in %.1f s)",
        "ru": "%sСловарь ключей: %s (подготовлен за %.1f с)",
        "uk": "%sСловник ключів: %s (підготовлено за %.1f с)",
    },
    "eng.procs": {"en": "Processes: %d", "ru": "Процессов: %d", "uk": "Процесів: %d"},
    "eng.gpu": {"en": "GPU: %s", "ru": "GPU: %s", "uk": "GPU: %s"},
    "eng.gpubrute": {
        "en": "%sGPU, brute force of key length %d (%s variants)",
        "ru": "%sGPU, перебор ключа длины %d (%s вариантов)",
        "uk": "%sGPU, перебір ключа довжини %d (%s варіантів)",
    },
    "eng.stage.brute": {
        "en": "Brute force, key length %d (%s variants)",
        "ru": "Перебор, длина ключа %d (%s вариантов)",
        "uk": "Перебір, довжина ключа %d (%s варіантів)",
    },
    "eng.stage.dict": {"en": "Dictionary: %s keys", "ru": "Словарь: %s ключей",
                       "uk": "Словник: %s ключів"},
    "eng.stage.freq": {
        "en": "Frequency analysis, key length %d (computed column by column)",
        "ru": "Частотный анализ, длина ключа %d (вычисление по столбцам)",
        "uk": "Частотний аналіз, довжина ключа %d (обчислення по стовпцях)",
    },
    "eng.stage.hill": {
        "en": "Heuristic, key length %d (%d restarts)",
        "ru": "Эвристика, длина ключа %d (%d рестартов)",
        "uk": "Евристика, довжина ключа %d (%d рестартів)",
    },
    "eng.rescore": {
        "en": "Final re-scoring of candidates against the dictionary…",
        "ru": "Финальная переоценка кандидатов по словарю…",
        "uk": "Фінальна переоцінка кандидатів за словником…",
    },
    "eng.finished": {
        "en": "Done in %.1f s. Candidates considered: %d",
        "ru": "Готово за %.1f с. Кандидатов рассмотрено: %d",
        "uk": "Готово за %.1f с. Кандидатів розглянуто: %d",
    },
    "eng.gpuonly": {
        "en": "The GPU is used for brute force only — computing on the CPU.",
        "ru": "GPU используется только для полного перебора — считаю на CPU.",
        "uk": "GPU використовується лише для повного перебору — рахую на CPU.",
    },
    "eng.error": {"en": "ERROR: %s", "ru": "ОШИБКА: %s", "uk": "ПОМИЛКА: %s"},
    "eng.gpuoom": {
        "en": "GPU memory is tight — batch reduced to %d keys and retried.",
        "ru": "Мало видеопамяти — батч уменьшен до %d ключей, повтор.",
        "uk": "Мало відеопам'яті — батч зменшено до %d ключів, повтор.",
    },
    "eng.gpufail": {
        "en": "GPU failed (%s) — this key length will be done on the CPU.",
        "ru": "Сбой GPU (%s) — эта длина ключа будет посчитана на CPU.",
        "uk": "Збій GPU (%s) — цю довжину ключа буде обчислено на CPU.",
    },
    "log.calib": {
        "en": "Speed calibrated on this machine: %s keys/s (%s)",
        "ru": "Скорость откалибрована на этой машине: %s ключей/с (%s)",
        "uk": "Швидкість відкалібровано на цій машині: %s ключів/с (%s)",
    },
    "msg.nodata.title": {
        "en": "Language data missing", "ru": "Нет языковых данных",
        "uk": "Немає мовних даних",
    },
}


class Translator:
    """``tr('btn.start')`` -> localised string; extra args are %-formatted in."""

    def __init__(self, lang=DEFAULT_UI):
        self.lang = lang if lang in UI_LANGS else DEFAULT_UI

    def set(self, lang):
        self.lang = lang if lang in UI_LANGS else DEFAULT_UI

    def __call__(self, key, *args):
        entry = STRINGS.get(key)
        if entry is None:
            return key
        s = entry.get(self.lang) or entry.get(DEFAULT_UI) or key
        return (s % args) if args else s


def tr(lang, key, *args):
    return Translator(lang)(key, *args)

