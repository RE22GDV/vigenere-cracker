# -*- coding: utf-8 -*-
"""Definitions of the supported plaintext languages.

An alphabet is named ``language-variant``: ru-32, ru-33, uk-33, en-26 and so on.
A variant is a specific letter set - Russian with 32 letters (no Yo) or 33,
Ukrainian with 33 letters or 32 (no Ge-upturn). The same plaintext enciphered
under different variants yields different ciphertext, so the tool can test all
variants of a language at once.
"""

from __future__ import annotations

from . import ru_data

# --------------------------------------------------------------------------- #
#  Alphabets
# --------------------------------------------------------------------------- #
RU_33 = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
RU_32 = "абвгдежзийклмнопрстуфхцчшщъыьэюя"          # without Yo
UK_33 = "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя"
UK_32 = UK_33.replace("ґ", "")                      # without Ge-upturn
EN_26 = "abcdefghijklmnopqrstuvwxyz"


# --------------------------------------------------------------------------- #
#  Built-in text samples: a fallback until the real data is downloaded
# --------------------------------------------------------------------------- #
UK_CORPUS = """
Одного весняного вечора у старому місті на березі річки зібралися люди щоб
послухати історію яку розповідав приїжджий чоловік у сірому пальті.
Він говорив повільно і неголосно але кожне його слово було чути у дальньому
кутку майдану бо всі мовчали і ніхто не наважувався перебити оповідача.
Наука починається там де людина перестає вірити очевидному і починає
перевіряти свої здогади за допомогою точних вимірювань та повторюваних дослідів.
Кожне нове питання породжує десять нових питань і в цьому полягає головна
радість дослідника який ніколи не залишається без роботи і без надії.
Шифр Віженера довгий час вважався незламним тому що одна й та сама буква
відкритого тексту перетворювалася на різні букви зашифрованого повідомлення.
Проте якщо відома довжина ключа задача розпадається на кілька простих задач
і кожна з них розв язується звичайним частотним аналізом за короткий час.
Ранок наступного дня видався холодним над водою підіймався туман і дерева
стояли нерухомо ніби чекали чогось важливого і нікому не відомого.
Хлопчик який слухав учорашню розповідь вийшов з дому раніше за всіх і пішов
вузькою стежкою вздовж берега туди де починався ліс і закінчувалися поля.
Море було спокійним і теплим хвилі ледве чутно набігали на пісок і йшли
назад залишаючи по собі тонку смугу піни та дрібні блискучі камінці.
Місто жило своїм звичайним життям машини їхали мокрим асфальтом люди поспішали
на роботу крамниці відчинялися й зачинялися і ніхто не думав про загадки.
Час іде однаково для всіх але кожен використовує його по своєму і врешті
виявляється що в одних вийшло ціле життя а в інших лише очікування.
Знання не приходить одразу воно збирається повільно з дрібних спостережень
випадкових здогадів і впертої праці яка триває день за днем.
Той хто шукає відповідь має бути готовим до того що відповідь виявиться зовсім
не такою якою він її уявляв і що доведеться визнати свою колишню неправоту.
Помилка це не поразка а джерело інформації яке коштує набагато дорожче
ніж випадковий успіх здобутий без жодного розуміння причин того що сталося.
"""

UK_WORDS = """
і в не на я бути з він а як це по але вони ми за із у який то все вона так
його да ти к для же ви є від коли навіть немає про якщо тільки її мені було
ось хто говорити рік свій знати мій до або час рука стати великий інший наш
під де діло сам раз щоб два там ніж око життя перший день тут дуже хотіти
голова треба без бачити йти тепер той може себе сказати людина жити слово
місце робота думати питання дім стояти друг сторона потрібно розуміти робити
новий дитина сила кінець вид система частина місто жінка гроші земля машина
батько проблема година право нога рішення двері образ історія влада закон
війна випадок ніч праця світло світ душа мати обличчя ранок вечір тиждень
місяць хвилина секунда шлях дорога небо сонце місяць зірка ліс поле річка
море гора камінь дерево квітка трава звір птах риба собака кішка кінь книга
лист газета мова буква число цифра ключ замок вікно стіна дах підлога стеля
стіл стілець ліжко шафа кімната кухня вулиця площа міст станція потяг літак
корабель автомобіль карта план мета завдання відповідь питання причина приклад
правило порядок спосіб метод досвід наука знання пам ять думка ідея мрія надія
страх радість смуток любов дружба правда брехня таємниця загадка шифр код
повідомлення текст сенс значення ім я прізвище вік школа університет учитель
учень студент лікар лікарня здоров я хвороба ліки їжа хліб вода молоко м ясо
чай кава цукор сіль олія одяг взуття сорочка сукня пальто шапка сумка ціна
товар магазин ринок служба посада начальник робітник завод фабрика урожай
зерно вогонь дим попіл вітер дощ сніг лід мороз спека тепло холод весна літо
осінь зима сьогодні завтра вчора зараз скоро пізно рано завжди ніколи іноді
часто рідко багато мало більше менше краще гірше швидко повільно тихо гучно
близько далеко високо низько всередині зовні попереду ззаду ліворуч праворуч
тут там скрізь ніде куди звідки чому навіщо скільки який чий один два три
чотири п ять шість сім вісім дев ять десять сто тисяча другий третій останній
старий молодий малий довгий короткий широкий вузький високий низький білий
чорний червоний синій зелений жовтий сірий темний світлий теплий холодний
гарячий сухий мокрий чистий брудний повний порожній важкий легкий твердий
м який гострий живий мертвий сильний слабкий розумний добрий злий веселий
сумний спокійний страшний красивий простий складний важкий важливий головний
корисний небезпечний ходити бігти стояти сидіти лежати спати їсти пити дихати
дивитися бачити чути слухати говорити мовчати кричати шепотіти читати писати
рахувати думати пам ятати забути зрозуміти дізнатися шукати знайти втрачати
брати давати взяти класти ставити відкрити закрити почати закінчити чекати
зустріти піти прийти повернутися залишитися виїхати приїхати увійти вийти
любити ненавидіти хотіти могти повинен потрібно можна не можна треба варто
статися відбуватися здаватися виявитися ставати робити працювати вчитися
грати співати танцювати сміятися плакати боятися радіти вірити сподіватися
допомагати заважати захищати перемагати збирати ділити з єднувати ламати
будувати міняти залишати отримувати надсилати приносити показувати ховати
відкривати вирішувати перевіряти пробувати намагатися помилятися виправляти
пояснювати запитувати відповідати просити вимагати обіцяти виконувати привіт
дякую будь ласка вибачте здрастуйте прощай друг ворог людина люди народ країна
держава місто село сім я батьки діти син дочка брат сестра дід баба чоловік
дружина товариш сусід гість господар знайди мене хто ти допоможи рятуй біжи
дивись слухай мовчи чекай іди стій секрет правда таємниця щоденник запис
розділ початок кінець фінал відповідь розгадка червона квітка
"""

EN_CORPUS = """
One spring evening in an old town on the bank of a river people gathered to
listen to a story told by a stranger in a grey coat.
He spoke slowly and quietly but every word he said could be heard in the far
corner of the square because everyone was silent and no one dared interrupt.
Science begins where a person stops believing the obvious and starts checking
their guesses with precise measurements and repeatable experiments.
Every new question gives birth to ten new questions and in this lies the main
joy of the researcher who is never left without work and without hope.
The Vigenere cipher was long considered unbreakable because the same letter of
the plain text turned into different letters of the encrypted message.
However if the length of the key is known the problem falls apart into several
simple problems and each of them is solved by ordinary frequency analysis.
Short messages are harder to break because statistics on a few dozen letters
work badly and one has to rely on a dictionary and on meaning.
A person reads the text as a whole and immediately sees familiar words while a
machine counts the probabilities of letter combinations and picks the best one.
The morning of the next day turned out cold fog rose above the water and the
trees stood motionless as if waiting for something important and unknown.
The sea was calm and warm the waves ran onto the sand almost without a sound
and went back leaving behind a thin strip of foam and small shining stones.
The city lived its usual life cars drove along the wet asphalt people hurried
to work shops opened and closed and nobody thought about riddles.
But in the old house on the corner of the street the light burned until morning
because someone there was working on a problem he could not solve for weeks.
He wrote letters into a table rearranged them counted coincidences and started
again from the beginning because he believed the solution existed and was near.
Sometimes the hardest thing is to understand that you already hold the answer
in your hands but you are looking at it from the wrong side.
Knowledge does not come at once it is gathered slowly from small observations
random guesses and stubborn work that continues day after day.
A mistake is not a defeat but a source of information that costs far more than
a random success obtained without any understanding of the causes.
Time passes the same for everyone but each uses it in their own way and in the
end it turns out that some have a whole life and others only waiting.
Simple rules give rise to complex behaviour this is true of nature of society
and of the programs we write every day without thinking about it.
"""

EN_WORDS = """
the be to of and a in that have i it for not on with he as you do at this but
his by from they we say her she or an will my one all would there their what
so up out if about who get which go me when make can like time no just him
know take people into year your good some could them see other than then now
look only come its over think also back after use two how our work first well
way even new want because any these give day most us man find here thing tell
very still should through where much before life own too little world own old
right down between never under while start might place great again same both
country problem hand part high week point week company system program question
number night water room mother area money story fact month lot right study book
eye job word business issue side kind head house service friend father power
hour game line end member law car city community name president team minute
idea kid body information back parent face others level office door health
person art war history party result change morning reason research girl guy
moment air teacher force education foot boy age policy process music market
sense nation plan college interest death course someone experience behind
reach local kill six remain effect suggest class control raise care perhaps
little late hard field else pass former sell major sometimes require along
development whether police view together across during however home small
number sound water side place year work part place case week company group
number example letter alphabet cipher key secret message text meaning code
find me who are you help save run look listen wait go stop truth riddle diary
record chapter beginning end final answer solution red flower
"""


# --------------------------------------------------------------------------- #
#  Language definitions.
#  "encoding" is a single-byte encoding covering the alphabet: it lets the model
#  turn text into letter indices with one 256-entry table lookup instead of a
#  per-character Python loop. cp1251 covers Russian and Ukrainian, latin-1
#  covers English.
# --------------------------------------------------------------------------- #
LANGUAGES = {
    "en": {
        "name": {"en": "English", "ru": "Английский", "uk": "Англійська"},
        "encoding": "latin-1",
        "variants": {"26": {"letters": EN_26, "folds": {}}},
        "auto": ["26"],
        "corpus": EN_CORPUS,
        "words": EN_WORDS,
        "files": ("corpus_en.txt", "words_en.txt", "words_en_frequent.txt"),
    },
    "ru": {
        "name": {"en": "Russian", "ru": "Русский", "uk": "Російська"},
        "encoding": "cp1251",
        "variants": {
            "32": {"letters": RU_32, "folds": {"ё": "е"}},
            "33": {"letters": RU_33, "folds": {}},
        },
        "auto": ["32", "33"],
        "corpus": ru_data.CORPUS,
        "words": ru_data.WORDS,
        "files": ("corpus_ru.txt", "words_ru.txt", "words_ru_frequent.txt"),
    },
    "uk": {
        "name": {"en": "Ukrainian", "ru": "Украинский", "uk": "Українська"},
        "encoding": "cp1251",
        "variants": {
            "33": {"letters": UK_33, "folds": {}},
            "32": {"letters": UK_32, "folds": {"ґ": "г"}},
        },
        "auto": ["33", "32"],
        "corpus": UK_CORPUS,
        "words": UK_WORDS,
        "files": ("corpus_uk.txt", "words_uk.txt", "words_uk_frequent.txt"),
    },
}

DEFAULT_LANG = "en"


def lang_codes():
    return ["en", "ru", "uk"]


def get(code):
    return LANGUAGES[code if code in LANGUAGES else DEFAULT_LANG]


def lang_name(code, ui_lang="en"):
    return get(code)["name"].get(ui_lang, get(code)["name"]["en"])


def split_key(key: str):
    """``ru-32`` -> ``('ru', '32')``. A bare variant is treated as Russian."""
    if "-" in key:
        lang, variant = key.split("-", 1)
        if lang in LANGUAGES and variant in LANGUAGES[lang]["variants"]:
            return lang, variant
    if key in LANGUAGES["ru"]["variants"]:
        return "ru", key
    return DEFAULT_LANG, LANGUAGES[DEFAULT_LANG]["auto"][0]


def auto_keys(lang: str):
    """Every alphabet variant of a language; they are tested simultaneously."""
    return ["%s-%s" % (lang, v) for v in get(lang)["auto"]]


def variant_keys(lang: str):
    return ["%s-%s" % (lang, v) for v in get(lang)["variants"]]


def builtin_corpus(lang: str) -> str:
    d = get(lang)
    return d["corpus"] + "\n" + d["words"]


def builtin_words(lang: str) -> list:
    d = get(lang)
    letters = set(d["variants"][d["auto"][0]]["letters"])
    for v in d["variants"].values():
        letters |= set(v["letters"]) | set(v["folds"])
    out = {}
    for w in d["words"].split():
        w = w.strip().lower()
        if len(w) >= 2 and all(c in letters for c in w):
            out[w] = True
    return list(out)
