import spacy
import pymorphy2
import sqlite3
import os
from spacy_llm.util import assemble
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer

# Підключення до бази даних (якщо бази даних не існує, то вона буде автоматично створена)
conn = sqlite3.connect('store.db')
c = conn.cursor()

# Створення таблиці
c.execute('''CREATE TABLE IF NOT EXISTS stationery
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
             name TEXT,
             quantity INTEGER,
             price REAL)''')
conn.commit()

# Функція для додавання продуктів
def add_product(name, quantity, price):
    c.execute("INSERT INTO stationery (name, quantity, price) VALUES (?, ?, ?)", (name, quantity, price))
    conn.commit()
    print("Продукт додано успішно.")

# Функція для видалення продуктів за назвою
def delete_product(name):
    c.execute("DELETE FROM stationery WHERE name=?", (name,))
    conn.commit()
    print("Продукт видалено успішно.")

# Функція для виводу продуктів за запитом
def display_products_by_query(query):
    c.execute(query)
    stationery = c.fetchall()
    return stationery

# Функція для виводу ціни продукту за його назвою
def get_product_price(name):
    c.execute("SELECT price FROM stationery WHERE LOWER(name)=?", (name.lower(),))
    price = c.fetchone()
    if price:
        return price[0]
    else:
        return None

# Функція для виводу кількості продукту за його назвою
def get_product_quantity(name):
    c.execute("SELECT quantity FROM stationery WHERE LOWER(name)=?", (name.lower(),))
    quantity = c.fetchone()
    if quantity:
        return quantity[0]
    else:
        return None

def get_all_products():
    c.execute("SELECT name FROM stationery")
    products = c.fetchall()
    return products

# Відкриваємо файл прикладів
with open('example.txt', 'r', encoding='utf-8') as file:
    data = file.readlines()

# Розділяємо на вхідні й вихідні дані
X = [line.split(',', 1)[1].strip() for line in data]
y = [line.split(',')[0].strip() for line in data]

vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = vectorizer.fit_transform(X)
# Функція тренування моделі LightGBM
def train_LightGBM():
    gbm = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=2, min_child_samples=1, min_data_in_bin=1, random_state=40)
    gbm.fit(X_tfidf, y)

    return gbm

# Тренуємо модель
gbm = train_LightGBM()

# Функція для визначення наміру користувача
def test_model(gbm_model, user_input):
    user_input_tfidf = vectorizer.transform([user_input])
    gbm_pred = gbm_model.predict(user_input_tfidf)[0]

    return gbm_pred

basket = []

def normalized(text):
    morph = pymorphy2.MorphAnalyzer(lang='uk')
    words = text.split()
    normal_words = [morph.parse(word)[0].normal_form for word in words]
    normal_text = " ".join(normal_words)
    return normal_text


# Функція обробки користувацького вводу
def process_query(query, prev_product, prev_intent, prev_num):
    morph = pymorphy2.MorphAnalyzer(lang='uk')
    doc = nlp(query)
    word_to_num = {
        "один": 1, "одна": 1, "одне": 1, "одну": 1, "два": 2, "дві": 2, "три": 3, "чотири": 4, "п'ять": 5,
        "шість": 6, "сім": 7, "вісім": 8, "дев'ять": 9, "десять": 10, "одинадцять": 11, "дванадцять": 12,
        "тринадцять": 13, "чотирнадцять": 14, "п'ятнадцять": 15, "шістнадцять": 16, "сімнадцять": 17, "вісімнадцять": 18,
        "дев'ятнадцять": 19, "двадцять": 20
    }
    response = ""
    quantity = 0
    product1 = ""
    other_product1 = ""
    # [print(token.i, token.text, token.lemma_, token.pos_, token.dep_, token.ent_type_) for token in doc]

    intent = str(test_model(gbm, query)).strip('"') # використовуємо модель для розпізнавання наміру користувача
    for token in doc:
        if get_product_quantity(token.lemma_.lower()) is not None:
            product1 = token.lemma_.lower()
        elif token.dep_ in ["nsubj", "obj", "nmod", "ROOT"] and (token.pos_ == "NOUN"):
            other_product1 = token.lemma_.lower()
        elif token.pos_ == "NUM":
            if token.text.isdigit():
                quantity = int(token.text)
            else:
                quantity = word_to_num.get(token.text.lower(), "")
            prev_num = quantity

    intent = "наявність товарів" if (intent == "наявність" and (not product1 and not other_product1)) else intent

    if not intent:
        intent = prev_intent

    if not quantity:
        quantity = prev_num

    if intent == "купити товар":
        prev_intent = "купити товар"
        if not product1:
            if not other_product1:
                product1 = prev_product
            else:
                prev_product = other_product1
        else:
            prev_product = product1
        if product1:
            real_quantity = get_product_quantity(product1)
            if real_quantity > 0:
                product_parse = morph.parse(product1)
                if quantity > 4:
                    product_parse = product_parse[0].inflect({'gent', 'plur'}).word
                else:
                    product_parse = product_parse[0].inflect({'nomn', 'plur'}).word
                if quantity > real_quantity:
                    response = f"{quantity} {product_parse} недоступно для покупки. У нас є лише {real_quantity} {product_parse}."
                else:
                    if quantity != 1:
                        response = f"{quantity} {product_parse} додано до Вашого кошика."
                    else:
                        response = f"{product1} додано до Вашого кошика."
                    basket.append((product1, quantity))
                    new_quantity = get_product_quantity(product1) - quantity
                    c.execute("UPDATE stationery SET quantity=? WHERE name=?", (new_quantity, product1))
                    conn.commit()
            else:
                response = f"На жаль, {product1} зараз недоступний."
        else:
            response = "Я не зрозумів, який товар ви хочете купити."
    elif intent == "дізнатися ціну":
        prev_intent = "дізнатися ціну"
        if not product1:
            if not other_product1:
                product1 = prev_product
            else:
                prev_product = other_product1
        else:
            prev_product = product1
        if product1:
            price = get_product_price(product1)
            if quantity != 1:
                product_parse = morph.parse(product1)
                if quantity > 4:
                    product_parse = product_parse[0].inflect({'gent', 'plur'}).word
                else:
                    product_parse = product_parse[0].inflect({'nomn', 'plur'}).word
                response = f"Ціна на {quantity} {product_parse} - {price * quantity} грн."
            else:
                response = f"{product1} коштує {price} грн."
        else:
            response = "Я не зрозумів, ціну на який продукт Ви хочете дізнатися. Здається в нас немає такого товару"
    elif intent == "наявність":
        prev_intent = "наявність"
        if not product1:
            if not other_product1:
                product1 = prev_product
            else:
                prev_product = other_product1
        else:
            prev_product = product1
        if product1:
            product_parse = morph.parse(product1)
            product_parse = product_parse[0].inflect({'gent', 'plur'}).word
            response = f"На складі залишилось {get_product_quantity(product1)} одиниць {product_parse}."
            prev_product = product1
        else:
            response = "Вибачте, даного товару в нас немає."
    elif intent == "наявність товарів":
        prev_intent = "наявність товарів"
        response += f"На даний момент у нашому магазині є:"
        products_list = display_products_by_query("SELECT name, quantity FROM stationery")
        for product in products_list:
            response += f"\n{product[0]}: кількість - {product[1]}"
    elif intent == "вартість кошика":
        prev_intent = "вартість кошика"
        total_price = 0
        if not basket:
            response = f"На даний момент Ваш кошик порожній."
        else:
            response += f"На даний момент у Вашому кошику є:"
            for product in basket:
                prod, quantity = product
                product_price = get_product_price(prod)
                total_price += quantity * product_price
                if quantity > 1:
                    prod_parse = morph.parse(prod)
                    if quantity > 4:
                        prod = prod_parse[0].inflect({'gent', 'plur'}).word
                    else:
                        prod = prod_parse[0].inflect({'nomn', 'plur'}).word
                response += f"\n{quantity} {prod}; вартість за одиницю: {product_price} загальна вартість продукту: {product_price*quantity}"
            response += f"\nЗагальна вартість кошика становить - {total_price}"
    elif intent == "очистити кошик":
        prev_intent = "очистити кошик"
        for product in basket:
            prod, quantity = product
            new_quantity = get_product_quantity(prod) + quantity
            c.execute("UPDATE stationery SET quantity=? WHERE name=?", (new_quantity, prod))
            conn.commit()
        basket.clear()
        response = "Вміст кошика порожній."
    else:
        response = "Вибачте, я не розумію вашого запиту."

    return response, prev_product, prev_intent, prev_num

def process_saler_query(query):
    doc = nlp(query)
    intent = None

    for token in doc:
        if token.lemma_.lower() in ["додати"]:
            intent = "додати товар"
        elif token.lemma_.lower() in ["видалити"]:
            intent = "видалити товар"
        elif token.lemma_.lower() in ["бути", "існувати", "наявний", "наявність", "цікавити", "доступний", "показати", "вивести", "виведи"]:
            intent = "вивести товари"
        elif token.lemma_.lower() in ["вихід", "завершити", "закрити", "зупинити"]:
            intent = "вихід"
    return intent

def customer():
    prev_product = ""
    prev_intent = ""
    prev_num = 1
    print("Ви можете скористатися такими фразами:\n1. Вивести весь товар"
          "\n\t- Що у вас є в наявності?\n\t- Які товари у вас доступні?\n\t- Чи можна побачити весь товар?"
          "\n2. Дізнатися вартість товару\n\t- Скільки коштує 4 ручки?\n\t- Яка вартість олівця?\n3. Купити товар"
          "\n\t- Я хочу купити 5 зошитів.\n\t- Мені потрібна ручка.\n4. Побачити вміст кошика"
          "\n\t- Що я вже маю в корзині?\n\t- Яка загальна вартість мого кошика?"
          "\n5. Очистити кошик\n\t- Я хочу видалити вміст кошика\n\t- Очисти мою корзину\n6. Вихід - для завершення роботи")
    user_input = input("Введіть ваш запит: ")
    while user_input != "вихід":
        response, prev_product, prev_intent, prev_num = process_query(user_input, prev_product, prev_intent, prev_num)
        print("Відповідь асистента:", response)
        user_input = input("Введіть ваш запит: ")

def saler():
    print("Ви можете скористатися такими фразами:\n1. Вивести товар\n2. Додати товар\n3. Видалити товар\n4. Вихід - для завершення роботи")
    while True:
        user_input = input("Введіть ваш запит: ")
        processed_input = process_saler_query(user_input)
        if processed_input == "додати товар":
            product = input("Введіть назву товару, який Ви хочете додати: ")
            try:
                number = int(input("Введіть кількість даного товару: "))
                if number >= 0:
                    try:
                        price = float(input("Введіть вартість даного товару: "))
                        if price > 0:
                            add_product(normalized(product.lower()), number, price)
                        else:
                            print("Вартість має бути більшою за 0.")
                    except ValueError:
                        print("Введене значення має бути додатнім числом.")
                else:
                    print("Кількість не може бути від'ємною.")
            except ValueError:
                print("Введене значення має бути додатнім цілим числом.")

        elif processed_input == "видалити товар":
            all_products = get_all_products()
            print("Список всіх товарів:")
            for product in all_products:
                print(product[0])
            product = input("Введіть назву товару, який Ви хочете видалити: ")
            delete_product(product)
        elif processed_input == "вивести товари":
            products_list = display_products_by_query("SELECT name, quantity, price FROM stationery")
            print("Список всіх товарів:")
            for product in products_list:
                print(f"{product[0]}: кількість - {product[1]}, вартість за одиницю - {product[2]}")
        elif processed_input == "вихід":
            print("Повернення до головного меню..")
            break
        else:
            print("Я не розумію ваш запит.")


def main():
    while True:
        user_input1 = input("Ким Ви представляєтесь (покупець/продавець)?: ")

        if user_input1.lower() == "вихід":
            print("Завершення роботи...")
            break
        elif user_input1.lower() == "покупець":
            customer()
        elif user_input1.lower() == "продавець":
            saler()
        else:
            print("Не вірно введене значення, будь ласка, зазначне правильно ким Ви є")


# nlp = spacy.load("uk_core_news_lg")
nlp = assemble(config_path="config.cfg")

main()

# Закриваємо з'єднання з базою даних
conn.close()

