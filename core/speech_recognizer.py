import os
import re
import argparse
from dotenv import load_dotenv
from speechkit import model_repository, configure_credentials, creds
from speechkit.stt import AudioProcessingType

load_dotenv()

configure_credentials(
    yandex_credentials=creds.YandexCredentials(
        api_key=os.getenv("YANDEX_API_KEY")
    )
)

model = model_repository.recognition_model()
model.model = 'general'
model.language = 'ru-RU'
model.audio_processing_type = AudioProcessingType.Full

MATH_REPLACEMENTS = {
    # Удаление служебных слов
    "вычисли": "", "посчитай": "", "найди": "", "реши": "",
    "интеграл": "", "производн": "", "предел": "", "дифференциал": "",
    "по dx": "", "по dy": "", "по": "", "от": ",", "до": ",",

    # Базовые операции
    "плюс": "+", "минус": "-", 
    "умноженное на": "*", "умножить на": "*", "×": "*",
    "деленное на": "/", "разделить на": "/", "÷": "/",
    "равно": "=", "равняется": "=", "=": "=",
    "скобка": "(", "открывается": "(", "закрывается": ")",

    # Степени и корни
    "в степени": "**", "степень": "**", 
    "квадрат": "**2", "квадрате": "**2", "в квадрате": "**2",
    "куб": "**3", "в кубе": "**3", 
    "корень квадратный": "sqrt", "корень": "sqrt",
    "√": "sqrt", "√(": "sqrt(",

    # Константы
    "пи": "pi", "число е": "e", "экспонента": "e",

    # Функции
    "синус": "sin(", "косинус": "cos(", "тангенс": "tan(",
    "арксинус": "asin(", "арккосинус": "acos(", "арктангенс": "atan(",
    "логарифм": "log(", "натуральный логарифм": "ln(", "экспонента": "exp(",

    # Переменные
    "икс": "x", "игрек": "y", "зет": "z", 
    "альфа": "alpha", "бета": "beta", "гамма": "gamma",

    # Специальные символы
    "стремится к": "->", "приближается к": "->", "бесконечность": "oo",
    "сумма": "Sum", "произведение": "Product", "модуль": "abs",

    # Фикс пробелов
    "  ": " ", "\t": " ", "   ": " ",
}

def normalize_math_text(text: str) -> str:
    text = re.sub(r"∫|\\int", "", text)  # Удаление символа интеграла
    
    # Экранирование специальных символов в шаблонах
    escaped_replacements = [
        (re.escape(pattern), replacement)
        for pattern, replacement in MATH_REPLACEMENTS.items()
    ]
    
    for pattern, replacement in escaped_replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    text = re.sub(r"\s+", " ", text).strip()
    return text

def recognize(audio_path: str) -> str:
    try:
        result = model.transcribe_file(audio_path)

        normalized_text = ""
        for res in result:
            if res.normalized_text:
                normalized_text = res.normalized_text
                break  # Берем первый распознанный текст

        if not normalized_text:
            print("Ошибка: не удалось распознать речь")
            return ""

        normalized_text = normalize_math_text(normalized_text)
        print(f"Распознанный текст: {normalized_text}")
        return normalized_text

    except Exception as e:
        print(f"Ошибка: {str(e)}")
        return ""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', type=str, required=True, help='Путь к аудиофайлу')
    args = parser.parse_args()
    result = recognize(args.audio)
    if result:
        print(f"Финальный текст: {result}")