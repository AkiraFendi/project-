import os
import argparse
from dotenv import load_dotenv
from speechkit import model_repository, configure_credentials, creds
from speechkit.stt import AudioProcessingType

# Загрузка переменных окружения
load_dotenv()

# Настройка SpeechKit
configure_credentials(
    yandex_credentials=creds.YandexCredentials(
        api_key=os.getenv("YANDEX_API_KEY")
    )
)

model = model_repository.recognition_model()
model.model = 'general'
model.language = 'ru-RU'
model.audio_processing_type = AudioProcessingType.Full

# Функция для нормализации текста
MATH_REPLACEMENTS = {
    "икс": "x",
    "икса": "x",
    "игрек": "y",
    "зет": "z",
    "плюс": "+",
    "минус": "-",
    "умножить на": "*",
    "разделить на": "/",
    "равно": "=",
    "скобка открывается": "(",
    "скобка закрывается": ")",
    "в степени": "**",
    "квадрате": "**2",
    "кубе": "**3",
    "корень": "sqrt",
    "пи": "pi",
    "синус": "sin",
    "косинус": "cos",
    "синус": "sin(",
    "косинус": "cos(",
    "тангенс": "tan(",
    "предел": "lim ",
    " стремится к ": "->",
    "приближается к": "->"
}


def normalize_math_text(text: str) -> str:
    text = text.lower()
    for word, symbol in MATH_REPLACEMENTS.items():
        text = text.replace(word, symbol)
    return text.strip()


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
