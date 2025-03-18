import os
from dotenv import load_dotenv
from argparse import ArgumentParser

# Загрузка переменных окружения
load_dotenv()

os.environ["PATH"] += os.pathsep + "C:/Users/Admin/Desktop/ffmpeg-2025-03-13-git-958c46800e-full_build/bin"

from speechkit import model_repository, configure_credentials, creds
from speechkit.stt import AudioProcessingType

configure_credentials(
    yandex_credentials=creds.YandexCredentials(
        api_key=os.getenv("YANDEX_API_KEY")
    )
)

def recognize(audio_path):
    try:
        # Проверяем существование файла
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Аудиофайл не найден: {audio_path}")

        # Настройка модели распознавания
        model = model_repository.recognition_model()
        model.model = 'general'  # Модель для общего распознавания
        model.language = 'ru-RU'  # Язык аудио
        model.audio_processing_type = AudioProcessingType.Full  # Полное распознавание

        # Запуск распознавания
        result = model.transcribe_file(audio_path)

        # Вывод результатов
        for channel, res in enumerate(result):
            print('=' * 80)
            print(f'Канал: {channel}')
            print(f'Исходный текст:\n{res.raw_text}\n')
            print(f'Нормализованный текст:\n{res.normalized_text}\n')

            if res.has_utterances():
                print('Фразы:')
                for utterance in res.utterances:
                    print(f'- {utterance}')

    except Exception as e:
        print(f"Ошибка: {str(e)}")
        exit(1)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--audio', type=str, required=True, help='Путь к аудиофайлу')
    args = parser.parse_args()

    recognize(args.audio)

