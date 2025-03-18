import os
import logging
from dotenv import load_dotenv
from bots.telegram_bot import TelegramBot

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def main():
    # Загрузка переменных окружения
    load_dotenv()

    # Проверка обязательных переменных
    if not os.getenv("TELEGRAM_TOKEN"):
        logger.error("Токен Telegram не найден в .env файле!")
        return

    # Инициализация и запуск бота
    try:
        logger.info("Запуск математического бота...")
        bot = TelegramBot()
        bot.run()
    except Exception as e:
        logger.error(f"Ошибка при работе бота: {str(e)}")


if __name__ == "__main__":
    main()