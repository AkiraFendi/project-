import os
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)
try:
    from core.math_rag import MathRAG
except ImportError:
    from ..core.math_rag import MathRAG
from core.safe_executor import SafeExecutor

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class TelegramBot:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_TOKEN")
        if not self.token:
            raise ValueError("TELEGRAM_TOKEN не найден в .env!")

        self.rag = MathRAG()
        self.executor = SafeExecutor()
        logger.info("Инициализация бота завершена")

    # Остальные методы без изменений...

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        logger.info(f"Получена команда /start от {update.effective_user.id}")
        welcome_text = """
        🧮 *Добро пожаловать в Math Solver Bot!* 🚀

Я помогу вам:
✅ Решать математические примеры
✅ Объяснять шаги решения
✅ Генерировать исполняемый код

*Примеры запросов:*
`2 + 2 * 2`
`интеграл от x^2`
`решить уравнение 3x + 5 = 20`

🛠 *Доступные команды:*
/start - начать работу
/help - справка
        """
        await update.message.reply_markdown_v2(welcome_text)

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help"""
        logger.info(f"Получена команда /help от {update.effective_user.id}")
        help_text = """
        📖 *Справка по использованию бота*

Просто отправьте мне:
- Любое математическое выражение
- Уравнение для решения
- Запрос на вычисление интеграла/производной

*Примеры:*
sin(pi/2) + cos(0)
lim x->0 (sin(x)/x)
derivative of x^3 + 2x

⚠️ *Ограничения:*
- Поддерживается только LaTeX-нотация
- Сложные запросы могут требовать времени
        """
        await update.message.reply_markdown_v2(help_text)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            user = update.effective_user
            logger.info(f"Сообщение от {user.id}: {update.message.text}")

            result = self.rag.solve_problem(update.message.text)

            if "error" in result:
                await update.message.reply_text(f"❌ Ошибка: {result['error']}")
                return

            response = self._format_response(result)
            await update.message.reply_markdown_v2(response)

        except Exception as e:
            logger.error(f"Ошибка: {str(e)}", exc_info=True)
            await update.message.reply_text("⚠️ Произошла внутренняя ошибка. Попробуйте позже.")

    def _format_response(self, result: dict) -> str:
        """Форматирование ответа с экранированием Markdown"""

        def escape(text: str) -> str:
            special_chars = {
                '_': r'\_', '*': r'\*', '[': r'\[', ']': r'\]',
                '(': r'\(', ')': r'\)', '~': r'\~', '`': r'\`',
                '>': r'\>', '#': r'\#', '+': r'\+', '-': r'\-',
                '=': r'\=', '|': r'\|', '{': r'\{', '}': r'\}',
                '.': r'\.', '!': r'\!'
            }
            return text.translate(str.maketrans(special_chars))

        if "error" in result:
            return f"❌ *Ошибка:*\n`{escape(result['error'])}`"

        parts = []
        if result.get("formatted_steps"):
            parts.append(f"📝 *Решение:*\n`{escape(result['formatted_steps'])}`")
        if result.get("result"):
            parts.append(f"✅ *Результат:* `{escape(result['result'])}`")
        if result.get("code"):
            parts.append(f"💻 *Код:*\n```python\n{result['code']}\n```")

        return "\n\n".join(parts)

    def run(self):
        try:
            logger.info("Инициализация бота...")
            app = ApplicationBuilder() \
                .token(self.token) \
                .build()

            # Регистрируем обработчики команд ПЕРВЫМИ
            app.add_handler(CommandHandler("start", self.start))
            app.add_handler(CommandHandler("help", self.help))

            # Затем обработчик обычных сообщений
            app.add_handler(MessageHandler(
                filters.TEXT & ~filters.COMMAND,  # Игнорируем команды
                self.handle_message
            ))

            logger.info("Бот запущен в режиме polling...")
            app.run_polling(
                drop_pending_updates=True,  # Игнорируем сообщения, отправленные до старта бота
                poll_interval=0.5
            )

        except Exception as e:
            logger.critical(f"Критическая ошибка: {str(e)}")
            raise


if __name__ == "__main__":
    try:
        logger.info("=== ЗАПУСК СИСТЕМЫ ===")
        bot = TelegramBot()
        bot.run()
    except Exception as e:
        logger.critical(f"ФАТАЛЬНАЯ ОШИБКА: {str(e)}", exc_info=True)
    finally:
        logger.info("=== ЗАВЕРШЕНИЕ РАБОТЫ ===")
