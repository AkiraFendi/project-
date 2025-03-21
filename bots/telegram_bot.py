import os
import logging
import tempfile
import subprocess
import sys
import time
import sqlite3
import asyncio
from core.speech_recognizer import normalize_math_text
from telegram import Update, User, Message
from sqlite3 import Error as SqliteError
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Message
from telegram.helpers import escape_markdown
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes
)
from telegram.error import BadRequest
from speechkit import model_repository, configure_credentials, creds
from speechkit.stt import AudioProcessingType

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.math_rag import MathRAG
from core.safe_executor import SafeExecutor

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

LOCALES = {
    "ru": {
        "start": """
        🎉 Добро пожаловать, {name}\\!

        🧮\\* Добро пожаловать в Math Solver Bot\\! \\*🚀

        Я помогу вам\\:
        ✅ Решать математические примеры
        ✅ Объяснять шаги решения
        ✅ Генерировать исполняемый код

        \\*Примеры запросов\\*\\:
        `2 \\+ 2 \\* 2`
        `интеграл от x\\^2`
        `решить уравнение 3x \\+ 5 = 20`

        🛠 \\*Доступные команды\\*\\:
        /start \\- начать работу
        /help \\- справка
        /history \\- история запросов
        /settings \\- настройки
        """,
        "help": """
        📖 \\*Справка по использованию бота\\*

        Просто отправьте мне\\:
        \\- Любое математическое выражение
        \\- Уравнение для решения
        \\- Запрос на вычисление интеграла/производной
        \\- Голосовое сообщение с задачей

        \\*Примеры\\*\\:
        `sin\\(pi/2\\) \\+ cos\\(0\\)`
        `lim x\\->0 \\(sin\\(x\\)/x\\)`
        `derivative of x\\^3 \\+ 2x`

        ⚠️ \\*Ограничения\\*\\:
        \\- Поддерживается LaTeX\\-нотация
        \\- Максимальная длина запроса\\: 500 символов
        """,
        "settings_text": "⚙️ \\*Настройки\\*\\:",
        "notif_on": "Уведомления: Вкл",
        "notif_off": "Уведомления: Выкл",
        "history_empty": "📜 История запросов пуста",
        "history_header": "📚 Последние 10 запросов",
        "history_query": "Запрос",
        "history_response": "Ответ",
        "error_query_too_long": "❌ Превышена максимальная длина запроса (500 символов)",
        "internal_error": "⚠️ Произошла внутренняя ошибка. Попробуйте позже.",
        "voice_too_long": "⚠️ Голосовое сообщение должно быть короче 30 секунд",
        "cooldown": "⏳ Слишком частые запросы. Подождите 2 секунды.",
        "language_changed": "🌍 Язык изменен на {lang}",
        "notifications_changed": "🔔 Уведомления: {status}",
        "voice_recognition_error": "❌ Не удалось распознать речь",
        "audio_conversion_error": "Ошибка конвертации аудио",
        "error_prefix": "Ошибка",
        "steps": "Пошаговое решение",
        "result": "Результат",
        "code": "Код",
        "empty_message_error": "Пожалуйста, отправьте текстовое или голосовое сообщение с задачей",
    },
    "en": {
        "start": """
        🎉 Welcome, {name}\\!

        🧮\\* Welcome to Math Solver Bot\\! \\*🚀

        I can help you\\:
        ✅ Solve math problems
        ✅ Explain solution steps
        ✅ Generate executable code

        \\*Examples\\*\\:
        `2 \\+ 2 \\* 2`
        `integrate x\\^2`
        `solve equation 3x \\+ 5 = 20`

        🛠 \\*Available commands\\*\\:
        /start \\- Start bot
        /help \\- Help
        /history \\- Query history
        /settings \\- Settings
        """,
        "help": """
        📖 \\*Usage Help\\*

        Just send me\\:
        \\- Any math expression
        \\- Equation to solve
        \\- Integral/derivative request
        \\- Voice message with problem

        \\*Examples\\*\\:
        `sin\\(pi/2\\) \\+ cos\\(0\\)`
        `lim x\\->0 \\(sin\\(x\\)/x\\)`
        `derivative of x\\^3 \\+ 2x`

        ⚠️ \\*Limitations\\*\\:
        \\- Supports LaTeX notation
        \\- Max query length\\: 500 characters
        """,
        "settings_text": "⚙️ \\*Settings\\*\\:",
        "notif_on": "Notifications: On",
        "notif_off": "Notifications: Off",
        "history_empty": "📜 Query history is empty",
        "history_header": "📚 Last 10 queries",
        "history_query": "Query",
        "history_response": "Response",
        "error_query_too_long": "❌ Query too long (max 500 characters)",
        "internal_error": "⚠️ Internal error. Please try again later.",
        "voice_too_long": "⚠️ Voice message must be shorter than 30 seconds",
        "cooldown": "⏳ Too many requests. Please wait 2 seconds.",
        "language_changed": "🌍 Language changed to {lang}",
        "notifications_changed": "🔔 Notifications: {status}",
        "voice_recognition_error": "❌ Speech recognition failed",
        "audio_conversion_error": "Audio conversion error",
        "error_prefix": "Error",
        "solution_steps": "Step-by-Step Solution",
        "result": "Result",
        "code": "Code",
        "empty_message_error": "Please send a text or voice message with your problem",
    }
}


class TelegramBot:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_TOKEN")
        self.yandex_key = os.getenv("YANDEX_API_KEY")

        if not all([self.token, self.yandex_key]):
            missing = []
            if not self.token: missing.append("TELEGRAM_TOKEN")
            if not self.yandex_key: missing.append("YANDEX_API_KEY")
            raise ValueError(f"Missing required env vars: {', '.join(missing)}")

        self.rag = MathRAG()
        self.executor = SafeExecutor()
        self.user_cooldown = {}
        self._init_db()
        self._configure_speechkit()
        logger.info("Bot initialized")

    def _init_db(self):
        self.conn = sqlite3.connect('math_bot.db', check_same_thread=False)
        cursor = self.conn.cursor()

        try:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    last_name TEXT,
                    registered_at DATETIME,
                    language TEXT DEFAULT 'ru',
                    notifications INTEGER DEFAULT 1 CHECK (notifications IN (0, 1))
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    query TEXT,
                    response TEXT,
                    created_at DATETIME,
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                )
            ''')

            self.conn.commit()
            logger.info("Database tables created successfully")

        except sqlite3.Error as e:
            logger.critical(f"Database initialization failed: {str(e)}")
            raise RuntimeError(f"Database error: {str(e)}") from e

    async def _register_user(self, user):
        if self._user_exists(user.id):
            return

        cursor = None
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO users 
                (user_id, username, first_name, last_name, registered_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                user.id,
                user.username or '',
                user.first_name or '',
                user.last_name or '',
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Registration error: {str(e)}")
            self.conn.rollback()
        finally:
            if cursor:
                cursor.close()

    def _check_notifications(self, user_id: int) -> bool:
        try:
            cursor = self.conn.cursor()
            cursor.execute('SELECT notifications FROM users WHERE user_id = ?', (user_id,))
            result = cursor.fetchone()
            return bool(result[0]) if result else True
        except Exception as e:
            logger.error(f"Notification check error: {str(e)}")
            return True
        finally:
            if cursor:
                cursor.close()

    def _get_text(self, user_id: int, key: str, **kwargs) -> str:
        try:
            cursor = self.conn.cursor()
            cursor.execute('SELECT language FROM users WHERE user_id = ?', (user_id,))
            result = cursor.fetchone()
            lang = result[0] if result else 'ru'
            text = LOCALES[lang].get(key, key)
            return text.format(**kwargs) if kwargs else text
        except sqlite3.Error as e:
            logger.error(f"Database error in _get_text: {str(e)}")
            return LOCALES['ru'].get(key, key)
        finally:
            if cursor:
                cursor.close()

    def _user_exists(self, user_id: int) -> bool:
        cursor = None
        try:
            cursor = self.conn.cursor()
            cursor.execute('SELECT 1 FROM users WHERE user_id = ?', (user_id,))
            return cursor.fetchone() is not None
        except sqlite3.Error as e:
            logger.error(f"User check error: {str(e)}")
            return False
        finally:
            if cursor:
                cursor.close()

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        await self._register_user(user)
        text = self._get_text(user.id, "start", name=escape_markdown(user.first_name, version=2))
        await update.message.reply_markdown_v2(text)

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        text = self._get_text(user.id, "help")
        await update.message.reply_markdown_v2(text)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        message = update.message

        try:
            if await self.check_cooldown(update, context):
                return

            await self._register_user(user)

            # Проверяем статус уведомлений
            if not self._check_notifications(user.id):
                logger.info(f"Notifications disabled for user {user.id}, ignoring message")
                return

                # Обработка голосовых сообщений
            if message.voice:
                await self.handle_voice(message, user)
                return

            # Получаем текст сообщения
            query = message.text.strip()
            lang = self._get_user_language(user.id)

            # Логируем полученный запрос
            logger.info(f"Received query: {query} from user {user.id}")

            result = self.rag.solve_problem(query, lang=lang)

            if "error" in result:
                error_msg = f"❌ {self._get_text(user.id, 'error_prefix')}: {result['error']}"
                await message.reply_text(error_msg)
                return

            response = self._format_response(user.id, result, lang)
            await message.reply_markdown_v2(response)

            self._log_response(user.id, query, response)

        except Exception as e:
            logger.error(f"Message handling error: {str(e)}", exc_info=True)
            await message.reply_text(self._get_text(user.id, "internal_error"))

    def _format_response(self, user_id: int, result: dict, lang: str) -> str:
        def escape(text: str) -> str:
            # Экранируем всё, кроме LaTeX-формул
            return escape_markdown(text, version=2).replace(r"\(", "$").replace(r"\)", "$")

        parts = []

        if result.get("formatted_steps"):
            steps = "\n".join([
                f"• {escape(step)}"
                for step in result["formatted_steps"].split("\n")
            ])
            parts.append(f"📝 *{escape('Пошаговое решение')}:*\n{steps}")

        if result.get("result"):
            parts.append(f"✅ *{escape('Результат')}:* `{escape(result['result'])}`")

        if result.get("code"):
            code = result['code'].replace('`', '`\u200b')  # Экранируем backticks
            parts.append(f"💻 *{escape('Код')}:*\n```python\n{code}\n```")

        return "\n\n".join(parts)

    def _get_user_language(self, user_id: int) -> str:
        cursor = self.conn.cursor()
        cursor.execute('SELECT language FROM users WHERE user_id = ?', (user_id,))
        result = cursor.fetchone()
        cursor.close()  # Добавить закрытие курсора
        return result[0] if result else 'ru'

    async def history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT query, response, created_at 
            FROM history 
            WHERE user_id = ?
            ORDER BY created_at DESC 
            LIMIT 10
        ''', (user.id,))

        history_records = cursor.fetchall()

        if not history_records:
            await update.message.reply_text(self._get_text(user.id, "history_empty"))
            return

        response = []
        for idx, (query, resp, date) in enumerate(history_records, 1):
            try:
                date_str = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').strftime('%d\\.%m %H\\:%M')
                escaped_query = escape_markdown(query[:50], version=2)
                escaped_response = escape_markdown(resp[:100], version=2) if resp else ""

                item = [
                    f"{escape_markdown(str(idx), version=2)}\\.",
                    f"▶️ *{self._get_text(user.id, 'history_query')}*: `{escaped_query}`",
                    f"◀️ *{self._get_text(user.id, 'history_response')}*: `{escaped_response}`" if resp else ""
                ]
                response.append("\n".join(filter(None, item)))

            except Exception as e:
                logger.error(f"Error formatting history item: {str(e)}")

        header = self._get_text(user.id, "history_header")
        try:
            await update.message.reply_markdown_v2(
                f"{escape_markdown(header, version=2)}:\n\n" +
                "\n\n".join(response)
            )
        except BadRequest as e:
            logger.error(f"Markdown error: {str(e)}")
            await update.message.reply_text(
                "📜 History:\n" +
                "\n".join([item.replace('*', '').replace('_', '') for item in response])
            )

    def _log_query(self, user_id: int, query: str):
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO history 
                (user_id, query, created_at)
                VALUES (?, ?, ?)
            ''', (user_id, query[:500], datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error logging query: {str(e)}")

    def _log_response(self, user_id: int, query: str, response: str):
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                UPDATE history 
                SET response = ?
                WHERE user_id = ? 
                AND id = (
                    SELECT id 
                    FROM history 
                    WHERE user_id = ? 
                    ORDER BY created_at DESC 
                    LIMIT 1
                )
            ''', (response, user_id, user_id))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error logging response: {str(e)}")

    async def settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        cursor = self.conn.cursor()
        cursor.execute('SELECT language, notifications FROM users WHERE user_id = ?', (user.id,))
        result = cursor.fetchone()

        lang, notif = result if result else ('ru', 1)

        keyboard = [
            [
                InlineKeyboardButton(
                    f"🇷🇺 Русский {'✅' if lang == 'ru' else ''}",
                    callback_data="lang_ru"
                ),
                InlineKeyboardButton(
                    f"🇺🇸 English {'✅' if lang == 'en' else ''}",
                    callback_data="lang_en"
                )
            ],
            [
                InlineKeyboardButton(
                    f"🔔 {self._get_text(user.id, 'notif_on')} {'✅' if notif else ''}",
                    callback_data="notif_on"
                ),
                InlineKeyboardButton(
                    f"🔕 {self._get_text(user.id, 'notif_off')} {'✅' if not notif else ''}",
                    callback_data="notif_off"
                )
            ]
        ]
        text = self._get_text(user.id, "settings_text")
        await update.message.reply_markdown_v2(text, reply_markup=InlineKeyboardMarkup(keyboard))

    async def check_cooldown(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
        user_id = update.effective_user.id
        now = time.time()

        if user_id in self.user_cooldown:
            elapsed = now - self.user_cooldown[user_id]
            if elapsed < 2:
                await update.message.reply_text(self._get_text(user_id, "cooldown"))
                return True
        self.user_cooldown[user_id] = now
        return False

    def _configure_speechkit(self):
        configure_credentials(
            yandex_credentials=creds.YandexCredentials(
                api_key=os.getenv("YANDEX_API_KEY")
            )
        )
        self.speech_model = model_repository.recognition_model()
        self.speech_model.model = 'general'
        self.speech_model.language = 'ru-RU'
        self.speech_model.audio_processing_type = AudioProcessingType.Full

    async def handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            user = update.effective_user
            message = update.effective_message
            logger.info(f"Получено голосовое сообщение от {user.id}")

            # Добавляем логирование длительности
            duration = message.voice.duration
            logger.info(f"Длительность сообщения: {duration} сек")

            # Проверяем статус уведомлений
            if not self._check_notifications(user.id):
                logger.info(f"Ignoring voice message from {user.id} (notifications off)")
                return

            if message.voice.duration > 30:
                await message.reply_text(self._get_text(user.id, "voice_too_long"))
                return

            voice = await message.voice.get_file()
            tmp_path = tempfile.mktemp(suffix=".ogg")
            converted_path = None

            try:
                await voice.download_to_drive(tmp_path)
                converted_path = await self._convert_audio(tmp_path)
                recognized_text = self._recognize_voice(converted_path)
                if recognized_text:
                    normalized_text = normalize_math_text(recognized_text)
                    await self._process_text_message(
                        chat_id=message.chat_id,
                        text=normalized_text,  # Используем нормализованный текст
                        user=user,
                        original_message=message
                    )

            except Exception as e:
                logger.error(f"Voice processing error: {str(e)}")
                await message.reply_text(self._get_text(user.id, "internal_error"))

            finally:
                for path in [tmp_path, converted_path]:
                    if path and os.path.exists(path):
                        os.remove(path)

        except Exception as e:
            logger.error(f"Critical voice error: {str(e)}")
            if message:
                await message.reply_text(self._get_text(user.id, "internal_error"))

    async def _process_text_message(self, chat_id: int, text: str, user: User, original_message: Message):
        try:
            self._log_query(user.id, text)
            logger.info(f"Processing text: {text}")

            # Получаем язык пользователя
            lang = self._get_user_language(user.id)  # <-- Добавлено получение языка

            # Передаем язык в solve_problem
            result = self.rag.solve_problem(text, lang=lang)

            # Добавляем lang в вызов _format_response
            response = self._format_response(user.id, result, lang)  # <-- Исправлено здесь
            await original_message.reply_markdown_v2(response)
            self._log_response(user.id, text, response)
        except Exception as e:
            logger.error(f"Text processing error: {str(e)}")
            await original_message.reply_text(self._get_text(user.id, "internal_error"))

    async def _convert_audio(self, input_path: str) -> str:
        output_path = input_path + ".wav"
        command = [
            "ffmpeg",
            "-i", input_path,
            "-ar", "16000",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-y", output_path
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"FFmpeg error: {stderr.decode()}")
                raise RuntimeError(self._get_text(None, "audio_conversion_error"))

            return output_path

        except Exception as e:
            logger.error(f"Audio conversion failed: {str(e)}")
            raise

    def _recognize_voice(self, file_path: str) -> str:
        try:
            result = self.speech_model.transcribe_file(file_path)
            text = ' '.join([res.normalized_text for res in result if res.normalized_text])

            # Очищаем текст от не-ASCII символов
            cleaned_text = text.encode('utf-8', 'ignore').decode('utf-8')
            return cleaned_text.strip()
        except Exception as e:
            logger.error(f"Recognition error: {str(e)}")
            return None

    async def settings_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        user_id = query.from_user.id
        data = query.data

        try:
            cursor = self.conn.cursor()

            if data.startswith("lang_"):
                new_lang = data.split("_")[1]
                cursor.execute('''
                    UPDATE users 
                    SET language = ?
                    WHERE user_id = ?
                ''', (new_lang, user_id))
                self.conn.commit()
                cursor.execute('SELECT language, notifications FROM users WHERE user_id = ?', (user_id,))
                lang, notif = cursor.fetchone()

            elif data.startswith("notif_"):
                # Инвертируем текущее значение
                cursor.execute('''
                    UPDATE users 
                    SET notifications = NOT notifications 
                    WHERE user_id = ?
                ''', (user_id,))
                self.conn.commit()
                cursor.execute('SELECT language, notifications FROM users WHERE user_id = ?', (user_id,))
                lang, notif = cursor.fetchone()

                # Логируем изменение
                status = "on" if notif else "off"
                logger.info(f"User {user_id} notifications: {status}")

            # Обновляем интерфейс
            keyboard = [
                [
                    InlineKeyboardButton(
                        f"🇷🇺 Русский {'✅' if lang == 'ru' else ''}",
                        callback_data="lang_ru"
                    ),
                    InlineKeyboardButton(
                        f"🇺🇸 English {'✅' if lang == 'en' else ''}",
                        callback_data="lang_en"
                    )
                ],
                [
                    InlineKeyboardButton(
                        f"🔔 {self._get_text(user_id, 'notif_on')} {'✅' if notif else ''}",
                        callback_data="notif_on"
                    ),
                    InlineKeyboardButton(
                        f"🔕 {self._get_text(user_id, 'notif_off')} {'✅' if not notif else ''}",
                        callback_data="notif_off"
                    )
                ]
            ]

            text = self._get_text(user_id, "settings_text")
            await query.edit_message_text(
                text=text,
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="MarkdownV2"
            )

            # Отправляем подтверждение
            if data.startswith("lang_"):
                await query.answer(self._get_text(user_id, "language_changed", lang=lang.upper()))
            elif data.startswith("notif_"):
                status_text = self._get_text(user_id, 'notif_on') if notif else self._get_text(user_id, 'notif_off')
                await query.answer(self._get_text(user_id, "notifications_changed", status=status_text))
                if not notif:
                    warning = self._get_text(user_id, "notif_off_warning")
                    await query.message.reply_text(warning)

        except Exception as e:
            logger.error(f"Settings handler error: {str(e)}")
            await query.edit_message_text(self._get_text(user_id, "internal_error"))
        finally:
            cursor.close()

    def run(self):
        try:
            logger.info("Starting bot...")
            app = ApplicationBuilder().token(self.token).build()

            handlers = [
                CommandHandler("start", self.start),
                CommandHandler("help", self.help),
                CommandHandler("history", self.history),
                CommandHandler("settings", self.settings),
                MessageHandler(filters.VOICE, self.handle_voice),
                MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message),
                CallbackQueryHandler(self.settings_handler)
            ]

            for handler in handlers:
                app.add_handler(handler)

            logger.info("Bot running in polling mode...")
            app.run_polling(
                drop_pending_updates=True,
                poll_interval=0.5,
                allowed_updates=Update.ALL_TYPES
            )

        except Exception as e:
            logger.critical(f"Critical error: {str(e)}")
        finally:
            self.conn.close()


if __name__ == "__main__":
    try:
        logger.info("=== STARTING SYSTEM ===")
        bot = TelegramBot()
        bot.run()
    except Exception as e:
        logger.critical(f"FATAL ERROR: {str(e)}")
    finally:
        logger.info("=== SHUTTING DOWN ===")