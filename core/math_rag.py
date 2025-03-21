import os
import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from sympy import (
    sympify, SympifyError, symbols, Eq, solve, simplify,
    diff, integrate, Derivative, Integral, latex, S, Equality
)
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication
)
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from gigachat import GigaChat
from telegram import Update, Message, User
from core.safe_executor import SafeExecutor, SecurityError
from dotenv import load_dotenv
load_dotenv()


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

transformations = standard_transformations + (implicit_multiplication,)


class MathRAG:
    LOCALES = {
        "ru": {
            "equation": {
                "original": "Исходное уравнение",
                "simplified": "Упрощенная форма",
                "solution": "Решение"
            },
            "derivative": {
                "function": "Исходная функция",
                "derivative": "Производная"
            },
            "integral": {
                "function": "Исходная функция",
                "integral": "Интеграл"
            },
            "result": "Результат",
            "code": "Код",
            "steps": "Пошаговое решение",
            "errors": {
                "empty_query": "Пустой запрос",
                "security": "Ошибка безопасности: {error}",
                "timeout": "Превышено время выполнения",
                "parse": "Ошибка парсинга: {error}",
                "no_solution": "Нет решения",
                "general": "Ошибка решения"
            }
        },
        "en": {
            "equation": {
                "original": "Original equation",
                "simplified": "Simplified form",
                "solution": "Solution"
            },
            "derivative": {
                "function": "Original function",
                "derivative": "Derivative"
            },
            "integral": {
                "function": "Original function",
                "integral": "Integral"
            },
            "result": "Result",
            "code": "Code",
            "steps": "Step-by-Step Solution",
            "errors": {
                "empty_query": "Empty query",
                "security": "Security error: {error}",
                "timeout": "Execution timeout",
                "parse": "Parsing error: {error}",
                "no_solution": "No solution",
                "general": "Solution error"
            }
        }
    }

    def __init__(self, db_path: str = "data/math_db", max_query_length: int = 500):
        self.db_path = db_path
        self.max_query_length = max_query_length
        self.embeddings = None
        self.vector_db = None
        self.llm = None
        self.executor = SafeExecutor()
        self.x = symbols('x')
        self._init_components()

    def _init_components(self):
        """Инициализация компонентов системы"""
        try:
            self._init_llm()
            self._init_embeddings()
            self._init_vector_db()
            logger.info("Компоненты MathRAG инициализированы")
        except Exception as e:
            logger.critical(f"Ошибка инициализации: {str(e)}")
            raise

    def _init_llm(self):
        """Инициализация языковой модели GigaChat"""
        if not os.getenv("GIGACHAT_CREDENTIALS"):
            raise ValueError("Не найдены учетные данные GigaChat!")

        self.llm = GigaChat(
            credentials=os.getenv("GIGACHAT_CREDENTIALS"),
            verify_ssl_certs=False,
            model="GigaChat-Pro",
            timeout=30
        )

    def _init_embeddings(self):
        """Инициализация модели эмбеддингов"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="cointegrated/LaBSE-en-ru",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def _init_vector_db(self):
        """Инициализация векторной базы данных"""
        try:
            os.makedirs(self.db_path, exist_ok=True)

            if os.path.exists(f"{self.db_path}/index.faiss"):
                self.vector_db = FAISS.load_local(
                    self.db_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("Векторная БД загружена")
            else:
                self._create_initial_db()
        except Exception as e:
            logger.error(f"Ошибка БД: {str(e)}")
            raise

    def _create_initial_db(self):
        """Создание начальной БД"""
        initial_docs = [
            Document(
                page_content="solve equation x + 2 = 4",
                metadata={
                    "code": self._base_equation_code(),
                    "lang": "en"
                }
            ),
            Document(
                page_content="решить уравнение x + 2 = 4",
                metadata={
                    "code": self._base_equation_code(),
                    "lang": "ru"
                }
            )
        ]
        self.vector_db = FAISS.from_documents(initial_docs, self.embeddings)
        self._save_db()
        logger.info("Создана новая векторная БД")

    def solve_problem(self, query: str, lang: str = "ru") -> Dict[str, Any]:
        logger.debug(f"Raw input query: {query}")


        """Основной метод решения математических задач"""
        try:
            query = self._normalize_input(query)
            logger.info(f"Обработка запроса: '{query}' (язык: {lang})")

            if not query:
                return self._error_response("empty_query", lang)

            if self._is_equation(query):
                return self._solve_equation(query, lang)

            if self._is_derivative(query):
                return self._solve_derivative(query, lang)

            if self._is_integral(query):
                return self._solve_integral(query, lang)

            return self._solve_general(query, lang)

        except SecurityError as e:
            logger.error(f"Нарушение безопасности: {str(e)}")
            return self._error_response("security", lang, error=str(e))
        except Exception as e:
            logger.error(f"Неожиданная ошибка: {str(e)}", exc_info=True)
            return self._error_response("general", lang)

    def _solve_equation(self, query: str, lang: str) -> Dict[str, Any]:
        try:
            equation = self._parse_equation(query, lang)
            solution = solve(equation, self.x)

            if not solution:
                return self._error_response("no_solution", lang)

            # Проверка численного решения
            if not all(s.is_real for s in solution):
                return self._error_response("complex_solution", lang)

            steps = [
                f"{self.LOCALES[lang]['equation']['original']}: {latex(equation)}",
                f"{self.LOCALES[lang]['equation']['simplified']}: {latex(equation.lhs - equation.rhs)} = 0",
                f"{self.LOCALES[lang]['equation']['solution']}: x = {latex(solution[0])}"
            ]

            return {
                "result": f"x = {solution[0]}",
                "formatted_steps": "\n".join(steps),
                "code": self._equation_code(equation, lang),
                "lang": lang
            }
        except SympifyError as e:
            return self._error_response("parse", lang, error=str(e))

    def _solve_derivative(self, query: str, lang: str) -> Dict[str, Any]:
        """Вычисление производных"""
        try:
            expr = self._parse_expression(query, lang)
            derivative = diff(expr, self.x)

            steps = [
                f"{self.LOCALES[lang]['derivative']['function']}: {latex(expr)}",
                f"{self.LOCALES[lang]['derivative']['derivative']}: {latex(derivative)}"
            ]

            return {
                "result": latex(derivative),
                "formatted_steps": "\n".join(steps),
                "code": self._derivative_code(expr, lang),
                "lang": lang
            }
        except SympifyError as e:
            return self._error_response("parse", lang, error=str(e))

    def _solve_integral(self, query: str, lang: str) -> Dict[str, Any]:
        """Вычисление интегралов"""
        try:
            expr = self._parse_expression(query, lang)
            integral = integrate(expr, self.x)

            steps = [
                f"{self.LOCALES[lang]['integral']['function']}: {latex(expr)}",
                f"{self.LOCALES[lang]['integral']['integral']}: {latex(integral)} + C"
            ]

            return {
                "result": latex(integral),
                "formatted_steps": "\n".join(steps),
                "code": self._integral_code(expr, lang),
                "lang": lang
            }
        except SympifyError as e:
            return self._error_response("parse", lang, error=str(e))

    def _solve_general(self, query: str, lang: str) -> Dict[str, Any]:
        """Решение общих задач с использованием RAG"""
        try:
            examples = self.vector_db.similarity_search(query, k=3)
            context = self._build_context(examples, lang)
            code = self._generate_code(query, context, lang)

            execution_result = self.executor.execute(code)

            if execution_result['status'] != 'success':
                return {"error": execution_result['message']}

            return {
                "result": execution_result['result'],
                "code": code,
                "formatted_steps": self._format_steps(execution_result.get('steps', []), lang),
                "lang": lang
            }
        except Exception as e:
            return self._error_response("general", lang)

    def _parse_equation(self, query: str, lang: str) -> Equality:
        try:
            # Удаляем ключевые слова и лишние пробелы
            query = ' '.join(query.replace("уравнение", "").replace("equation", "").split()).strip()

            # Разделяем с учетом разных форматов равенства
            if 'равно' in query:
                lhs, rhs = query.split('равно', 1)
            else:
                lhs, rhs = query.split('=', 1)

            return Eq(
                parse_expr(lhs.strip(), transformations=transformations),
                parse_expr(rhs.strip(), transformations=transformations)
            )
        except (ValueError, SympifyError) as e:
            raise ValueError(self._get_localized("errors.parse", lang).format(error=str(e)))

    def _parse_expression(self, query: str, lang: str):
        """Парсинг математических выражений"""
        try:
            expr_str = query.split("от")[-1].split("derivative")[-1].strip()
            return parse_expr(expr_str, transformations=transformations)
        except SympifyError as e:
            raise ValueError(self._get_localized("errors.parse", lang).format(error=str(e)))

    def _generate_code(self, query: str, context: str, lang: str) -> str:
        """Генерация кода с помощью LLM"""
        prompt = (
            f"Generate Python code to solve: {query}\n"
            f"Context examples:\n{context}\n\n"
            "Requirements:\n"
            "- Use SymPy library\n"
            "- Add UTF-8 encoding declaration\n"
            "- Avoid unsafe operations\n"
            "- Output result with print()\n\n"
            "Python code:"
        )

        try:
            response = self.llm.chat(prompt)
            code = response.choices[0].message.content
            return self._sanitize_code(code, lang)
        except Exception as e:
            raise RuntimeError(f"Ошибка генерации кода: {str(e)}")

    def _sanitize_code(self, code: str, lang: str) -> str:
        """Санация сгенерированного кода"""
        clean_code = code.replace("```python", "").replace("```", "").strip()

        dangerous_patterns = [
            "__", "os.", "sys.", "open(", "eval(", "exec(",
            "import os", "import sys", "subprocess"
        ]

        for pattern in dangerous_patterns:
            if pattern in clean_code:
                error_msg = self.LOCALES[lang]["errors"]["security"].format(error=pattern)
                raise SecurityError(error_msg)

        if not clean_code.startswith("# -*- coding: utf-8 -*-"):
            clean_code = "# -*- coding: utf-8 -*-\n" + clean_code

        return clean_code

    def _build_context(self, examples: List[Document], lang: str) -> str:
        """Построение контекста из примеров"""
        context = []
        for i, ex in enumerate(examples, 1):
            if ex.metadata.get("lang", "ru") == lang:
                context.append(f"Пример {i}:\n{ex.page_content}\nКод:\n{ex.metadata['code']}")
        return "\n\n".join(context)

    def add_examples(self, csv_path: str):
        """Добавление примеров из CSV"""
        df = pd.read_csv(csv_path)
        docs = [
            Document(
                page_content=row['text'],
                metadata={
                    "code": row['code'],
                    "lang": row.get('lang', 'ru')
                }
            ) for _, row in df.iterrows()
        ]
        self.vector_db.add_documents(docs)
        self._save_db()

    def _save_db(self):
        """Сохранение векторной БД"""
        self.vector_db.save_local(self.db_path)

    def _error_response(self, error_type: str, lang: str, **kwargs) -> Dict[str, Any]:
        """Формирование ответа с ошибкой"""
        error_template = self.LOCALES[lang]["errors"].get(error_type, "Неизвестная ошибка")
        return {"error": error_template.format(**kwargs)}

    def _get_localized(self, key: str, lang: str) -> str:
        """Получение локализованного текста"""
        keys = key.split('.')
        value = self.LOCALES[lang]
        for k in keys:
            value = value.get(k, {})
        return value if isinstance(value, str) else key

    def _normalize_input(self, text: str) -> str:
        """Нормализация математических обозначений"""
        replacements = (
            ('^', '**'),
            ('÷', '/'),
            ('×', '*'),
            ('–', '-'),
            ('−', '-'),
            ('\\', ''),
            ('‘', "'"),
            ('’', "'"),
            ('`', "'")
        )
        for old, new in replacements:
            text = text.replace(old, new)
        return text.strip()

    def _is_equation(self, query: str) -> bool:
        """Проверка на уравнение"""
        keywords = ["реши", "solve", "уравнение", "equation", "="]
        return any(kw in query for kw in keywords) and "=" in query

    def _equation_code(self, equation: Equality, lang: str) -> str:
        return f"""# -*- coding: utf-8 -*-
    from sympy import symbols, Eq, solve

    x = symbols('x')
    equation = Eq({equation.lhs.evalf()}, {equation.rhs.evalf()})
    solution = solve(equation, x)
    print(solution[0] if solution else "Нет решения")
    """

    def _is_derivative(self, query: str) -> bool:
        """Проверка на производную"""
        keywords = ["производн", "derivative", "diff"]
        return any(kw in query for kw in keywords)

    def _is_integral(self, query: str) -> bool:
        """Проверка на интеграл"""
        keywords = ["интеграл", "integral", "∫"]
        return any(kw in query for kw in keywords)

    def _format_steps(self, steps: List[str], lang: str) -> str:
        """Форматирование шагов решения"""
        return "\n".join([f"• {step}" for step in steps])

    @staticmethod
    def _base_equation_code() -> str:
        """Базовый пример кода для уравнения"""
        return (
            "from sympy import symbols, Eq, solve\n"
            "x = symbols('x')\n"
            "eq = Eq(x + 2, 4)\n"
            "print(solve(eq, x)[0])"
        )


if __name__ == "__main__":
    rag = MathRAG()

    # Пример использования
    test_cases = [
        ("решить уравнение x^2 - 9 = 0", "ru"),
        ("find derivative of sin(x) + 3x^2", "en"),
        ("∫(x^3) dx", "en")
    ]

    for query, lang in test_cases:
        print(f"\nЗапрос: {query} ({lang})")
        result = rag.solve_problem(query, lang)

        if "error" in result:
            print(f"Ошибка: {result['error']}")
        else:
            print(f"Результат: {result['result']}")
            print(f"Шаги решения:\n{result['formatted_steps']}")
            print(f"Код:\n{result['code']}")
