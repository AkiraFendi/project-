import os
import logging
import re
from sympy import S
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
            "limit": {
                "original": "Исходный предел",
                "solution": "Результат предела",
                "original": "Исходный предел",
                "solution": "Предел при"
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
                "complex_solution": "Комплексные решения не поддерживаются",
                "general": "Ошибка решения",
                "undefined_query": "Не удалось распознать запрос. Пожалуйста, уточните ваш вопрос или предоставьте больше контекста."
            }
        },
        "en": {
            "equation": {
                "original": "Original equation",
                "simplified": "Simplified form",
                "solution": "Solution"
            },
            "limit": {
                "original": "Original limit",
                "solution": "Limit result",
                "original": "Original limit",
                "solution": "Limit as"
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
                "complex_solution": "Complex solutions not supported",
                "general": "Solution error",
                "undefined_query": "Unable to process your request. Please clarify your question or provide more context."
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
        """Инициализация GigaChat"""
        if not os.getenv("GIGACHAT_CREDENTIALS"):
            raise ValueError("Не найдены учетные данные GigaChat!")

        self.llm = GigaChat(
            credentials=os.getenv("GIGACHAT_CREDENTIALS"),
            verify_ssl_certs=False,
            model="GigaChat-Pro",
            timeout=30
        )

    def _init_embeddings(self):
        """Инициализация эмбеддингов"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="cointegrated/LaBSE-en-ru",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def _init_vector_db(self):
        """Инициализация векторной БД"""
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
                metadata={"code": self._base_equation_code(), "lang": "en"}
            ),
            Document(
                page_content="решить уравнение x + 2 = 4",
                metadata={"code": self._base_equation_code(), "lang": "ru"}
            )
        ]
        self.vector_db = FAISS.from_documents(initial_docs, self.embeddings)
        self._save_db()
        logger.info("Создана новая векторная БД")

    def _save_db(self):
        """Сохранение БД"""
        self.vector_db.save_local(self.db_path)

    def _normalize_input(self, text: str) -> str:
        """Нормализация математических выражений"""
        replacements = [
            (r'\b(sin|cos|tan|cot|sec|csc)\s+(\w+)', r'\1(\2)'),

            (r'lim\s+([a-zA-Z]+)\s*->\s*([\d.]+)\s*\((.*)\)', r'limit(\3, \1, \2)'),

            (r'(\d)([a-zA-Z])', r'\1*\2'),

            (r'\^', '**'), (r'÷', '/'), (r'×', '*'),
            (r'[–−]', '-'), (r'\s+', '')
        ]

        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)

        return text.strip()

    def solve_problem(self, query: str, lang: str = "ru") -> Dict[str, Any]:
        """Основной метод решения задач"""
        try:
            query = self._normalize_input(query)
            logger.info(f"Обработка запроса: '{query}' (язык: {lang})")

            if not query:
                return self._error_response("empty_query", lang)

            try:
                expr = sympify(query, evaluate=False)
                if expr.is_Number or not expr.free_symbols:
                    result = expr.evalf()
                    return self._format_simple_calculation(query, result, lang)
            except SympifyError:
                pass

            if self._is_equation(query):
                return self._solve_equation(query, lang)
            if self._is_derivative(query):
                return self._solve_derivative(query, lang)
            if self._is_integral(query):
                return self._solve_integral(query, lang)
            if self._is_limit(query):
                return self._solve_limit(query, lang)

            general_result = self._solve_general(query, lang)
            if "error" in general_result:
                return self._error_response("undefined_query", lang)
            return general_result

        except SecurityError as e:
            logger.error(f"Нарушение безопасности: {str(e)}")
            return self._error_response("security", lang, error=str(e))
        except Exception as e:
            logger.error(f"Неожиданная ошибка: {str(e)}", exc_info=True)
            return self._error_response("general", lang)

    def _is_limit(self, query: str) -> bool:
        keywords = ["lim", "предел", "limit"]
        return any(kw in query.lower() for kw in keywords)

    def _solve_limit(self, query: str, lang: str) -> Dict[str, Any]:
        try:
            # Парсинг выражения предела
            match = re.match(r"limit\((.*),\s*([a-zA-Z]+),\s*([\d.]+)\)", query)
            if not match:
                raise ValueError("Некорректный формат предела")

            expr_str, var_str, point_str = match.groups()
            var = symbols(var_str)
            point = float(point_str)
            expr = parse_expr(expr_str, transformations=transformations)

            limit_result = expr.limit(var, point)

            steps = [
                f"{self.LOCALES[lang]['limit']['original']}: {latex(expr)}",
                f"{self.LOCALES[lang]['limit']['solution']} {var} → {point}: {latex(limit_result)}"
            ]

            return {
                "result": latex(limit_result),
                "formatted_steps": "\n".join(steps),
                "code": self._limit_code(expr, var, point, lang),
                "lang": lang
            }
        except Exception as e:
            return self._error_response("parse", lang, error=str(e))

    def _parse_limit(self, query: str) -> tuple:
        """Парсинг выражения предела"""
        try:
            # Ищем паттерн limit(var, point)(expr)
            match = re.match(r"limit\((\w+),\s*([\d.]+)\)(.*)", query)
            if not match:
                raise ValueError("Некорректный формат предела")

            var_str = match.group(1)
            point = float(match.group(2))
            expr_str = match.group(3).strip()

            var = symbols(var_str)
            expr = parse_expr(expr_str, transformations=transformations)

            return expr, var, point
        except Exception as e:
            raise ValueError(f"Ошибка парсинга: {str(e)}")

    def _limit_code(self, expr, var, point, lang: str) -> str:
        """Генерация корректного кода для пределов"""
        return f"""# -*- coding: utf-8 -*-
    from sympy import symbols, limit

    {var} = symbols('{var}')
    print(limit({latex(expr)}, {var}, {point}))"""


    def _format_simple_calculation(self, query: str, result: Any, lang: str) -> Dict[str, Any]:
        formatted_result = self._format_number(result)
        return {
            "result": formatted_result,
            "formatted_steps": f"{self.LOCALES[lang]['steps']}: {query} = {formatted_result}",
            "code": f"from sympy import sympify\nprint(sympify('{query}').evalf())",
            "lang": lang
        }

    def _format_number(self, number: Any) -> str:
        try:
            num = float(number)
            if num.is_integer():
                return str(int(num))
            return f"{num:.3f}".rstrip('0').rstrip('.')
        except:
            return str(number)

    def _is_equation(self, query: str) -> bool:
        keywords = ["реши", "solve", "уравнение", "equation", "="]
        return any(kw in query.lower() for kw in keywords) and "=" in query

    def _solve_equation(self, query: str, lang: str) -> Dict[str, Any]:
        try:
            equation = self._parse_equation(query, lang)
            logger.info(f"Уравнение после парсинга: {latex(equation)}")

            solution = solve(equation, self.x, domain=S.Reals, check=False)

            real_solutions = [s for s in solution if s.is_real and not s.has(S.ComplexInfinity)]

            if not real_solutions:
                return self._error_response("no_solution", lang)

            solutions_str = ", ".join([f"x = {self._format_symbolic(s)}" for s in real_solutions])

            steps = [
                f"{self.LOCALES[lang]['equation']['original']}: {latex(equation)}",
                f"{self.LOCALES[lang]['equation']['solution']}: {solutions_str}"
            ]

            return {
                "result": solutions_str,
                "formatted_steps": "\n".join(steps),
                "code": self._equation_code(equation, lang),
                "lang": lang
            }
        except (SympifyError, ValueError) as e:
            return self._error_response("parse", lang, error=str(e))

    def _format_symbolic(self, expr):
        """Форматирование символьных констант"""
        from sympy import pi, E
        replacements = {pi: 'π', E: 'e'}
        return replacements.get(expr, latex(expr))

    def _parse_equation(self, query: str, lang: str) -> Equality:
        try:

            keywords = ["решить", "уравнение", "solve", "equation"]
            for kw in keywords:
                query = query.replace(kw, "")
            query = query.strip()

            if '=' not in query:
                raise ValueError("Уравнение должно содержать знак '='")

            lhs, rhs = query.split('=', 1)
            lhs_expr = parse_expr(lhs.strip(), transformations=transformations)
            rhs_expr = parse_expr(rhs.strip(), transformations=transformations)

            return Eq(lhs_expr, rhs_expr)
        except (ValueError, SympifyError) as e:
            raise ValueError(self.LOCALES[lang]["errors"]["parse"].format(error=str(e)))

    def _equation_code(self, equation: Equality, lang: str) -> str:
        """Исправленный код для уравнений"""
        return f"""# -*- coding: utf-8 -*-
    from sympy import symbols, Eq, solve

    x = symbols('x')
    equation = Eq({latex(equation.lhs)}, {latex(equation.rhs)})
    solution = solve(equation, x)
    print(solution)"""

    def _is_derivative(self, query: str) -> bool:
        keywords = ["производн", "derivative", "diff"]
        return any(kw in query.lower() for kw in keywords)

    def _solve_derivative(self, query: str, lang: str) -> Dict[str, Any]:
        try:
            expr_str = query.split("derivative")[-1].strip()
            expr = parse_expr(expr_str, transformations=transformations)
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

    def _derivative_code(self, expr, lang: str) -> str:
        return f"""# -*- coding: utf-8 -*-
from sympy import symbols, diff

x = symbols('x')
expr = {latex(expr)}
print(diff(expr, x))"""

    def _is_integral(self, query: str) -> bool:
        keywords = ["интеграл", "integral", "∫", "проинтегрируй", "integrate"]
        return any(kw in query.lower() for kw in keywords)

    def _solve_integral(self, query: str, lang: str) -> Dict[str, Any]:
        try:
            expr_str = query.split("от")[-1].strip()
            expr = parse_expr(expr_str, transformations=transformations)
            integral = integrate(expr, self.x)

            steps = [
                f"{self.LOCALES[lang]['integral']['function']}: {latex(expr)}",
                f"{self.LOCALES[lang]['integral']['integral']}: {latex(integral)} + C"
            ]

            return {
                "result": f"{latex(integral)} + C",
                "formatted_steps": "\n".join(steps),
                "code": f"""from sympy import symbols, integrate
x = symbols('x')
print(integrate('{expr_str}', x))""",
                "lang": lang
            }
        except SympifyError as e:
            return self._error_response("parse", lang, error=str(e))

    def _solve_general(self, query: str, lang: str) -> Dict[str, Any]:
        try:
            examples = self.vector_db.similarity_search(query, k=3)
            if not examples:
                return self._error_response("undefined_query", lang)
            context = self._build_context(examples, lang)
            code = self._generate_code(query, context, lang)
            execution_result = self.executor.execute(code)

            if execution_result['status'] != 'success':
                return self._error_response("undefined_query", lang)

            return {
                "result": execution_result['result'],
                "code": code,
                "formatted_steps": self._format_steps(execution_result.get('steps', []), lang),
                "lang": lang
            }
        except Exception as e:
            return self._error_response("general", lang)

    def _generate_code(self, query: str, context: str, lang: str) -> str:
        prompt = f"Generate Python code to solve: {query}\nContext examples:\n{context}\n\nRequirements:\n- Use SymPy\n- Add UTF-8 encoding\n- Avoid unsafe operations\nPython code:"
        response = self.llm.chat(prompt)
        code = response.choices[0].message.content
        return self._sanitize_code(code, lang)

    def _sanitize_code(self, code: str, lang: str) -> str:
        clean_code = code.replace("```python", "").replace("```", "").strip()
        dangerous_patterns = ["__", "os.", "sys.", "open(", "eval(", "exec(", "import os", "import sys", "subprocess"]
        for pattern in dangerous_patterns:
            if pattern in clean_code:
                raise SecurityError(self.LOCALES[lang]["errors"]["security"].format(error=pattern))
        if not clean_code.startswith("# -*- coding: utf-8 -*-"):
            clean_code = "# -*- coding: utf-8 -*-\n" + clean_code
        return clean_code

    def _build_context(self, examples: List[Document], lang: str) -> str:
        context = []
        for i, ex in enumerate(examples, 1):
            if ex.metadata.get("lang", "ru") == lang:
                context.append(f"Пример {i}:\n{ex.page_content}\nКод:\n{ex.metadata['code']}")
        return "\n\n".join(context)

    def add_examples(self, csv_path: str):
        df = pd.read_csv(csv_path)
        docs = [
            Document(
                page_content=row['text'],
                metadata={"code": row['code'], "lang": row.get('lang', 'ru')}
            ) for _, row in df.iterrows()
        ]
        self.vector_db.add_documents(docs)
        self._save_db()

    def _error_response(self, error_type: str, lang: str, **kwargs) -> Dict[str, Any]:
        error_template = self.LOCALES[lang]["errors"].get(error_type, "Неизвестная ошибка")
        return {"error": error_template.format(**kwargs)}

    def _format_steps(self, steps: List[str], lang: str) -> str:
        return "\n".join([f"• {step}" for step in steps])

    @staticmethod
    def _base_equation_code() -> str:
        return (
            "from sympy import symbols, Eq, solve\n"
            "x = symbols('x')\n"
            "eq = Eq(x + 2, 4)\n"
            "print(solve(eq, x)[0])"
        )