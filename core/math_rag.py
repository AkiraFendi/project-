import os
import pandas as pd
import logging
from dotenv import load_dotenv
from gigachat import GigaChat
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from sympy import latex, sympify
from .safe_executor import SafeExecutor, SecurityError

# Загрузка переменных окружения
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MathRAG:
    def __init__(self, db_path="data/math_db"):
        self.db_path = db_path
        self.embeddings = None
        self.vector_db = None
        self.llm = None
        self.executor = SafeExecutor()

        if not os.getenv("GIGACHAT_CREDENTIALS"):
            raise ValueError("GigaChat credentials not found in .env!")

        self._init_llm()
        self._init_embeddings()
        self._init_vector_db()

    def _init_llm(self):
        try:
            self.llm = GigaChat(
                credentials=os.getenv("GIGACHAT_CREDENTIALS"),
                verify_ssl_certs=False,
                model="GigaChat-Pro"
            )
        except Exception as e:
            logger.error(f"GigaChat init error: {str(e)}")
            raise


    def _init_embeddings(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="cointegrated/LaBSE-en-ru"
        )

    def _init_vector_db(self):
        try:
            os.makedirs(self.db_path, exist_ok=True)
            if os.path.exists(f"{self.db_path}/index.faiss"):
                self.vector_db = FAISS.load_local(
                    self.db_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                initial_doc = Document(
                    page_content="init",
                    metadata={"code": "from sympy import symbols"}
                )
                self.vector_db = FAISS.from_documents([initial_doc], self.embeddings)
                self._save_db()
        except Exception as e:
            logger.error(f"Vector DB init error: {str(e)}")
            raise

    def add_examples(self, examples_path: str):
        try:
            df = pd.read_csv(examples_path)
            docs = [
                Document(
                    page_content=row['text'],
                    metadata={"code": row['code']}
                ) for _, row in df.iterrows()
            ]
            self.vector_db.add_documents(docs)
            self._save_db()
            logger.info(f"Added {len(docs)} examples from {examples_path}")
        except Exception as e:
            logger.error(f"Error loading examples: {str(e)}")
            raise

    def generate_code(self, query: str) -> str:
        try:
            examples = self.vector_db.similarity_search(query, k=3)
            context = "\n".join([
                f"Пример {i + 1}:\nЗапрос: {ex.page_content}\nКод: {ex.metadata['code']}"
                for i, ex in enumerate(examples)
            ])

            prompt = f"""Сгенерируй Python-код для решения задачи:
{query}

Требования:
1. ТОЛЬКО КОД НА PYTHON, без комментариев и текста
2. Используй только sympy
3. Объяви все переменные через symbols()
4. Результат сохрани в переменную result
5. Добавь строку print(result) в конце
6. Не используй функции (def) и другие print

Примеры:
{context}

Код:"""

            response = self.llm.chat(prompt)
            raw_code = response.choices[0].message.content
            return self._sanitize_code(raw_code)

        except Exception as e:
            logger.error(f"Code generation error: {str(e)}")
            raise

    def _sanitize_code(self, raw_code: str) -> str:
        clean_code = raw_code.replace("```python", "").replace("```", "").strip()

        # Удаление всех не-ASCII символов
        clean_code = ''.join([c if ord(c) < 128 else ' ' for c in clean_code])

        # Удаление пустых строк и лишних пробелов
        lines = [line.strip() for line in clean_code.split('\n') if line.strip()]
        clean_code = '\n'.join(lines)

        if "def " in clean_code:
            raise SecurityError("Обнаружено объявление функции")
        return clean_code

    def solve_problem(self, query: str) -> dict:
        try:
            code = self.generate_code(query)
            logger.info(f"Generated code:\n{code}")
            result = self.executor.execute(code)

            if "error" not in result:
                result["code"] = code
                if "steps" in result:
                    result["formatted_steps"] = self._format_steps(result["steps"])

            return result
        except SecurityError as e:
            return {"error": f"Security Error: {str(e)}"}
        except Exception as e:
            return {"error": str(e)}

    def _format_steps(self, steps: list) -> str:
        formatted = []
        for step in steps:
            try:
                if "=" in step:
                    left, right = step.split("=", 1)
                    formatted_step = f"{latex(sympify(left))} = {latex(sympify(right))}"
                    formatted.append(f"• ${formatted_step}$")
                else:
                    formatted.append(f"• {step}")
            except:
                formatted.append(f"• {step}")
        return "\n".join(formatted)

    def _save_db(self):
        try:
            self.vector_db.save_local(self.db_path)
        except Exception as e:
            logger.error(f"DB save error: {str(e)}")
            raise


if __name__ == "__main__":
    rag = MathRAG()
    rag.add_examples("data/math_examples.csv")
    test_query = "решить уравнение x^2 - 4 = 0"
    result = rag.solve_problem(test_query)
    print(result)
