import ast
import os
import tempfile
import docker
import logging
from typing import Dict
from docker.errors import DockerException
from requests.exceptions import ReadTimeout

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Кастомное исключение для ошибок безопасности"""
    pass


class SafeExecutor:
    def __init__(self):
        self.allowed_imports = {'sympy', 'math', 'numpy'}
        self.blocked_keywords = {
            '__', 'os.', 'sys.', 'open(', 'eval(', 'exec(',
            'import os', 'import sys', 'subprocess', 'shutil',
            'socket', 'tempfile', 'system', 'fork', 'kill',
            'popen', 'pty', 'chmod', 'rmtree', 'remove'
        }
        self.client = docker.from_env()
        self.timeout = 15  # секунд
        self.mem_limit = '100m'  # лимит памяти
        self.cpu_quota = 50000  # лимит CPU

    def _validate_code(self, code: str):
        """Проверка кода на опасные конструкции"""
        # Проверка по ключевым словам
        for kw in self.blocked_keywords:
            if kw in code:
                raise SecurityError(f"Обнаружено запрещенное ключевое слово: {kw}")

        # Проверка импортов через AST
        try:
            parsed = ast.parse(code)
            for node in ast.walk(parsed):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    module = node.names[0].name.split('.')[0]
                    if module not in self.allowed_imports:
                        raise SecurityError(f"Запрещенный импорт: {module}")
        except SyntaxError as e:
            raise SecurityError(f"Синтаксическая ошибка: {str(e)}")

    def execute(self, code: str) -> Dict[str, any]:
        """Безопасное выполнение кода в Docker-контейнере"""
        result = {"status": "unknown", "message": "", "result": None}
        container = None
        temp_path = None

        try:
            self._validate_code(code)

            # Добавляем обязательные заголовки
            header = (
                "# -*- coding: utf-8 -*-\n"
                "from sympy import *\n"
                "import math\n"
                "import numpy as np\n\n"
            )
            code = header + code

            # Создаем временный файл
            with tempfile.NamedTemporaryFile(
                    mode='w',
                    suffix='.py',
                    delete=False,
                    encoding='utf-8'
            ) as f:
                f.write(code)
                temp_path = f.name
                logger.info(f"Создан временный файл: {temp_path}")

            # Конфигурация Docker
            container_config = {
                'image': 'python:3.9-slim',
                'command': f"timeout {self.timeout} python /tmp/{os.path.basename(temp_path)}",
                'volumes': {
                    temp_path: {
                        'bind': f'/tmp/{os.path.basename(temp_path)}',
                        'mode': 'ro'
                    }
                },
                'detach': True,
                'mem_limit': self.mem_limit,
                'cpu_period': 100000,
                'cpu_quota': self.cpu_quota,
                'network_mode': 'none',
                'auto_remove': True,
                'stderr': True,
                'stdout': True
            }

            # Запускаем контейнер
            container = self.client.containers.run(**container_config)
            logger.info(f"Запущен контейнер: {container.id}")

            # Ожидаем завершения
            try:
                exit_code = container.wait(timeout=self.timeout + 5)['StatusCode']
                logs = container.logs().decode('utf-8', errors='replace').strip()
                logger.debug(f"Логи выполнения:\n{logs}")
            except ReadTimeout:
                raise TimeoutError(f"Превышено время ожидания ({self.timeout}s)")

            # Обработка результатов
            if exit_code != 0:
                result.update({
                    "status": "error",
                    "type": "runtime",
                    "message": f"Код выхода: {exit_code}",
                    "details": logs
                })
            else:
                try:
                    output = ast.literal_eval(logs)
                    result.update({
                        "status": "success",
                        "result": output
                    })
                except:
                    result.update({
                        "status": "success",
                        "result": logs
                    })

        except SecurityError as e:
            result.update({
                "status": "error",
                "type": "security",
                "message": str(e)
            })
        except TimeoutError as e:
            result.update({
                "status": "error",
                "type": "timeout",
                "message": str(e)
            })
        except DockerException as e:
            result.update({
                "status": "error",
                "type": "docker",
                "message": str(e)
            })
        except Exception as e:
            result.update({
                "status": "error",
                "type": "unknown",
                "message": f"Неизвестная ошибка: {str(e)}"
            })
        finally:
            # Очистка временных файлов
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    logger.info(f"Удален временный файл: {temp_path}")
                except Exception as e:
                    logger.error(f"Ошибка удаления файла: {str(e)}")

            # Остановка контейнера при необходимости
            if container:
                try:
                    container.stop(timeout=1)
                    logger.info(f"Остановлен контейнер: {container.id}")
                except Exception as e:
                    logger.error(f"Ошибка остановки контейнера: {str(e)}")

        return result


if __name__ == "__main__":
    executor = SafeExecutor()

    # Пример безопасного кода
    test_code = """
from sympy import symbols, integrate

x = symbols('x')
result = integrate(x**2, x)
print({"result": str(result)})
    """

    # Пример опасного кода
    dangerous_code = "import os; os.remove('important_file.txt')"

    # Тестирование
    print("Тест безопасного кода:")
    print(executor.execute(test_code))

    print("\nТест опасного кода:")
    print(executor.execute(dangerous_code))

