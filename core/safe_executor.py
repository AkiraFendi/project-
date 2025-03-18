import docker
import tempfile
import os
import ast

class SecurityError(Exception):
    pass

class SafeExecutor:
    def __init__(self):
        self.allowed_imports = {'sympy', 'math', 'numpy'}
        self.blocked_keywords = {
            'os.', 'subprocess', 'open(', '__', 'eval', 'exec',
            'system', 'shutil', 'socket', 'tempfile'
        }
        self.client = docker.from_env()

    def _validate_code(self, code: str):
        for kw in self.blocked_keywords:
            if kw in code:
                raise SecurityError(f"Запрещенная операция: {kw}")

    def execute(self, code: str) -> dict:
        try:
            self._validate_code(code)

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_path = f.name

            container = self.client.containers.run(
                image='math-solver',
                command=f"python /tmp/{os.path.basename(temp_path)}",
                volumes={temp_path: {'bind': f'/tmp/{os.path.basename(temp_path)}', 'mode': 'ro'}},
                detach=True
            )

            exit_code = container.wait()['StatusCode']
            logs = container.logs().decode().strip()
            container.remove()

            os.remove(temp_path)

            if exit_code != 0:
                return {"error": f"Exit code {exit_code}: {logs}"}

            try:
                # Пробуем распарсить вывод как Python-объект
                parsed_result = ast.literal_eval(logs)
                return {"result": parsed_result}
            except:
                # Если не получается, возвращаем как строку
                return {"result": logs}

        except docker.errors.ContainerError as e:
            return {"error": f"Execution error: {e.stderr.decode()}"}
        except Exception as e:
            return {"error": str(e)}