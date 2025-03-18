from core.safe_executor import SafeExecutor

executor = SafeExecutor()

# Пример безопасного кода
safe_code = """
from sympy import symbols, solve
x = symbols('x')
steps = ['Решение уравнения x² - 4 = 0']
result = solve(x**2 - 4, x)
"""

# Пример опасного кода (попытка удалить файлы)
dangerous_code = """
import os
os.system("rm -rf /")
"""

if __name__ == "__main__":
    print("=== Тест безопасного кода ===")
    safe_result = executor.execute(safe_code)
    print("Результат:", safe_result)

    print("\n=== Тест опасного кода ===")
    dangerous_result = executor.execute(dangerous_code)
    print("Результат:", dangerous_result)