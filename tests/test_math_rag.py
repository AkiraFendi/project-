import os
import pytest
from core.math_rag import MathRAG


@pytest.fixture
def rag_system():
    test_db_path = os.path.abspath("test_db")
    test_examples = [
        {
            "text": "решить уравнение x^2 - 4",
            "code": """
                from sympy import symbols, solve
                x = symbols('x')
                result = solve(x**2 - 4, x)
                print(result)
            """
        },
        {
            "text": "вычислить интеграл x^2",
            "code": """
                from sympy import symbols, integrate
                x = symbols('x')
                result = integrate(x**2, x)
                print(result)
            """
        }
    ]

    # Create test CSV
    with open("test_examples.csv", "w", encoding="utf-8") as f:
        f.write("text,code\n")
        for ex in test_examples:
            f.write(f'"{ex["text"]}","{ex["code"]}"\n')

    rag = MathRAG(db_path=test_db_path)
    rag.add_examples("test_examples.csv")
    yield rag

    # Cleanup
    if os.path.exists(test_db_path):
        for root, dirs, files in os.walk(test_db_path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(test_db_path)
    if os.path.exists("test_examples.csv"):
        os.remove("test_examples.csv")


def test_equation_solving(rag_system):
    query = "решить уравнение x^2 - 4"
    result = rag_system.solve_problem(query)

    assert "error" not in result, f"Error: {result.get('error', '')}"
    assert result.get("result") == [-2, 2], "Wrong solution"
    assert "x = symbols('x')" in result.get("code", ""), "Missing symbols declaration"


def test_integration(rag_system):
    query = "вычислить интеграл x^2"
    result = rag_system.solve_problem(query)

    assert "error" not in result, f"Error: {result.get('error', '')}"
    assert "x**3/3" in str(result.get("result", "")), "Wrong integral"
    assert "integrate(x**2, x)" in result.get("code", ""), "Missing integrate call"