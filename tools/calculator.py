"""
tools.calculator
================
Pure Python synchronous math evaluator. No LLM used.
"""
import math
from typing import Any, Dict

SAFE_NAMES = {name: getattr(math, name) for name in dir(math) if not name.startswith('_')}
SAFE_NAMES.update({
    "abs": abs, 
    "round": round, 
    "min": min, 
    "max": max,
    "sum": sum, 
    "pow": pow, 
    "int": int, 
    "float": float
})

ALLOWED_CHARS = set("0123456789.+-*/(),%^eE ")

def calculator(expression: str) -> Dict[str, Any]:
    """Evaluate a mathematical expression safely."""
    # Check for invalid characters (excluding function names)
    # We'll just strip letters that belong to SAFE_NAMES and check the rest
    
    clean_expr = expression
    for name in SAFE_NAMES.keys():
        clean_expr = clean_expr.replace(name, "")
        
    for char in clean_expr:
        if char not in ALLOWED_CHARS:
            return {"error": f"Invalid character '{char}' in expression"}

    # Replace ^ with ** for python eval
    eval_expr = expression.replace("^", "**")

    try:
        result = eval(eval_expr, {"__builtins__": {}}, SAFE_NAMES)
        return {
            "result": float(result),
            "expression": expression,
            "formatted": f"{result:,.6g}"
        }
    except ZeroDivisionError:
        return {"error": "Division by zero"}
    except Exception as e:
        return {"error": str(e)}
