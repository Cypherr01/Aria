"""
tools.code_interpreter
======================
Sandboxed Python code execution using RestrictedPython.
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

logger = logging.getLogger(__name__)

BLOCKED_MODULES = frozenset({
    "subprocess", "os", "sys", "socket", "requests", "urllib",
    "ftplib", "smtplib", "ctypes", "importlib", "shutil", "glob",
    "builtins", "pickle", "eval", "exec"
})

def _execute_code(code: str) -> Dict[str, Any]:
    """Synchronous execution of restricted code."""
    import collections
    import datetime
    import decimal
    import fractions
    import itertools
    import json
    import math
    import random
    import re
    import statistics
    import string
    import time
    
    import RestrictedPython
    from RestrictedPython import compile_restricted, safe_builtins

    try:
        byte_code = compile_restricted(code, '<inline>', 'exec')
    except SyntaxError as e:
        return {"success": False, "stderr": f"SyntaxError: {e}"}
    except Exception as e:
        return {"success": False, "stderr": f"{type(e).__name__}: {e}"}

    restricted_globals = {
        "__builtins__": safe_builtins,
        "math": math,
        "json": json,
        "re": re,
        "datetime": datetime,
        "collections": collections,
        "itertools": itertools,
        "random": random,
        "string": string,
        "decimal": decimal,
        "fractions": fractions,
        "statistics": statistics
    }

    # Block imports
    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in BLOCKED_MODULES:
            raise ImportError(f"Importing '{name}' is not allowed in this sandbox.")
        return __import__(name, globals, locals, fromlist, level)
        
    restricted_globals["__builtins__"]["__import__"] = safe_import
    class CustomPrint:
        def __init__(self, _getattr_=None): pass
        def _call_print(self, *objects, **kwargs):
            import builtins
            builtins.print(*objects, **kwargs)

    restricted_globals["__builtins__"]["print"] = print
    restricted_globals["_print_"] = CustomPrint
    restricted_globals["_getattr_"] = RestrictedPython.Guards.safer_getattr
    restricted_globals["_getitem_"] = RestrictedPython.Eval.default_guarded_getitem
    restricted_globals["_getiter_"] = RestrictedPython.Eval.default_guarded_getiter
    restricted_globals["_write_"] = RestrictedPython.Guards.full_write_guard

    stdout_buffer = io.StringIO()
    start_time = time.monotonic()
    
    try:
        with contextlib.redirect_stdout(stdout_buffer):
            exec(byte_code, restricted_globals)
        success = True
        stderr = ""
    except Exception as e:
        success = False
        stderr = f"{type(e).__name__}: {e}"

    execution_time_ms = (time.monotonic() - start_time) * 1000

    return {
        "code": code,
        "stdout": stdout_buffer.getvalue(),
        "stderr": stderr,
        "success": success,
        "execution_time_ms": execution_time_ms,
        "return_value": restricted_globals.get("result", None)
    }

async def code_interpreter(code: str, timeout_sec: int = 15, **kwargs) -> Dict[str, Any]:
    """Execute Python code in a restricted sandbox with a timeout."""
    loop = asyncio.get_running_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _execute_code, code),
            timeout=timeout_sec
        )
        return result
    except asyncio.TimeoutError:
        return {
            "code": code,
            "stdout": "",
            "stderr": f"TimeoutError: execution exceeded {timeout_sec}s",
            "success": False,
            "execution_time_ms": timeout_sec * 1000,
            "return_value": None
        }
    except Exception as e:
        return {
            "code": code,
            "stdout": "",
            "stderr": f"{type(e).__name__}: {str(e)}",
            "success": False,
            "execution_time_ms": 0,
            "return_value": None
        }
