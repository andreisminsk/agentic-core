"""Calculator tool — safely evaluate mathematical expressions."""

import ast
import math
import operator


_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_NAMES = {
    "pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf, "nan": math.nan,
    "abs": abs, "round": round, "min": min, "max": max,
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "log": math.log, "log2": math.log2, "log10": math.log10, "ln": math.log,
    "exp": math.exp, "pow": math.pow, "ceil": math.ceil, "floor": math.floor,
    "factorial": math.factorial, "gcd": math.gcd,
    "degrees": math.degrees, "radians": math.radians,
    "hypot": math.hypot, "dist": math.dist,
}
# cbrt added in Python 3.11
if hasattr(math, "cbrt"):
    _SAFE_NAMES["cbrt"] = math.cbrt


def _safe_eval(expr):
    tree = ast.parse(expr, mode="eval")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, complex)):
                return node.value
            raise ValueError(f"Unsupported constant: {type(node.value).__name__}")
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in _SAFE_OPS:
                raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
            return _SAFE_OPS[type(node.op)](_eval(node.operand))
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in _SAFE_OPS:
                raise ValueError(f"Unsupported binary op: {type(node.op).__name__}")
            return _SAFE_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.Call):
            func = _eval(node.func) if isinstance(node.func, ast.Name) else None
            if func is None or func not in _SAFE_NAMES.values():
                raise ValueError("Unsupported function call")
            args = [_eval(a) for a in node.args]
            return func(*args)
        elif isinstance(node, ast.Name):
            if node.id in _SAFE_NAMES:
                return _SAFE_NAMES[node.id]
            raise ValueError(f"Unsupported name: {node.id}")
        elif isinstance(node, ast.Tuple):
            return tuple(_eval(e) for e in node.elts)
        elif isinstance(node, ast.List):
            return [_eval(e) for e in node.elts]
        else:
            raise ValueError(f"Unsupported expression: {type(node).__name__}")

    return _eval(tree)


class CalculatorTool:
    name = "calculator"
    description = "Evaluate mathematical expressions safely."
    system_prompt = (
        "## calculator\n"
        "Evaluates mathematical expressions safely. ALWAYS use this tool for ANY "
        "mathematical computation — never attempt mental arithmetic.\n"
        "Parameters: expression (string, required), precision (int, optional).\n"
        "Supports: +, -, *, /, //, %, **, sqrt, sin, cos, tan, log, exp, factorial, "
        "gcd, hypot, abs, round, min, max, ceil, floor, pi, e, tau, and more."
    )
    auto_approve = True

    def execute(self, params, workdir=None):
        expr = params.get("expression", "").strip()
        if not expr:
            return "Error: 'expression' parameter is required"
        precision = params.get("precision")
        try:
            result = _safe_eval(expr)
        except Exception as exc:
            return f"Error: Failed to evaluate '{expr}': {exc}"
        if precision is not None and isinstance(result, (int, float)):
            result = round(result, int(precision))
        if isinstance(result, float) and result == int(result) and abs(result) < 1e15:
            result = int(result)
        return f"{expr} = {result}"
