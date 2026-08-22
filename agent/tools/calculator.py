"""
Calculator Tool

Evaluates mathematical expressions. The LLM sends something like
"245 * 18 + 32" and this tool computes and returns the result.

Safety note:
  We do NOT use Python's eval() directly because that would let someone
  execute arbitrary code: eval("__import__('os').system('rm -rf /')").
  Instead, we use a restricted approach that only allows math operations.

  In an interview, if someone asks "how do you handle code injection in the
  calculator?", you can explain this restricted evaluation approach.
"""

import ast
import operator
from agent.tools.base import BaseTool


# These are the ONLY operations we allow. Anything else (imports, function
# calls, variable assignments) will be rejected.
SAFE_OPERATORS = {
    ast.Add: operator.add,        # +
    ast.Sub: operator.sub,        # -
    ast.Mult: operator.mul,       # *
    ast.Div: operator.truediv,    # /
    ast.Pow: operator.pow,        # **
    ast.FloorDiv: operator.floordiv,  # //
    ast.Mod: operator.mod,        # %
    ast.USub: operator.neg,       # unary minus (e.g., -5)
}


def safe_eval(expression: str) -> float:
    """
    Safely evaluate a math expression by parsing it into an AST (Abstract
    Syntax Tree) and only allowing numeric operations.

    How this works:
    1. ast.parse() converts "245 * 18" into a tree structure
    2. We walk the tree node by node
    3. If we see a number → return it
    4. If we see an operator (+, -, *, /) → apply it
    5. If we see ANYTHING else (function call, import, variable) → raise error

    This is the same approach used by many production systems to safely
    evaluate user-provided math expressions.
    """
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree.body)


def _eval_node(node):
    """Recursively evaluate an AST node."""

    # A plain number like 42 or 3.14
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    # A binary operation like 2 + 3 or 10 * 5
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return SAFE_OPERATORS[op_type](left, right)

    # A unary operation like -5
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        operand = _eval_node(node.operand)
        return SAFE_OPERATORS[op_type](operand)

    # Anything else (function calls, variables, imports) is rejected
    raise ValueError(
        f"Unsupported expression type: {type(node).__name__}. "
        "Only numeric operations are allowed."
    )


class CalculatorTool(BaseTool):
    """
    Tool for evaluating mathematical expressions.

    Usage by the LLM:
        TOOL_CALL: calculator | INPUT: 245 * 18 + 32
    """

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Use this tool to evaluate mathematical expressions. "
            "Send a valid math expression like '245 * 18 + 32' or "
            "'(100 / 4) ** 2'. Supports +, -, *, /, **, //, and %."
        )

    def run(self, tool_input: str) -> str:
        try:
            expression = tool_input.strip()
            result = safe_eval(expression)

            # If the result is a whole number, show it without decimals
            # 245 * 18 = 4410, not 4410.0
            if isinstance(result, float) and result == int(result):
                result = int(result)

            return f"{expression} = {result}"
        except ZeroDivisionError:
            return "Error: Division by zero is not allowed."
        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            return f"Error evaluating expression: {str(e)}"
