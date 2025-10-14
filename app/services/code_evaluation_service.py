from __future__ import annotations

from typing import Tuple, List
import ast
import json

from app.services.llm_service import llm_service


def _analyze_python_ast(code: str) -> dict:
	"""Lightweight static signals from Python source using AST."""
	try:
		tree = ast.parse(code)
	except Exception:
		return {
			"uses_recursion": False,
			"uses_memoization": False,
			"uses_dynamic_programming": False,
			"loop_nesting_depth": 0,
			"uses_slicing_heavily": (code.count(":") > 10),
			"uses_list_or_set_comprehension": ("[" in code and "] for" in code) or ("{" in code and " for" in code),
			"function_count": code.count("def "),
			"comment_density": _comment_density(code),
			"estimated_time_complexity_hint": None,
		}

	max_loop_depth = 0
	current_depth = 0
	uses_recursion = False
	uses_memo = False
	uses_dp = False
	comp_used = False

	func_defs: List[str] = []

	class Visitor(ast.NodeVisitor):
		def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
			func_defs.append(node.name)
			self.generic_visit(node)

		def visit_For(self, node: ast.For) -> None:
			nonlocal current_depth, max_loop_depth
			current_depth += 1
			max_loop_depth = max(max_loop_depth, current_depth)
			self.generic_visit(node)
			current_depth -= 1

		def visit_While(self, node: ast.While) -> None:
			nonlocal current_depth, max_loop_depth
			current_depth += 1
			max_loop_depth = max(max_loop_depth, current_depth)
			self.generic_visit(node)
			current_depth -= 1

		def visit_Call(self, node: ast.Call) -> None:
			nonlocal uses_recursion, uses_memo
			try:
				func_name = getattr(node.func, "id", None) or getattr(getattr(node.func, "attr", None), "id", None)
			except Exception:
				func_name = None
			if func_name and func_name in func_defs:
				uses_recursion = True
			if isinstance(node.func, ast.Attribute) and node.func.attr == "lru_cache":
				uses_memo = True
			self.generic_visit(node)

		def visit_ListComp(self, node: ast.ListComp) -> None:
			nonlocal comp_used
			comp_used = True
			self.generic_visit(node)

		def visit_SetComp(self, node: ast.SetComp) -> None:
			nonlocal comp_used
			comp_used = True
			self.generic_visit(node)

		def visit_Subscript(self, node: ast.Subscript) -> None:
			nonlocal uses_dp
			# Heuristic: nested subscripts often indicate DP tables/matrices
			if isinstance(node.value, ast.Subscript):
				uses_dp = True
			self.generic_visit(node)

	Visitor().visit(tree)

	# Very rough slicing heuristic
	uses_slicing = code.count(":") > 10

	# Simple time complexity hint
	hint = None
	if max_loop_depth >= 2 and not uses_recursion:
		hint = "Likely O(n^2) due to nested loops"
	elif uses_recursion and not uses_memo:
		hint = "Recursive without memoization; may be exponential"
	elif uses_recursion and uses_memo:
		hint = "Recursive with memoization; likely polynomial"

	return {
		"uses_recursion": uses_recursion,
		"uses_memoization": uses_memo,
		"uses_dynamic_programming": uses_dp,
		"loop_nesting_depth": max_loop_depth,
		"uses_slicing_heavily": uses_slicing,
		"uses_list_or_set_comprehension": comp_used,
		"function_count": len(func_defs),
		"comment_density": _comment_density(code),
		"estimated_time_complexity_hint": hint,
	}


def _comment_density(code: str) -> float:
	lines = [l for l in code.splitlines()]
	if not lines:
		return 0.0
	comment_lines = sum(1 for l in lines if l.strip().startswith("#"))
	code_lines = sum(1 for l in lines if l.strip() and not l.strip().startswith("#"))
	if code_lines == 0:
		return 0.0
	return round(min(1.0, comment_lines / max(1, code_lines)), 3)


async def evaluate_code(problem: str | None, code: str, language: str, conversation_context: str = "") -> Tuple[str, dict]:
	"""Returns (llm_text, static_signals)."""
	lang = (language or "").lower().strip() or "python"
	if lang == "py" or lang == "python":
		static = _analyze_python_ast(code)
	else:
		# Fallback lightweight heuristics for non-Python
		static = {
			"uses_recursion": ("def " in code and "(" in code and ")" in code and "return" in code and ("recurs" in code.lower() or "self(" in code or "function" in code)),
			"uses_memoization": ("memo" in code.lower() or "cache" in code.lower()),
			"uses_dynamic_programming": ("dp" in code.lower() or "table" in code.lower()),
			"loop_nesting_depth": code.count("for ") + code.count("while "),
			"uses_slicing_heavily": code.count(":") > 10,
			"uses_list_or_set_comprehension": ("for (" in code and ") {" in code) or ("=>" in code and "map" in code),
			"function_count": code.lower().count("function") + code.count("def "),
			"comment_density": _comment_density(code),
			"estimated_time_complexity_hint": None,
		}

	critique = await llm_service.evaluate_code_with_critique(problem or "", code, lang, conversation_context)
	return critique, static


