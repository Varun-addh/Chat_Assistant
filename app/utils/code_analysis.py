from __future__ import annotations

import ast
from typing import Dict, Any, List


def _find_calls(node: ast.AST) -> List[str]:
	calls: List[str] = []
	for n in ast.walk(node):
		if isinstance(n, ast.Call):
			func = n.func
			if isinstance(func, ast.Name):
				calls.append(func.id)
			elif isinstance(func, ast.Attribute):
				calls.append(func.attr)
	return calls


def analyze_python_code(code: str) -> Dict[str, Any]:
	"""Lightweight static analysis for Python solutions.

	Detects common approach signals used in interviews.
	This is conservative and resilient to errors; on parse failure it returns empty signals.
	"""
	result: Dict[str, Any] = {
		"parse_ok": False,
		"functions": [],
		"imports": [],
		"uses_recursion": False,
		"uses_memoization": False,
		"uses_lru_cache": False,
		"uses_dp_table": False,
		"uses_iterative_stack": False,
		"nested_loop_depth": 0,
	}
	try:
		tree = ast.parse(code)
		result["parse_ok"] = True
	except Exception:
		return result

	# Basic metadata
	for node in tree.body:
		if isinstance(node, ast.FunctionDef):
			result["functions"].append(node.name)
		elif isinstance(node, ast.Import):
			for alias in node.names:
				result["imports"].append(alias.name)
		elif isinstance(node, ast.ImportFrom):
			mod = node.module or ""
			for alias in node.names:
				result["imports"].append(f"{mod}.{alias.name}" if mod else alias.name)

	# Recursion and memoization
	for node in ast.walk(tree):
		if isinstance(node, ast.FunctionDef):
			# Self-call indicates recursion
			for call in _find_calls(node):
				if call == node.name:
					result["uses_recursion"] = True
			# Detect manual memo dict usage: if 'memo' in arguments or assignments
			for arg in node.args.args:
				if arg.arg.lower() in {"memo", "cache", "dp"}:
					result["uses_memoization"] = True
			for n in ast.walk(node):
				if isinstance(n, ast.Assign):
					for t in n.targets:
						if isinstance(t, ast.Name) and t.id.lower() in {"memo", "cache", "dp"}:
							result["uses_memoization"] = True

	# lru_cache decorator
	for n in ast.walk(tree):
		if isinstance(n, ast.FunctionDef) and n.decorator_list:
			for d in n.decorator_list:
				if isinstance(d, ast.Name) and d.id == "lru_cache":
					result["uses_lru_cache"] = True
					result["uses_memoization"] = True
				elif isinstance(d, ast.Attribute) and d.attr == "lru_cache":
					result["uses_lru_cache"] = True
					result["uses_memoization"] = True

	# DP table usage: presence of list-of-lists or dict assigned and indexed by variables
	for n in ast.walk(tree):
		if isinstance(n, ast.Subscript):
			# Heuristic: if target is a Name and slice is Name or Tuple, count as dp table usage
			value = n.value
			if isinstance(value, ast.Name) and isinstance(n.slice, (ast.Slice, ast.Index, ast.Tuple)):
				result["uses_dp_table"] = True

	# Iterative stack usage
	for n in ast.walk(tree):
		if isinstance(n, ast.Assign):
			for t in n.targets:
				if isinstance(t, ast.Name) and t.id.lower() in {"stack", "queue"}:
					result["uses_iterative_stack"] = True

	# Nested loop depth (rough estimate)
	def loop_depth(node: ast.AST) -> int:
		max_d = 0
		for child in ast.iter_child_nodes(node):
			d = loop_depth(child)
			if isinstance(child, (ast.For, ast.While)):
				max_d = max(max_d, 1 + d)
			else:
				max_d = max(max_d, d)
		return max_d

	result["nested_loop_depth"] = loop_depth(tree)
	return result


