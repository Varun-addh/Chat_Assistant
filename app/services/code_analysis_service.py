from __future__ import annotations

import ast
import re
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ExecutionStepType(Enum):
    VARIABLE_ASSIGNMENT = "variable_assignment"
    FUNCTION_CALL = "function_call"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    RETURN = "return"
    EXPRESSION = "expression"
    IMPORT = "import"
    FUNCTION_DEF = "function_def"


@dataclass
class VariableState:
    name: str
    value: Any
    type: str
    scope: str
    line_number: int
    memory_location: str  # stack, heap, etc.


@dataclass
class ExecutionStep:
    line_number: int
    line_content: str
    step_type: ExecutionStepType
    description: str
    variables_changed: List[VariableState]
    data_flow: List[str]  # How data moves
    memory_operations: List[str]  # Stack/heap operations
    execution_context: str  # Function scope, loop iteration, etc.


@dataclass
class CodeAnalysis:
    execution_steps: List[ExecutionStep]
    variable_timeline: Dict[str, List[VariableState]]
    execution_flow: List[str]
    memory_map: Dict[str, Any]
    complexity_analysis: Dict[str, Any]
    data_flow_diagram: str  # Mermaid diagram
    execution_flow_diagram: str  # Mermaid diagram


class CodeAnalyzer:
    """Revolutionary code analysis system that provides deep insights into program execution."""
    
    def __init__(self):
        self.variable_tracker = {}
        self.execution_context = []
        self.memory_map = {"stack": {}, "heap": {}}
        self.line_number = 0
        
    def analyze_code(self, code: str, language: str = "python") -> CodeAnalysis:
        """Perform comprehensive code analysis with line-by-line execution tracking."""
        if language.lower() in ["python", "py"]:
            return self._analyze_python_code(code)
        else:
            return self._analyze_generic_code(code)
    
    def _analyze_python_code(self, code: str) -> CodeAnalysis:
        """Deep analysis of Python code using AST and execution simulation."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return self._create_error_analysis(code, "Syntax Error")
        
        execution_steps = []
        variable_timeline = {}
        
        # First pass: collect all variables and their initial states
        self._collect_variables(tree, code)
        
        # Second pass: simulate execution step by step
        self._simulate_execution(tree, code, execution_steps, variable_timeline)
        
        # Generate diagrams
        data_flow_diagram = self._generate_data_flow_diagram(execution_steps)
        execution_flow_diagram = self._generate_execution_flow_diagram(execution_steps)
        
        # Analyze complexity
        complexity_analysis = self._analyze_complexity(tree, code)
        
        return CodeAnalysis(
            execution_steps=execution_steps,
            variable_timeline=variable_timeline,
            execution_flow=self._extract_execution_flow(execution_steps),
            memory_map=self.memory_map,
            complexity_analysis=complexity_analysis,
            data_flow_diagram=data_flow_diagram,
            execution_flow_diagram=execution_flow_diagram
        )
    
    def _collect_variables(self, tree: ast.AST, code: str) -> None:
        """Collect all variables and their initial states."""
        lines = code.split('\n')
        
        class VariableCollector(ast.NodeVisitor):
            def __init__(self, analyzer):
                self.analyzer = analyzer
                
            def visit_Assign(self, node: ast.Assign) -> None:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        self.analyzer.variable_tracker[var_name] = {
                            'first_seen': node.lineno,
                            'assignments': [],
                            'scope': self.analyzer._get_current_scope()
                        }
                self.generic_visit(node)
                
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self.analyzer.execution_context.append(f"function_{node.name}")
                self.generic_visit(node)
                self.analyzer.execution_context.pop()
                
            def visit_For(self, node: ast.For) -> None:
                self.analyzer.execution_context.append("for_loop")
                self.generic_visit(node)
                self.analyzer.execution_context.pop()
                
            def visit_While(self, node: ast.While) -> None:
                self.analyzer.execution_context.append("while_loop")
                self.generic_visit(node)
                self.analyzer.execution_context.pop()
        
        VariableCollector(self).visit(tree)
    
    def _simulate_execution(self, tree: ast.AST, code: str, execution_steps: List[ExecutionStep], variable_timeline: Dict[str, List[VariableState]]) -> None:
        """Simulate program execution line by line."""
        lines = code.split('\n')
        
        class ExecutionSimulator(ast.NodeVisitor):
            def __init__(self, analyzer, steps, timeline):
                self.analyzer = analyzer
                self.execution_steps = steps
                self.variable_timeline = timeline
                self.current_line = 0
                
            def visit_Assign(self, node: ast.Assign) -> None:
                self._process_assignment(node, lines[node.lineno - 1])
                self.generic_visit(node)
                
            def visit_Expr(self, node: ast.Expr) -> None:
                self._process_expression(node, lines[node.lineno - 1])
                self.generic_visit(node)
                
            def visit_If(self, node: ast.If) -> None:
                self._process_conditional(node, lines[node.lineno - 1])
                self.generic_visit(node)
                
            def visit_For(self, node: ast.For) -> None:
                self._process_loop(node, lines[node.lineno - 1], "for")
                self.generic_visit(node)
                
            def visit_While(self, node: ast.While) -> None:
                self._process_loop(node, lines[node.lineno - 1], "while")
                self.generic_visit(node)
                
            def visit_Return(self, node: ast.Return) -> None:
                self._process_return(node, lines[node.lineno - 1])
                self.generic_visit(node)
                
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self._process_function_def(node, lines[node.lineno - 1])
                self.generic_visit(node)
                
            def _process_assignment(self, node: ast.Assign, line_content: str) -> None:
                variables_changed = []
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        # Simulate value assignment
                        value = self._evaluate_expression(node.value)
                        var_state = VariableState(
                            name=var_name,
                            value=value,
                            type=type(value).__name__,
                            scope=self.analyzer._get_current_scope(),
                            line_number=node.lineno,
                            memory_location="stack"
                        )
                        variables_changed.append(var_state)
                        
                        # Update timeline
                        if var_name not in self.variable_timeline:
                            self.variable_timeline[var_name] = []
                        self.variable_timeline[var_name].append(var_state)
                
                step = ExecutionStep(
                    line_number=node.lineno,
                    line_content=line_content.strip(),
                    step_type=ExecutionStepType.VARIABLE_ASSIGNMENT,
                    description=f"Variable assignment: {', '.join([v.name for v in variables_changed])}",
                    variables_changed=variables_changed,
                    data_flow=[f"Value flows from expression to {v.name}" for v in variables_changed],
                    memory_operations=["Stack allocation"] if variables_changed else [],
                    execution_context=self.analyzer._get_current_scope()
                )
                self.execution_steps.append(step)
                
            def _process_expression(self, node: ast.Expr, line_content: str) -> None:
                step = ExecutionStep(
                    line_number=node.lineno,
                    line_content=line_content.strip(),
                    step_type=ExecutionStepType.EXPRESSION,
                    description="Expression evaluation",
                    variables_changed=[],
                    data_flow=["Expression result computed"],
                    memory_operations=["Temporary value on stack"],
                    execution_context=self.analyzer._get_current_scope()
                )
                self.execution_steps.append(step)
                
            def _process_conditional(self, node: ast.If, line_content: str) -> None:
                step = ExecutionStep(
                    line_number=node.lineno,
                    line_content=line_content.strip(),
                    step_type=ExecutionStepType.CONDITIONAL,
                    description="Conditional branch evaluation",
                    variables_changed=[],
                    data_flow=["Condition evaluated, branch selected"],
                    memory_operations=["Condition result on stack"],
                    execution_context=self.analyzer._get_current_scope()
                )
                self.execution_steps.append(step)
                
            def _process_loop(self, node: ast.AST, line_content: str, loop_type: str) -> None:
                step = ExecutionStep(
                    line_number=node.lineno,
                    line_content=line_content.strip(),
                    step_type=ExecutionStepType.LOOP,
                    description=f"{loop_type.capitalize()} loop initialization",
                    variables_changed=[],
                    data_flow=["Loop condition evaluated"],
                    memory_operations=["Loop counter on stack"],
                    execution_context=self.analyzer._get_current_scope()
                )
                self.execution_steps.append(step)
                
            def _process_return(self, node: ast.Return, line_content: str) -> None:
                step = ExecutionStep(
                    line_number=node.lineno,
                    line_content=line_content.strip(),
                    step_type=ExecutionStepType.RETURN,
                    description="Function return",
                    variables_changed=[],
                    data_flow=["Return value computed and passed back"],
                    memory_operations=["Return value on stack"],
                    execution_context=self.analyzer._get_current_scope()
                )
                self.execution_steps.append(step)
                
            def _process_function_def(self, node: ast.FunctionDef, line_content: str) -> None:
                step = ExecutionStep(
                    line_number=node.lineno,
                    line_content=line_content.strip(),
                    step_type=ExecutionStepType.FUNCTION_DEF,
                    description=f"Function definition: {node.name}",
                    variables_changed=[],
                    data_flow=["Function object created"],
                    memory_operations=["Function object on heap"],
                    execution_context="global"
                )
                self.execution_steps.append(step)
                
            def _evaluate_expression(self, node: ast.AST) -> Any:
                """Simplified expression evaluation for demonstration."""
                if isinstance(node, ast.Constant):
                    return node.value
                elif isinstance(node, ast.Name):
                    return f"<variable:{node.id}>"
                elif isinstance(node, ast.BinOp):
                    left = self._evaluate_expression(node.left)
                    right = self._evaluate_expression(node.right)
                    return f"<operation:{type(node.op).__name__}>"
                elif isinstance(node, ast.List):
                    return f"<list:{len(node.elts)}_elements>"
                elif isinstance(node, ast.Call):
                    return f"<function_call:{getattr(node.func, 'id', 'unknown')}>"
                else:
                    return f"<expression:{type(node).__name__}>"
        
        ExecutionSimulator(self, execution_steps, variable_timeline).visit(tree)
    
    def _get_current_scope(self) -> str:
        """Get current execution scope."""
        if not self.execution_context:
            return "global"
        return ".".join(self.execution_context)
    
    def _extract_execution_flow(self, steps: List[ExecutionStep]) -> List[str]:
        """Extract high-level execution flow."""
        flow = []
        for step in steps:
            if step.step_type == ExecutionStepType.FUNCTION_DEF:
                flow.append(f"Define function: {step.description}")
            elif step.step_type == ExecutionStepType.VARIABLE_ASSIGNMENT:
                flow.append(f"Assign: {step.description}")
            elif step.step_type == ExecutionStepType.CONDITIONAL:
                flow.append(f"Conditional: {step.description}")
            elif step.step_type == ExecutionStepType.LOOP:
                flow.append(f"Loop: {step.description}")
            elif step.step_type == ExecutionStepType.RETURN:
                flow.append(f"Return: {step.description}")
        return flow
    
    def _generate_data_flow_diagram(self, steps: List[ExecutionStep]) -> str:
        """Generate Mermaid diagram showing data flow."""
        diagram_lines = [
            "flowchart TD",
            "    Start([Program Start])",
        ]
        
        for i, step in enumerate(steps):
            if step.step_type == ExecutionStepType.VARIABLE_ASSIGNMENT:
                for var in step.variables_changed:
                    diagram_lines.append(f"    L{step.line_number}[Line {step.line_number}: {var.name} = {var.value}]")
                    if i > 0:
                        diagram_lines.append(f"    L{steps[i-1].line_number} --> L{step.line_number}")
        
        diagram_lines.append("    End([Program End])")
        if steps:
            diagram_lines.append(f"    L{steps[-1].line_number} --> End")
        
        return "\n".join(diagram_lines)
    
    def _generate_execution_flow_diagram(self, steps: List[ExecutionStep]) -> str:
        """Generate Mermaid diagram showing execution flow."""
        diagram_lines = [
            "flowchart TD",
            "    Start([Program Start])",
        ]
        
        for i, step in enumerate(steps):
            step_id = f"Step{i+1}"
            step_label = f"Line {step.line_number}: {step.step_type.value.replace('_', ' ').title()}"
            diagram_lines.append(f"    {step_id}[{step_label}]")
            
            if i == 0:
                diagram_lines.append(f"    Start --> {step_id}")
            else:
                diagram_lines.append(f"    Step{i} --> {step_id}")
        
        if steps:
            diagram_lines.append(f"    Step{len(steps)} --> End([Program End])")
        else:
            diagram_lines.append("    Start --> End([Program End])")
        
        return "\n".join(diagram_lines)
    
    def _analyze_complexity(self, tree: ast.AST, code: str) -> Dict[str, Any]:
        """Analyze time and space complexity."""
        complexity = {
            "time_complexity": "O(1)",
            "space_complexity": "O(1)",
            "loop_depth": 0,
            "recursive_calls": 0,
            "data_structures": [],
            "algorithm_patterns": []
        }
        
        # Analyze loops
        max_loop_depth = 0
        current_depth = 0
        
        class ComplexityAnalyzer(ast.NodeVisitor):
            def visit_For(self, node: ast.For):
                nonlocal current_depth, max_loop_depth
                current_depth += 1
                max_loop_depth = max(max_loop_depth, current_depth)
                self.generic_visit(node)
                current_depth -= 1
                
            def visit_While(self, node: ast.While):
                nonlocal current_depth, max_loop_depth
                current_depth += 1
                max_loop_depth = max(max_loop_depth, current_depth)
                self.generic_visit(node)
                current_depth -= 1
        
        ComplexityAnalyzer().visit(tree)
        
        complexity["loop_depth"] = max_loop_depth
        
        # Determine time complexity based on loop depth
        if max_loop_depth == 0:
            complexity["time_complexity"] = "O(1)"
        elif max_loop_depth == 1:
            complexity["time_complexity"] = "O(n)"
        elif max_loop_depth == 2:
            complexity["time_complexity"] = "O(n²)"
        else:
            complexity["time_complexity"] = f"O(n^{max_loop_depth})"
        
        return complexity
    
    def _analyze_generic_code(self, code: str) -> CodeAnalysis:
        """Fallback analysis for non-Python code."""
        lines = code.split('\n')
        execution_steps = []
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('//') or line.startswith('#'):
                continue
                
            step_type = ExecutionStepType.EXPRESSION
            description = "Code execution"
            
            if '=' in line and not '==' in line:
                step_type = ExecutionStepType.VARIABLE_ASSIGNMENT
                description = "Variable assignment"
            elif line.startswith('if ') or line.startswith('while ') or line.startswith('for '):
                step_type = ExecutionStepType.CONDITIONAL
                description = "Control flow"
            elif line.startswith('return '):
                step_type = ExecutionStepType.RETURN
                description = "Return statement"
                
            step = ExecutionStep(
                line_number=i,
                line_content=line,
                step_type=step_type,
                description=description,
                variables_changed=[],
                data_flow=["Generic data flow"],
                memory_operations=["Generic memory operation"],
                execution_context="global"
            )
            execution_steps.append(step)
        
        return CodeAnalysis(
            execution_steps=execution_steps,
            variable_timeline={},
            execution_flow=[step.description for step in execution_steps],
            memory_map={"stack": {}, "heap": {}},
            complexity_analysis={"time_complexity": "Unknown", "space_complexity": "Unknown"},
            data_flow_diagram="flowchart TD\n    Start([Program Start])\n    End([Program End])\n    Start --> End",
            execution_flow_diagram="flowchart TD\n    Start([Program Start])\n    End([Program End])\n    Start --> End"
        )
    
    def _create_error_analysis(self, code: str, error: str) -> CodeAnalysis:
        """Create analysis for code with errors."""
        return CodeAnalysis(
            execution_steps=[],
            variable_timeline={},
            execution_flow=[f"Error: {error}"],
            memory_map={"stack": {}, "heap": {}},
            complexity_analysis={"error": error},
            data_flow_diagram="flowchart TD\n    Error([Syntax Error])\n    Error",
            execution_flow_diagram="flowchart TD\n    Error([Syntax Error])\n    Error"
        )


# Global analyzer instance
code_analyzer = CodeAnalyzer()
