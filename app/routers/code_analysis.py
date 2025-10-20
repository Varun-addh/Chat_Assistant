from __future__ import annotations

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json

from app.services.code_analysis_service import code_analyzer, CodeAnalysis
from app.utils.security import verify_api_key


router = APIRouter()


class CodeAnalysisRequest(BaseModel):
    code: str
    language: str = "python"
    include_diagrams: bool = True
    include_memory_analysis: bool = True
    include_complexity_analysis: bool = True


class VariableState(BaseModel):
    name: str
    value: Any
    type: str
    scope: str
    line_number: int
    memory_location: str


class ExecutionStep(BaseModel):
    line_number: int
    line_content: str
    step_type: str
    description: str


class CodeAnalysisResponse(BaseModel):
    execution_steps: List[ExecutionStep]
    execution_flow: List[str]
    complexity_analysis: Dict[str, Any]
    summary: str


@router.post("/analyze", response_model=CodeAnalysisResponse)
async def analyze_code(
    request: CodeAnalysisRequest
):
    """
    Revolutionary code analysis endpoint that provides:
    - Line-by-line execution tracking
    - Variable state changes
    - Data flow visualization
    - Memory operations
    - Complexity analysis
    - Interactive diagrams
    """
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="Empty code provided")
    
    try:
        # Perform comprehensive analysis
        analysis = code_analyzer.analyze_code(request.code, request.language)
        
        # Convert to response format (without variables/diagrams)
        execution_steps = []
        for step in analysis.execution_steps:
            execution_steps.append(ExecutionStep(
                line_number=step.line_number,
                line_content=step.line_content,
                step_type=step.step_type.value,
                description=step.description
            ))
        
        # Skip variable timeline (removed per new contract)
        
        # Generate summary only
        summary = _generate_summary(analysis)
        
        response = CodeAnalysisResponse(
            execution_steps=execution_steps,
            execution_flow=analysis.execution_flow,
            complexity_analysis=analysis.complexity_analysis,
            summary=summary
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/analyze/step-by-step")
async def analyze_step_by_step(
    request: CodeAnalysisRequest
):
    """
    Interactive step-by-step code analysis with per-line explanation and visual flow.
    """
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="Empty code provided")
    
    try:
        analysis = code_analyzer.analyze_code(request.code, request.language)
        
        # Create minimal step-by-step breakdown with explanation + visual flow
        step_by_step = []
        for i, step in enumerate(analysis.execution_steps):
            step_detail = {
                "step_number": i + 1,
                "line_number": step.line_number,
                "line_content": step.line_content,
                "explanation": _explain_line(step),
                "visual_flow": _generate_line_visual_flow(step, i + 1)
            }
            step_by_step.append(step_detail)
        
        return JSONResponse(content={
            "code": request.code,
            "language": request.language,
            "total_steps": len(step_by_step),
            "step_by_step_analysis": step_by_step,
            "overall_complexity": analysis.complexity_analysis
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step-by-step analysis failed: {str(e)}")


def _generate_summary(analysis: CodeAnalysis) -> str:
    """Generate a concise summary of the code analysis."""
    total_steps = len(analysis.execution_steps)
    variables = len(analysis.variable_timeline)
    complexity = analysis.complexity_analysis.get("time_complexity", "Unknown")
    
    return f"""
This code executes in {total_steps} steps, managing {variables} variables with {complexity} time complexity.
The program follows a {'recursive' if 'recursive' in str(analysis.execution_flow) else 'iterative'} approach,
with {'nested loops' if analysis.complexity_analysis.get('loop_depth', 0) > 1 else 'simple control flow'}.
    """.strip()


def _explain_line(step) -> str:
    """Provide a concise explanation of what the specific line does."""
    if step.step_type.value == "variable_assignment":
        return f"Assigns new value(s) to {', '.join([v.name for v in step.variables_changed])}."
    if step.step_type.value == "conditional":
        return "Evaluates a condition to choose the next execution path."
    if step.step_type.value == "loop":
        return "Checks loop condition and schedules the next iteration or exit."
    if step.step_type.value == "function_call":
        return "Invokes a function and transfers control to its body."
    if step.step_type.value == "return":
        return "Stops current function and returns a value to the caller."
    return "Executes the statement on this line."


def _generate_line_visual_flow(step, step_number: int) -> str:
    """Generate a visual flow (Mermaid) describing how this specific line executes."""
    if step.step_type.value == "variable_assignment":
        diagram = f"""
flowchart TD
    A[Step {step_number}: Execute Line]
    A --> B[Line {step.line_number}: {step.line_content}]
    B --> C[Evaluate RHS]
    C --> D[Assign LHS]
"""
        return diagram.strip()
    if step.step_type.value == "conditional":
        return f"""
flowchart TD
    A[Step {step_number}: Execute Line]
    A --> B[Line {step.line_number}: {step.line_content}]
    B --> C{Condition}
    C -->|true| D[Take True Branch]
    C -->|false| E[Take False Branch]
""".strip()
    if step.step_type.value == "loop":
        return f"""
flowchart TD
    A[Step {step_number}: Execute Line]
    A --> B[Line {step.line_number}: {step.line_content}]
    B --> C{Loop Condition}
    C -->|true| D[Run Body]
    D --> C
    C -->|false| E[Exit Loop]
""".strip()
    if step.step_type.value == "function_call":
        return f"""
flowchart TD
    A[Step {step_number}: Execute Line]
    A --> B[Line {step.line_number}: {step.line_content}]
    B --> C[Push Call Frame]
    C --> D[Enter Function]
""".strip()
    if step.step_type.value == "return":
        return f"""
flowchart TD
    A[Step {step_number}: Execute Line]
    A --> B[Line {step.line_number}: {step.line_content}]
    B --> C[Prepare Return Value]
    C --> D[Pop Call Frame]
    D --> E[Return to Caller]
""".strip()
    return f"""
flowchart TD
    A[Step {step_number}: Execute Line]
    A --> B[Line {step.line_number}: {step.line_content}]
""".strip()


def _get_previous_value(var_name: str, timeline: Dict, current_line: int) -> Any:
    # Deprecated: retained for backward-compat imports; no longer used in responses
    if var_name not in timeline:
        return None
    previous_states = [state for state in timeline[var_name] if state.line_number < current_line]
    if previous_states:
        return previous_states[-1].value
    return None
