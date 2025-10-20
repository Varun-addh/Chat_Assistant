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
    variables_changed: List[VariableState]
    data_flow: List[str]
    memory_operations: List[str]
    execution_context: str


class CodeAnalysisResponse(BaseModel):
    execution_steps: List[ExecutionStep]
    execution_flow: List[str]


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
        
        # Convert to response format
        execution_steps = []
        for step in analysis.execution_steps:
            variables_changed = []
            for var in step.variables_changed:
                variables_changed.append(VariableState(
                    name=var.name,
                    value=var.value,
                    type=var.type,
                    scope=var.scope,
                    line_number=var.line_number,
                    memory_location=var.memory_location
                ))
            
            execution_steps.append(ExecutionStep(
                line_number=step.line_number,
                line_content=step.line_content,
                step_type=step.step_type.value,
                description=step.description,
                variables_changed=variables_changed,
                data_flow=step.data_flow,
                memory_operations=step.memory_operations,
                execution_context=step.execution_context
            ))
        
        response = CodeAnalysisResponse(
            execution_steps=execution_steps,
            execution_flow=analysis.execution_flow,
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/analyze/step-by-step")
async def analyze_step_by_step(
    request: CodeAnalysisRequest
):
    """
    Step-by-step code analysis focused on concise per-line explanations and a visual flow.
    """
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="Empty code provided")
    
    try:
        analysis = code_analyzer.analyze_code(request.code, request.language)
        
        # Create simplified step-by-step breakdown
        step_by_step = []
        for i, step in enumerate(analysis.execution_steps):
            step_detail = {
                "step_number": i + 1,
                "line_number": step.line_number,
                "line_content": step.line_content,
                "line_explanation": _generate_line_explanation(step),
                "visual_flow": _generate_step_diagram(step, i + 1)
            }
            step_by_step.append(step_detail)

        return JSONResponse(content={
            "code": request.code,
            "language": request.language,
            "total_steps": len(step_by_step),
            "steps": step_by_step
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step-by-step analysis failed: {str(e)}")


def _generate_line_explanation(step) -> str:
    """Generate a concise, code-level explanation for a single executed line."""
    base = step.line_content.strip()
    if step.step_type.value == "variable_assignment":
        assigns = ", ".join([f"{v.name} ← {v.value}" for v in step.variables_changed])
        return f"Assign: {assigns}"
    if step.step_type.value == "function_call":
        return f"Call: {base}"
    if step.step_type.value == "return":
        return f"Return: {base.split('return', 1)[-1].strip()}"
    if step.step_type.value == "conditional":
        return f"Evaluate condition: {base}"
    if step.step_type.value == "loop":
        return f"Loop step: {base}"
    return f"Execute: {base}"


def _generate_summary(analysis: CodeAnalysis) -> str:
    """Kept for potential internal use; not returned to clients anymore."""
    return ""


def _generate_professional_insights(analysis: CodeAnalysis) -> str:
    return ""


def _explain_what_happens(step) -> str:
    return ""


def _explain_why_important(step) -> str:
    return ""


def _get_previous_value(var_name: str, timeline: Dict, current_line: int) -> Any:
    """Get the previous value of a variable before the current line."""
    if var_name not in timeline:
        return None
    
    previous_states = [state for state in timeline[var_name] if state.line_number < current_line]
    if previous_states:
        return previous_states[-1].value
    return None


def _generate_step_diagram(step, step_number: int) -> str:
    """Generate a Mermaid diagram for a specific step."""
    if step.step_type.value == "variable_assignment":
        diagram = f"""
flowchart TD
    A[Step {step_number}: Variable Assignment]
    A --> B[Line {step.line_number}: {step.line_content}]
    """
        for var in step.variables_changed:
            diagram += f"    B --> C[{var.name} = {var.value}]\n"
        return diagram.strip()
    
    elif step.step_type.value == "conditional":
        return f"""
flowchart TD
    A[Step {step_number}: Conditional]
    A --> B[Line {step.line_number}: {step.line_content}]
    B --> C[Evaluate Condition]
    C --> D[Choose Branch]
    """.strip()
    
    elif step.step_type.value == "loop":
        return f"""
flowchart TD
    A[Step {step_number}: Loop]
    A --> B[Line {step.line_number}: {step.line_content}]
    B --> C[Initialize Loop]
    C --> D[Check Condition]
    D --> E[Execute Body]
    E --> D
    """.strip()
    
    else:
        return f"""
flowchart TD
    A[Step {step_number}: {step.step_type.value.replace('_', ' ').title()}]
    A --> B[Line {step.line_number}: {step.line_content}]
    """.strip()
