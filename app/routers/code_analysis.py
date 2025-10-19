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
    variables_changed: List[VariableState]
    data_flow: List[str]
    memory_operations: List[str]
    execution_context: str


class CodeAnalysisResponse(BaseModel):
    execution_steps: List[ExecutionStep]
    variable_timeline: Dict[str, List[VariableState]]
    execution_flow: List[str]
    memory_map: Dict[str, Any]
    complexity_analysis: Dict[str, Any]
    data_flow_diagram: Optional[str] = None
    execution_flow_diagram: Optional[str] = None
    summary: str
    beginner_explanation: str
    professional_insights: str


@router.post("/analyze", response_model=CodeAnalysisResponse)
async def analyze_code(
    request: CodeAnalysisRequest,
    _: None = Depends(verify_api_key)
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
        
        # Convert variable timeline
        variable_timeline = {}
        for var_name, states in analysis.variable_timeline.items():
            variable_timeline[var_name] = [
                VariableState(
                    name=state.name,
                    value=state.value,
                    type=state.type,
                    scope=state.scope,
                    line_number=state.line_number,
                    memory_location=state.memory_location
                )
                for state in states
            ]
        
        # Generate explanations
        summary = _generate_summary(analysis)
        beginner_explanation = _generate_beginner_explanation(analysis)
        professional_insights = _generate_professional_insights(analysis)
        
        response = CodeAnalysisResponse(
            execution_steps=execution_steps,
            variable_timeline=variable_timeline,
            execution_flow=analysis.execution_flow,
            memory_map=analysis.memory_map,
            complexity_analysis=analysis.complexity_analysis,
            data_flow_diagram=analysis.data_flow_diagram if request.include_diagrams else None,
            execution_flow_diagram=analysis.execution_flow_diagram if request.include_diagrams else None,
            summary=summary,
            beginner_explanation=beginner_explanation,
            professional_insights=professional_insights
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/analyze/step-by-step")
async def analyze_step_by_step(
    request: CodeAnalysisRequest,
    _: None = Depends(verify_api_key)
):
    """
    Interactive step-by-step code analysis with detailed explanations.
    Perfect for learning and debugging.
    """
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="Empty code provided")
    
    try:
        analysis = code_analyzer.analyze_code(request.code, request.language)
        
        # Create detailed step-by-step breakdown
        step_by_step = []
        for i, step in enumerate(analysis.execution_steps):
            step_detail = {
                "step_number": i + 1,
                "line_number": step.line_number,
                "line_content": step.line_content,
                "step_type": step.step_type.value,
                "description": step.description,
                "what_happens": _explain_what_happens(step),
                "why_important": _explain_why_important(step),
                "variables_changed": [
                    {
                        "name": var.name,
                        "old_value": _get_previous_value(var.name, analysis.variable_timeline, step.line_number),
                        "new_value": var.value,
                        "type": var.type,
                        "memory_location": var.memory_location
                    }
                    for var in step.variables_changed
                ],
                "data_flow": step.data_flow,
                "memory_operations": step.memory_operations,
                "execution_context": step.execution_context,
                "visual_diagram": _generate_step_diagram(step, i + 1) if request.include_diagrams else None
            }
            step_by_step.append(step_detail)
        
        return JSONResponse(content={
            "code": request.code,
            "language": request.language,
            "total_steps": len(step_by_step),
            "step_by_step_analysis": step_by_step,
            "overall_complexity": analysis.complexity_analysis,
            "execution_summary": {
                "total_lines_executed": len(analysis.execution_steps),
                "unique_variables": len(analysis.variable_timeline),
                "memory_operations": sum(len(step.memory_operations) for step in analysis.execution_steps),
                "control_flow_changes": len([s for s in analysis.execution_steps if s.step_type.value in ["conditional", "loop"]])
            }
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


def _generate_beginner_explanation(analysis: CodeAnalysis) -> str:
    """Generate beginner-friendly explanation."""
    explanation = "## How This Code Works (Beginner Guide)\n\n"
    
    if not analysis.execution_steps:
        return explanation + "This code has syntax errors and cannot be executed."
    
    explanation += "### Step-by-Step Breakdown:\n\n"
    
    for i, step in enumerate(analysis.execution_steps[:5]):  # Show first 5 steps
        explanation += f"**Step {i+1} (Line {step.line_number}):** {step.description}\n"
        explanation += f"- What happens: {step.line_content}\n"
        if step.variables_changed:
            explanation += f"- Variables changed: {', '.join([v.name for v in step.variables_changed])}\n"
        explanation += "\n"
    
    if len(analysis.execution_steps) > 5:
        explanation += f"... and {len(analysis.execution_steps) - 5} more steps\n\n"
    
    explanation += "### Key Concepts:\n"
    explanation += "- **Variables**: Store data that can change during execution\n"
    explanation += "- **Control Flow**: How the program decides what to do next\n"
    explanation += "- **Memory**: Where data is stored (stack for local variables, heap for complex objects)\n"
    
    return explanation


def _generate_professional_insights(analysis: CodeAnalysis) -> str:
    """Generate professional-level insights."""
    insights = "## Professional Analysis\n\n"
    
    # Complexity analysis
    complexity = analysis.complexity_analysis
    insights += f"### Complexity Analysis\n"
    insights += f"- **Time Complexity**: {complexity.get('time_complexity', 'Unknown')}\n"
    insights += f"- **Space Complexity**: {complexity.get('space_complexity', 'Unknown')}\n"
    insights += f"- **Loop Depth**: {complexity.get('loop_depth', 0)}\n\n"
    
    # Memory analysis
    insights += "### Memory Management\n"
    stack_vars = sum(1 for timeline in analysis.variable_timeline.values() 
                    for state in timeline if state.memory_location == "stack")
    insights += f"- **Stack Variables**: {stack_vars}\n"
    insights += f"- **Heap Objects**: {len(analysis.memory_map.get('heap', {}))}\n\n"
    
    # Performance insights
    insights += "### Performance Insights\n"
    if complexity.get('loop_depth', 0) > 2:
        insights += "- ⚠️ Nested loops detected - consider optimization\n"
    if any('recursive' in str(step.description) for step in analysis.execution_steps):
        insights += "- 🔄 Recursive implementation - watch for stack overflow\n"
    if len(analysis.variable_timeline) > 10:
        insights += "- 📊 High variable count - consider refactoring\n"
    
    return insights


def _explain_what_happens(step) -> str:
    """Explain what happens in a specific step."""
    if step.step_type.value == "variable_assignment":
        return f"Values are assigned to variables: {', '.join([v.name for v in step.variables_changed])}"
    elif step.step_type.value == "conditional":
        return "A condition is evaluated to determine which code path to take"
    elif step.step_type.value == "loop":
        return "A loop begins, which will repeat code multiple times"
    elif step.step_type.value == "function_call":
        return "A function is called, transferring control to that function"
    elif step.step_type.value == "return":
        return "A value is returned from the current function"
    else:
        return "Code is executed and evaluated"


def _explain_why_important(step) -> str:
    """Explain why this step is important."""
    if step.step_type.value == "variable_assignment":
        return "Variables store data that the program needs to remember and use later"
    elif step.step_type.value == "conditional":
        return "Conditionals allow the program to make decisions and adapt its behavior"
    elif step.step_type.value == "loop":
        return "Loops allow the program to repeat actions efficiently without duplicating code"
    elif step.step_type.value == "function_call":
        return "Function calls organize code into reusable, modular pieces"
    elif step.step_type.value == "return":
        return "Return statements provide results back to the calling code"
    else:
        return "This step contributes to the overall program logic and execution"


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
