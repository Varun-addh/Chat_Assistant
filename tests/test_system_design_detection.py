from app.services.llm import is_system_design_question


def test_is_system_design_question_true_for_obvious_cases() -> None:
	assert is_system_design_question("System design: build a URL shortener")
	assert is_system_design_question("High level design for Twitter timeline")
	assert is_system_design_question("Design a rate limiter")


def test_is_system_design_question_false_for_other_intents() -> None:
	assert not is_system_design_question("Draw an ER diagram for a blog database schema")
	assert not is_system_design_question("Design an algorithm to reverse a linked list")
	assert not is_system_design_question("Create a UI wireframe for a shopping app")
