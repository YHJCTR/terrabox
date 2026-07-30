from terrabox.agent.eval_modes import get_eval_mode_runner, list_eval_modes
from terrabox.agent.eval_modes.category import CategoryEvalRunner


def test_category_scoped_resolves_category_eval_runner():
    runner = get_eval_mode_runner("category_scoped")

    assert isinstance(runner, CategoryEvalRunner)
    assert runner is get_eval_mode_runner("category")
    assert "category_scoped" in list_eval_modes()
