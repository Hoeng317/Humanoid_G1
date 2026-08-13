import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def test_live_task_contract_documentation_placeholder():
    """The actual live check is `g1.sh inspect --live`; kept marked to avoid nested Kit startup."""
    assert True

