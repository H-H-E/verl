# tests/test_pr_template.py
import pytest
from pathlib import Path

PROJECT_ROOT_PR_TEST = Path(__file__).resolve().parent.parent 
PR_TEMPLATE_PATH = PROJECT_ROOT_PR_TEST / ".github" / "PULL_REQUEST_TEMPLATE.md"

def test_pr_template_exists_and_has_key_headings():
    """
    Tests that .github/PULL_REQUEST_TEMPLATE.md exists and contains 
    key headings like "## Description", "## Type of change", and "## Checklist".
    """
    assert PR_TEMPLATE_PATH.exists(), f"{PR_TEMPLATE_PATH} does not exist."
    assert PR_TEMPLATE_PATH.is_file(), f"{PR_TEMPLATE_PATH} is not a file."

    try:
        content = PR_TEMPLATE_PATH.read_text(encoding="utf-8")
    except Exception as e: # pragma: no cover
        pytest.fail(f"Could not read {PR_TEMPLATE_PATH}: {e}")
        
    # Check for key headings using simple string 'in' check
    # This is robust against minor markdown formatting variations (e.g., extra spaces).
    
    expected_headings = [
        "## Description",
        "## Type of change",
        "## Checklist"
    ]
    
    missing_headings = []
    for heading in expected_headings:
        if heading not in content:
            missing_headings.append(heading)
            
    assert not missing_headings, \
        f"{PR_TEMPLATE_PATH} is missing the following key headings: {missing_headings}"

    # Optionally, check for a specific item from the checklist to ensure content integrity
    assert "- [ ] My code follows the style guidelines of this project." in content, \
        f"{PR_TEMPLATE_PATH} is missing a key checklist item."
    assert "- [ ] Linking to #1782 in description." in content, \
        f"{PR_TEMPLATE_PATH} is missing the specific linking checklist item."

```
