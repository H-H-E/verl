# tests/test_changelog_format.py
import pytest
from pathlib import Path

PROJECT_ROOT_CHANGELOG_TEST = Path(__file__).resolve().parent.parent
CHANGELOG_MD_PATH = PROJECT_ROOT_CHANGELOG_TEST / "CHANGELOG.md"

def test_changelog_exists_and_has_correct_header():
    """
    Tests that CHANGELOG.md exists and starts with the "# Changelog" heading.
    Also checks for the "[Unreleased]" section heading.
    """
    assert CHANGELOG_MD_PATH.exists(), f"{CHANGELOG_MD_PATH} does not exist."
    assert CHANGELOG_MD_PATH.is_file(), f"{CHANGELOG_MD_PATH} is not a file."

    first_line_content = ""
    full_content = ""
    try:
        with open(CHANGELOG_MD_PATH, "r", encoding="utf-8") as f:
            first_line_content = f.readline().strip()
        # Read full content for other checks if needed (like [Unreleased] section)
        full_content = CHANGELOG_MD_PATH.read_text(encoding="utf-8")

    except Exception as e: # pragma: no cover
        pytest.fail(f"Could not read {CHANGELOG_MD_PATH}: {e}")

    # Assert that the first line is exactly "# Changelog"
    expected_header = "# Changelog"
    assert first_line_content == expected_header, \
        f"{CHANGELOG_MD_PATH} should start with '{expected_header}', but started with '{first_line_content}'."

    # Check for the presence of "[Unreleased]" section heading as it's part of the specified format.
    # This is important for standard changelog formats like Keep a Changelog.
    expected_unreleased_section_header = "## [Unreleased]"
    assert expected_unreleased_section_header in full_content, \
        f"{CHANGELOG_MD_PATH} does not contain the '{expected_unreleased_section_header}' section heading."

    # Optional: Check for at least one bullet point under [Unreleased] if that's a policy
    # This regex looks for "## [Unreleased]" followed by lines, then at least one line starting with "- "
    # import re
    # unreleased_section_pattern = re.compile(r"##\s*\[Unreleased\]\s*
([\s\S]*?)(##\s*\[\d|\Z)", re.MULTILINE)
    # unreleased_match = unreleased_section_pattern.search(full_content)
    # if unreleased_match:
    #     unreleased_content = unreleased_match.group(1)
    #     assert re.search(r"^\s*-\s+", unreleased_content, re.MULTILINE), \
    #         f"The [Unreleased] section in {CHANGELOG_MD_PATH} does not seem to have any bullet points (e.g., '- Add new feature')."
    # else: # pragma: no cover (should be caught by previous assert if header is missing)
    #     pass # Already handled by the check for "## [Unreleased]" header

```
