# tests/test_docs_links.py
import pytest
from pathlib import Path
import re
import logging # For logging checked paths

# Define project root assuming tests are in tests/something.py
PROJECT_ROOT_DOC_LINKS = Path(__file__).resolve().parent.parent

QUICK_START_MD_PATH = PROJECT_ROOT_DOC_LINKS / "docs" / "quick_start.md"

logger_doc_links = logging.getLogger(__name__)

def test_quick_start_md_exists():
    """Checks if the quick_start.md file itself exists."""
    assert QUICK_START_MD_PATH.exists(), f"{QUICK_START_MD_PATH} does not exist."
    assert QUICK_START_MD_PATH.is_file(), f"{QUICK_START_MD_PATH} is not a file."

def test_quick_start_referenced_files_exist():
    """
    Parses docs/quick_start.md to find references to local files
    and checks if they exist relative to the project root.
    """
    if not QUICK_START_MD_PATH.exists(): # pragma: no cover (covered by previous test)
        pytest.skip(f"{QUICK_START_MD_PATH} not found, cannot test links.")

    try:
        content = QUICK_START_MD_PATH.read_text(encoding="utf-8")
    except Exception as e: # pragma: no cover
        pytest.fail(f"Could not read {QUICK_START_MD_PATH}: {e}")

    # Regex to find all content within backticks: `(.*?)`
    found_paths_in_backticks = re.findall(r"`([^`]+)`", content)

    # Predefined list of paths expected to be mentioned and exist.
    # These are based on the content of docs/quick_start.md from Task 20.1.
    # Paths are relative to the project root.
    expected_paths_to_check = [
        "configs/atropos_example.yaml",
        "recipe/atropos/launch_atropos_verl.py",
        "requirements.txt",
        # Directories mentioned as being created or used (existence check)
        "logs/",
        "runs/",
        "metrics/",
        "environments/" # Default path for --env-script-path
    ]

    # Filter and add plausible paths from backticked content
    for potential_path in found_paths_in_backticks:
        # Normalize: remove trailing punctuation that might be part of the sentence.
        cleaned_path = potential_path.rstrip(".,;:!?")

        # Heuristic to identify if it's a plausible file/dir path:
        # - Not a URL
        # - Contains path characters like '/' or known extensions or is a known filename
        # - Avoids very short strings that are likely not paths (e.g. `lr`)
        if cleaned_path.startswith("http"):
            continue
        if len(cleaned_path) < 3 and not (cleaned_path.endswith("/") or "." in cleaned_path) : # Avoid very short, non-path like strings
            continue

        is_plausible_path = False
        if "/" in cleaned_path or cleaned_path.endswith((".yaml", ".py", ".txt", ".md", "/")) or \
           cleaned_path in ["requirements.txt", "setup.py"] or \
           any(known_dir in cleaned_path for known_dir in ["configs", "recipe", "docs", "tests", "verl", "scripts", "logs", "runs", "metrics", "environments"]):
            is_plausible_path = True

        if is_plausible_path and cleaned_path not in expected_paths_to_check:
            # Further filter: avoid adding command snippets that are not paths
            # e.g., `python recipe/atropos/launch_atropos_verl.py --config ...` -> we only want the script path
            # For now, this simple addition is okay; complex command parsing is overkill here.
            # We are mainly interested in standalone paths mentioned in backticks.
            # Split by space and take the first part if it looks like a path.
            # This is still heuristic.
            first_part = cleaned_path.split(" ")[0]
            if "/" in first_part or first_part.endswith((".yaml", ".py", ".txt", ".md")) or first_part in ["requirements.txt"]:
                 if first_part not in expected_paths_to_check:
                    expected_paths_to_check.append(first_part)
            elif cleaned_path.endswith("/"): # If it's explicitly a directory
                 if cleaned_path not in expected_paths_to_check:
                    expected_paths_to_check.append(cleaned_path)


    # Remove duplicates and sort for consistent test output
    unique_paths_to_check = sorted(list(set(expected_paths_to_check)))

    if not unique_paths_to_check: # pragma: no cover (should always have predefined paths)
        pytest.skip("No file paths found or configured to check in quick_start.md content.")

    missing_files = []
    for file_path_str in unique_paths_to_check:
        # Ensure path is treated as relative to project root
        # Path.is_absolute() can be used if we need to handle absolute paths differently
        full_path = PROJECT_ROOT_DOC_LINKS / file_path_str

        # For paths ending with '/', we check if it's a directory.
        # For others, we check if it's a file (or could be a dir if no extension).
        # .exists() works for both files and directories.
        if not full_path.exists():
            missing_files.append(file_path_str)

    if missing_files:
        # Log for easier debugging in CI
        logger_doc_links.error(f"Missing files/directories from {QUICK_START_MD_PATH}: {missing_files}")
        for mf in missing_files:
             logger_doc_links.error(f"Expected at: {PROJECT_ROOT_DOC_LINKS / mf}")

    assert not missing_files, \
        f"The following files/directories referenced in {QUICK_START_MD_PATH} do not exist at project root or are malformed: {missing_files}"

    logger_doc_links.info(f"Successfully verified existence of {len(unique_paths_to_check)} referenced paths in {QUICK_START_MD_PATH}.")
    # For verbosity in test logs, print checked paths:
    # print(f"Verified paths: {unique_paths_to_check}")

```
