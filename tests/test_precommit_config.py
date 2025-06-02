# tests/test_precommit_config.py
import subprocess
import pytest

# Define the tools to check
# Ruff handles black, isort, flake8
TOOLS = ["ruff", "mypy"] 

@pytest.mark.parametrize("tool", TOOLS)
def test_tool_is_installed_and_runnable(tool):
    '''Test that each tool can be called via subprocess and returns a help message or version.'''
    try:
        # Most CLI tools support a --version or --help flag
        # For ruff, `ruff --version` or `ruff --help`
        # For mypy, `mypy --version` or `mypy --help`
        if tool == "ruff":
            result = subprocess.run([tool, "--version"], capture_output=True, text=True, check=False)
        elif tool == "mypy":
            result = subprocess.run([tool, "--version"], capture_output=True, text=True, check=False)
        else:
            # Default to --help for other tools if added in future
            result = subprocess.run([tool, "--help"], capture_output=True, text=True, check=False)

        # Check for a successful exit code (0 for version/help, or specific codes if known)
        # Some tools might output version to stderr, so check both stdout and stderr for content.
        assert result.returncode == 0, f"{tool} command failed with exit code {result.returncode}\nStderr: {result.stderr}"
        assert result.stdout or result.stderr, f"{tool} command did not produce output to stdout or stderr"

    except FileNotFoundError:
        pytest.fail(f"{tool} is not installed or not found in PATH.")
    except Exception as e:
        pytest.fail(f"An error occurred while trying to run {tool}: {e}")
