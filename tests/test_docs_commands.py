# tests/test_docs_commands.py
import pytest
from pathlib import Path
import re
import subprocess
import shlex # For parsing command lines safely
import logging

PROJECT_ROOT_DOC_CMDS = Path(__file__).resolve().parent.parent 
QUICK_START_MD_PATH_CMDS = PROJECT_ROOT_DOC_CMDS / "docs" / "quick_start.md"

logger_doc_cmds = logging.getLogger(__name__)

# List of commands that are generally safe to echo for syntax check.
# The prepare_command_for_dry_run prefixes the whole line with echo,
# so this list is more for conceptual understanding or future refinement if needed.
ECHO_SAFE_COMMANDS = ["python", "pip", "tensorboard", "git", "cd", "mkdir", "ls", "cat", "echo"]

def extract_code_blocks(markdown_content: str) -> List[str]:
    """Extracts shell/bash/python code blocks from markdown content."""
    code_blocks = []
    # Pattern for fenced code blocks (```lang\ncode\n``` or ```\ncode\n```)
    # Captures language (optional, group 1) and code content (group 2).
    # Uses re.DOTALL so `.` matches newlines within the code block.
    pattern = re.compile(r"```(\w*)\s*\n(.*?)\n```", re.DOTALL)
    matches = pattern.finditer(markdown_content)
    for match in matches:
        lang = match.group(1).lower().strip()
        code = match.group(2).strip()
        
        # We are interested in shell-like commands and python commands.
        # Shell: bash, shell, sh, zsh, or no language specified (empty string for lang)
        # Python: python, py
        if lang in ["bash", "shell", "sh", "zsh", "powershell", "cmd", ""] or "python" in lang or "py" in lang:
            code_blocks.append(code)
    return code_blocks

def prepare_command_for_dry_run(line: str) -> Optional[str]:
    """
    Prepares a command line for a dry-run syntax check using 'echo'.
    If the line is a comment, empty, or clearly not a command, returns None.
    """
    stripped_line = line.strip()
    
    # Ignore empty lines and comments
    if not stripped_line or stripped_line.startswith("#"):
        return None

    # Remove common shell prompts if they are part of the copied line
    # Common prompts: $, C:\>, bash-3.2$, etc.
    # This regex handles simple prompts like '$ ' or '> '
    prompt_pattern = r"^(?:[$#%>\s]|(?:bash-\d+\.\d+\$)\s)*" # More generic prompt start
    stripped_line_no_prompt = re.sub(prompt_pattern, "", stripped_line).strip()

    if not stripped_line_no_prompt: # If line was only a prompt
        return None

    # Check for common non-executable lines that might appear in code blocks
    if stripped_line_no_prompt.lower().startswith("output:") or \
       stripped_line_no_prompt.lower().startswith("example:") or \
       stripped_line_no_prompt.lower().startswith("note:") or \
       re.match(r"^\.\.\.", stripped_line_no_prompt) or \
       re.match(r"^\[.*?\]$", stripped_line_no_prompt): # e.g. [INFO] or [ERROR]
        return None
        
    # Attempt to parse with shlex to detect basic syntax errors like unmatched quotes early.
    # If shlex fails, it's a strong indicator of a shell syntax error.
    # In such a case, echoing the raw (but prompt-stripped) line is still a valid test
    # to see if the shell itself chokes on it.
    try:
        _ = shlex.split(stripped_line_no_prompt) 
    except ValueError:
        logger_doc_cmds.warning(f"shlex parsing failed for line (potential syntax error): '{stripped_line_no_prompt}'. Will still attempt echo dry-run.")
        # Even if shlex fails, prefix with echo to test shell's own parsing of the problematic line.
        # The error might be what we want to catch.

    # Prefix the (prompt-stripped) command with 'echo' for a syntax dry-run.
    # The goal is to see if the shell can parse the rest of the line without error
    # when `echo` is the command.
    return f"echo {stripped_line_no_prompt}"


@pytest.mark.skipif(not QUICK_START_MD_PATH_CMDS.exists(), reason=f"Quick start guide {QUICK_START_MD_PATH_CMDS} not found.")
def test_docs_commands_syntax(caplog):
    """
    Parses docs/quick_start.md for commands in code blocks,
    prepares them by prefixing with 'echo', and runs them via shell
    to check for basic syntax validity (expecting exit code 0).
    """
    caplog.set_level(logging.DEBUG) 
    logger_doc_cmds.info(f"Testing command syntax in {QUICK_START_MD_PATH_CMDS}")

    try:
        content = QUICK_START_MD_PATH_CMDS.read_text(encoding="utf-8")
    except Exception as e: # pragma: no cover
        pytest.fail(f"Could not read {QUICK_START_MD_PATH_CMDS}: {e}")

    code_blocks = extract_code_blocks(content)
    if not code_blocks: # pragma: no cover
        pytest.skip(f"No relevant code blocks (shell, python, etc.) found in {QUICK_START_MD_PATH_CMDS}.")

    all_commands_to_test: List[str] = []
    original_lines_map: Dict[str, str] = {} # Map prepared command back to original line for error reporting

    for block_idx, block_content in enumerate(code_blocks):
        for line_idx, line_content in enumerate(block_content.splitlines()):
            prepared_cmd = prepare_command_for_dry_run(line_content)
            if prepared_cmd:
                all_commands_to_test.append(prepared_cmd)
                original_lines_map[prepared_cmd] = line_content # Store original for context

    if not all_commands_to_test: # pragma: no cover
        pytest.skip(f"No executable command lines found in code blocks of {QUICK_START_MD_PATH_CMDS}.")

    logger_doc_cmds.info(f"Found {len(all_commands_to_test)} command lines to dry-run with 'echo'.")

    failed_commands = []

    for i, command_to_run in enumerate(all_commands_to_test):
        original_line = original_lines_map.get(command_to_run, command_to_run.replace("echo ", "", 1))
        logger_doc_cmds.debug(f"Dry-running command {i+1}/{len(all_commands_to_test)} (Original: '{original_line.strip()}'): {command_to_run}")
        try:
            # Using shell=True because `command_to_run` is a full string to be parsed by shell.
            # This is safe as it's prefixed with `echo`.
            result = subprocess.run(command_to_run, shell=True, capture_output=True, text=True, check=False, timeout=5)
            
            if result.returncode != 0:
                failed_commands.append({
                    "original_line": original_line,
                    "dry_run_command": command_to_run,
                    "return_code": result.returncode,
                    "stderr": result.stderr,
                    "stdout": result.stdout
                })
        except subprocess.TimeoutExpired: # pragma: no cover
            failed_commands.append({
                "original_line": original_line,
                "dry_run_command": command_to_run,
                "error": "TimeoutExpired"
            })
        except Exception as e: # pragma: no cover
            failed_commands.append({
                "original_line": original_line,
                "dry_run_command": command_to_run,
                "error": str(e)
            })
            
    if failed_commands:
        error_messages = ["One or more document commands failed syntax dry-run:"]
        for failure in failed_commands:
            error_messages.append(f"  Original Line: '{failure['original_line']}'")
            error_messages.append(f"  Dry-run Cmd  : '{failure['dry_run_command']}'")
            if "return_code" in failure:
                error_messages.append(f"  Exit Code    : {failure['return_code']}")
                if failure['stderr']: error_messages.append(f"  Stderr       : {failure['stderr'].strip()}")
                if failure['stdout']: error_messages.append(f"  Stdout       : {failure['stdout'].strip()}") # Echo usually prints to stdout
            else:
                error_messages.append(f"  Error        : {failure['error']}")
            error_messages.append("-" * 20)
        pytest.fail("\n".join(error_messages))
            
    logger_doc_cmds.info("All document command lines dry-ran successfully via 'echo'.")

```
