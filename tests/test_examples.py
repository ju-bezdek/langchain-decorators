import os
import subprocess
import sys
import pytest
from pathlib import Path


def get_code_examples_dir():
    """Get the path to the code_examples directory."""
    current_dir = Path(__file__).parent
    code_examples_dir = current_dir.parent / "code_examples"
    return code_examples_dir


def get_example_files():
    """Get all Python files in the code_examples directory."""
    code_examples_dir = get_code_examples_dir()
    if not code_examples_dir.exists():
        return []

    python_files = list(code_examples_dir.glob("*.py"))

    skip_files = [
        "custom_template_block_builder_llama2.py",  # This requires deepinfra API token
    ]
    return [
        f for f in python_files if f.name != "__init__.py" if f.name not in skip_files
    ]


@pytest.mark.parametrize("example_file", get_example_files(), ids=lambda x: x.name)
def test_example_script_syntax_valid(example_file):
    """Test that each example script has valid Python syntax and imports."""

    # Test syntax by compiling the file
    try:
        with open(example_file, "r") as f:
            source = f.read()
        compile(source, str(example_file), "exec")
    except SyntaxError as e:
        pytest.fail(f"Syntax error in {example_file.name}: {e}")
    except Exception as e:
        pytest.fail(f"Failed to compile {example_file.name}: {e}")


@pytest.mark.parametrize("example_file", get_example_files(), ids=lambda x: x.name)
def test_example_script_runs_without_error(example_file):
    """Test that each example script runs without throwing exceptions."""

    # Run the script in a subprocess
    try:
        result = subprocess.run(
            [sys.executable, str(example_file)],
            cwd=str(example_file.parent),
            capture_output=True,
            text=True,
            timeout=30,  # 30 second timeout
        )

        # Check if the script executed successfully
        if result.returncode != 0:
            error_msg = f"Script {example_file.name} failed with return code {result.returncode}\n"
            error_msg += f"STDOUT:\n{result.stdout}\n"
            error_msg += f"STDERR:\n{result.stderr}"
            pytest.fail(error_msg)

    except subprocess.TimeoutExpired:
        pytest.fail(f"Script {example_file.name} timed out after 30 seconds")
    except Exception as e:
        pytest.fail(f"Failed to run script {example_file.name}: {str(e)}")


def test_code_examples_directory_exists():
    """Test that the code_examples directory exists."""
    code_examples_dir = get_code_examples_dir()
    assert (
        code_examples_dir.exists()
    ), f"Code examples directory not found at {code_examples_dir}"


def test_example_files_found():
    """Test that we found some example files to test."""
    example_files = get_example_files()
    assert (
        len(example_files) > 0
    ), "No Python example files found in code_examples directory"


if __name__ == "__main__":
    print(get_example_files())
