#!/bin/bash

# Replace path to your virtual environment
VENV_PATH="venv"

# Replace with path your Python script
SCRIPT_NAME="src/app.py"

# Define the path to the Python executable within the virtual environment
PYTHON_EXECUTABLE="$VENV_PATH/bin/python"

# Run the script using the specified Python executable
$PYTHON_EXECUTABLE -m streamlit run "$SCRIPT_NAME"