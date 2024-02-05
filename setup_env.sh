#!/bin/bash

# Prompt user for environment name
read -p "Enter the name for the virtual environment: " venv_name

# Create virtual environment
python3 -m venv "$venv_name"

# Activate virtual environment
source "$venv_name/bin/activate"

# Install dependencies
pip install -r requirements.txt

echo "Virtual environment set up and dependencies installed successfully."
