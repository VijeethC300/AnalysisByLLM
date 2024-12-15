#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "httpx",
#     "chardet",
#     "matplotlib",
#     "pandas",
#     "python-dotenv",
#     "requests",
#     "seaborn",
# ]
# ///

# Import Libraries
import os
import sys
import json
import requests
import chardet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

# Constants
API_URL = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
MODEL = "gpt-4o-mini"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AIPROXY_TOKEN}",
}

FOLDERS = ["goodreads", "happiness", "media"]


# Function Definitions
def detect_encoding(file_path):
    """Detect the encoding of a file."""
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read(100000))  # Read first 100KB
    return result.get("encoding", "utf-8")


def clean_file_encoding(file_path, encoding):
    """Ensure the file is in UTF-8 encoding."""
    if encoding != "utf-8":
        with open(file_path, "r", encoding=encoding, errors="ignore") as infile:
            content = infile.read()
        with open(file_path, "w", encoding="utf-8") as outfile:
            outfile.write(content)


def read_sample_data(file_path, num_lines=10):
    """Read the first few lines of the dataset."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return "".join([f.readline() for _ in range(num_lines)])


def create_folders_with_readme(folder_names):
    """Create folders and add README.md and placeholder images."""
    for folder in folder_names:
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "README.md"), "w") as readme:
            readme.write(
                f"# {folder.capitalize()}\n\nThis is the README file for the {folder} folder."
            )
        with open(os.path.join(folder, "Image01.png"), "wb") as img:
            img.write(b"")


def prepare_json_payload(content, sample_data, functions):
    """Prepare the JSON payload for the API request."""
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": content},
            {"role": "user", "content": sample_data},
        ],
        "functions": functions,
        "function_call": {"name": "get_column_type"},
    }


def post_to_api(url, headers, json_data):
    """Send a POST request to the API."""
    response = requests.post(url, headers=headers, json=json_data)
    try:
        return response.json()
    except ValueError:
        raise Exception("Error parsing API response as JSON")


def clean_and_summarize_data(file_path, column_info):
    """
    Cleans and summarizes a dataset based on provided column metadata.

    Parameters:
    file_path (str): Path to the dataset (CSV format).
    column_info (list): List of dictionaries with column_name and column_type.

    Returns:
    tuple: Cleaned dataset (pd.DataFrame) and summary statistics (pd.DataFrame).
    """
    try:
        data = pd.read_csv(file_path)
        data = data.sample(n=min(10, len(data)), random_state=42)  # Limit to 10 samples

        # Clean and validate columns
        for col_meta in column_info:
            col_name, col_type = col_meta["column_name"], col_meta["column_type"]
            if col_name in data.columns:
                if col_type == "date":
                    data[col_name] = pd.to_datetime(data[col_name], errors="coerce")
                elif col_type in ["integer", "float"]:
                    data[col_name] = pd.to_numeric(data[col_name], errors="coerce")
                elif col_type == "string":
                    data[col_name] = data[col_name].astype(str)
                elif col_type == "boolean":
                    data[col_name] = data[col_name].astype(bool)
                data = data[data[col_name].notna()]
            else:
                print(f"Warning: Column '{col_name}' not found in dataset.")

        summary = data.describe(include="all").transpose()
        return data, summary
    except Exception as e:
        raise Exception(f"Error during data cleaning: {e}")


# Main Block
if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]

        # Encoding Detection and Cleaning
        encoding = detect_encoding(filename)
        clean_file_encoding(filename, encoding)

        # Read Sample Data
        sample_data = read_sample_data(filename)

        # JSON Payload Preparation
        functions = [
            {
                "name": "get_column_type",
                "description": "Identify column names and their data types from a dataset",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "column_metadata": {
                            "type": "array",
                            "description": "Meta data for each column.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "column_name": {
                                        "type": "string",
                                        "description": "Name of the column.",
                                    },
                                    "column_type": {
                                        "type": "string",
                                        "description": "DataType of the column.",
                                    },
                                },
                                "required": ["column_name", "column_type"],
                            },
                        },
                    },
                    "required": ["column_metadata"],
                },
            }
        ]

        # Prompt to the LLM
        prompt_content = """
            "Analyze the given dataset."
            "The first line is the header and the subsequest lines are sample data."
            "Identify the column names with their data types from a CSV file."
            "Infer the data type by considering majority of the values in each column."
            "Columns may have unclean data in them. Ignore those cells"
            "Supported types are 'string', 'integer', 'boolean', 'float', 'date'."
            "Analyze the given data and return a JSON object where,"
            "each entry has the column name and its inferred data type."
            "The response should directly use the 'get_column_type' function."
        """

        json_payload = prepare_json_payload(prompt_content, sample_data, functions)

        # API Interaction
        api_response = post_to_api(API_URL, HEADERS, json_payload)
        column_info = (
            api_response.get("choices", [{}])[0]
            .get("message", {})
            .get("function_call", {})
            .get("arguments", {})
        )
        try:
            column_metadata = json.loads(column_info).get("column_metadata", [])
        except json.JSONDecodeError:
            print("Error: Invalid JSON response from API")
            sys.exit(1)

        # Data Cleaning and Summarization
        try:
            cleaned_data, summary_stats = clean_and_summarize_data(
                filename, column_metadata
            )
            print("Cleaned Data:")
            print(cleaned_data)
            print("\nSummary Statistics:")
            print(summary_stats)
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Error: Please provide a CSV filename as an argument.")
