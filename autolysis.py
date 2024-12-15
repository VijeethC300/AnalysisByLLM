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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import requests
import os
import sys
import chardet

# Imports the load_dotenv function from the python-dotenv library.
from dotenv import load_dotenv

# Loads environment variables from a .env file.
load_dotenv()

# Retrieves the value of the AIPROXY_TOKEN environment variable.
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

# Sets the URL for the OpenAI API endpoint.
url = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
model = "gpt-4o-mini"

# Header defintion for the HTTP request and authorization.
headers={"Content-Type":"application/json","Authorization":f"Bearer{AIPROXY_TOKEN}"}

# Prompt to the LLM
content  = """
     "Analyze the given dataset."
    "The first line is the header and the subsequest lines are sample data."
    "Identify the column names with their data types from a CSV file."
    "Infer the data type by considering majority of the values in each column."
    "Columns may have unclean data in them. Ignore those cells"
    "Supported types are 'string', 'integer', 'boolean', 'float', 'date'."
    "Analyze the given data and return a JSON object where each entry has the column name and its inferred data type."
    "The response should directly use the 'get_column_type' function."
"""

## This is function schema definition.

functions= [
    {
        "name": "get_column_type",
        "description": "Identify column names and their data types from a dataset",
        "parameters": {
            "type":"object",
            "properties": {
                "column_metadata": {
                    "type":"array",
                    "description":"Meta data for each column.",
                    "minItems": 1,
                    "items":{
                        "type": "object",
                        "properties": {
                            "column_name": {
                                "type": "string",
                                "description": "Name of the column."
                            },
                            "column_type": {
                                "type": "string",
                                "description": "DataType of the column."
                                
                            }
                        },
                        "required": ["column_name","column_type"]
                    }
                },  
            },
            "required":["column_metadata"]
        }
    }

]

#######################################################################
# To get file name entered in command prompt

def process_data(filename):
    with open(filename, 'r') as file:
        # Process the data from the CSV file here
        print(f"Processing data from: {filename}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        process_data(filename)
    else:
        print("Error: Please provide a CSV filename as an argument.")

#######################################################################
# To manage differnt file formats

encoding_details=dict()
with open(filename, "rb") as file:
    encoding_details = chardet.detect(file.read(100000))  # Reads the first 100KB of the file
    encoding_value = encoding_details['encoding']
    #print(encoding_value)

if encoding_value != 'utf-8':
    # Remove non-ASCII characters and save to a new file
    with open(filename, "r", encoding=encoding_value, errors="ignore") as infile:
        content = infile.read()

    with open(filename, "w", encoding='utf-8') as outfile:
        outfile.write(content)

# Read the cleaned file
df = pd.read_csv(filename)

#######################################################################

with open(file=filename, mode='r', encoding='utf-8',errors='ignore') as f:  # Ignore invalid characters
    data = ''.join([f.readline() for i in range (10)])

#######################################################################
# Specifiction as required by OpenAI
json_data = {
    "model": model,
    "messages": [
        {"role":"system", "content": content},
        {"role":"user", "content": data}

    ],
    #"functions": json.loads(json.dumps (functions)),
    "functions": functions,
    "function_call": {"name": "get_column_type"}
}

#######################################################################

r = requests.post(url=url, headers=headers, json=json_data)

#######################################################################

try:
    function_call = r.json()['choices'][0]['message']['function_call']
except (KeyError, IndexError):
    print("Error: Could not retrieve function_call from the API response.")

#######################################################################

output = r.json()['choices'][0]['message']['function_call']['arguments']

#######################################################################

# To create folder name
folders = ["goodreads", "happiness", "media"]

# Loop through each folder
for folder in folders:
    # Create folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    # Create README.md file inside the folder
    readme_path = os.path.join(folder, "README.md")
    with open(readme_path, "w") as readme_file:
        readme_file.write(f"# {folder.capitalize()}\n\nThis is the README file for the {folder} folder.")

    # Create Image01.png file inside the folder
    image_path = os.path.join(folder, "Image01.png")
    with open(image_path, "wb") as image_file:
        # Write an empty file or placeholder content
        image_file.write(b"")

####################################################################################

# Read random samples from the data set and do baisc clean-up

def clean_and_summarize_data(file_path, column_info):
    """
    Cleans and summarizes a dataset.

    Parameters:
    file_path (str): Path to the dataset (CSV format).
    column_info (list): List of dictionaries with column_name and column_type.

    Returns:
    tuple: Cleaned dataset (pd.DataFrame) and summary statistics (pd.DataFrame).
    """
    # Load the dataset
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        raise Exception(f"File not found: {file_path}")

    # Limit data analysis to a random sample of 10 records
    data = data.sample(n=10, random_state=42)

    # Validate and clean data based on column_info
    for column_info_item in column_info:
        if not isinstance(column_info_item, dict):
            raise Exception("Each item in column_info must be a dictionary with 'column_name' and 'column_type'.")

        column = column_info_item.get('column_name')
        dtype = column_info_item.get('column_type')

        if not column or not dtype:
            raise Exception("Each column_info dictionary must have 'column_name' and 'column_type' keys.")

        if column in data.columns:
            if dtype == 'date':
				# Convert to datetime, drop rows with invalid values
                data[column] = pd.to_datetime(data[column], errors='coerce')
                data = data[data[column].notna()]
            elif dtype == 'category':
                 # Ensure column is categorical
                data[column] = data[column].astype('category')
            elif dtype == 'integer':
                # Convert to integer, drop rows with invalid values
                data[column] = pd.to_numeric(data[column], errors='coerce', downcast='integer')
                data = data[data[column].notna()]
            elif dtype == 'float':
                # Convert to float, drop rows with invalid values
                data[column] = pd.to_numeric(data[column], errors='coerce')
                data = data[data[column].notna()]
            elif dtype == 'number':
                # Convert to float, drop rows with invalid values
                data[column] = pd.to_numeric(data[column], errors='coerce')
                data = data[data[column].notna()]
            elif dtype == 'boolean':
                # Convert to boolean
                data[column] = data[column].astype(bool)
            elif dtype in ['string', 'text', 'long_text']:
                # Treat as string
                data[column] = data[column].astype(str)
            elif dtype == 'binary':
                # Leave binary data as is (optional additional validation if required)
                pass
            elif dtype == 'uuid':
                # Validate UUID format
                data[column] = data[column].apply(lambda x: x if pd.notna(x) and isinstance(x, str) and len(x) == 36 else np.nan)
                data = data[data[column].notna()]
            else:
                print(f"Unsupported data type '{dtype}' for column '{column}'.")

        else:
            print(f"Column '{column}' not found in dataset.")

    # Generate summary statistics
    summary = data.describe(include='all').transpose()

    # Add additional summary metrics for categorical columns
    for col in data.select_dtypes(include='category').columns:
        summary.loc[col, 'unique_values'] = data[col].nunique()
        summary.loc[col, 'most_frequent'] = data[col].mode()[0]

    return data, summary

# Example usage
if __name__ == "__main__":
    #import json

    # Accept column info dynamically from a JSON string
    column_info_input = output
    try:
        column_info_dict = json.loads(column_info_input)

        if not isinstance(column_info_dict, dict) or 'column_metadata' not in column_info_dict:
            raise Exception("Invalid format. The input JSON must contain a 'column_metadata' key with a list of dictionaries.")

        column_info = column_info_dict['column_metadata']
        if not isinstance(column_info, list):
            raise Exception("'column_metadata' must be a list of dictionaries.")

    except json.JSONDecodeError:
        raise Exception("Invalid JSON format for column info.")

    file_path = filename
    try:
        cleaned_data, summary_stats = clean_and_summarize_data(file_path, column_info)
        print("Cleaned Data:")
        print(cleaned_data)
        print("\nSummary Statistics:")
        print(summary_stats)
    except Exception as e:
        print(f"Error: {e}")

####################################################################################