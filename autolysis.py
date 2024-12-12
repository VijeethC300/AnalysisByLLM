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

# Imports the load_dotenv function from the python-dotenv library.
from dotenv import load_dotenv

# Loads environment variables from a .env file.
load_dotenv()

import os

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
    "Columns may have unclean data in them. Ignore those cells"
    "You are tasked with identifying column names with their data types from a CSV file."
    "Infer the data type by considering majority of the values in each column."
    "Supported types are 'string', 'integer', 'boolean', 'float', 'date'."
    "Analyze the given data and return a JSON object where each entry has the column name and its inferred data type."
    "The response should directly use the 'get_column_type' function."
"""

## This is schema.
## function schema definition

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

with open(file='house-rent.csv', mode='r', encoding='utf-8') as f:
    data = ''.join([f.readline() for i in range (10)])

data

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

json.loads(json.dumps(functions))

df = pd.read_csv("house-rent.csv")

r = requests.post(url=url, headers=headers, json=json_data)

try:
    function_call = r.json()['choices'][0]['message']['function_call']
    #print(function_call)  # Print the function_call if successful
except (KeyError, IndexError):
    print("Error: Could not retrieve function_call from the API response.")

output = r.json()['choices'][0]['message']['function_call']['arguments']

print(output)

#json_data

# Define folder names
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