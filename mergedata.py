import os
import json

data_root = './data'

output_file = os.path.join(data_root, 'multidata.jsonl')

jsonl_files = []
for root, dirs, files in os.walk(data_root):
    for file in files:
        if file.endswith('.jsonl') and "alldata" not in file:
            jsonl_files.append(os.path.join(root, file))

combined_data = []
for file_path in jsonl_files:
    with open(file_path, 'r') as f:
        for line in f:
            try:
                combined_data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {file_path}: {e}")

with open(output_file, 'w') as f:
    for record in combined_data:
        f.write(json.dumps(record) + '\n')

output_file
