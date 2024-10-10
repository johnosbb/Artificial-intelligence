from datasets import load_dataset
import json

# Load the original dataset
dataset = load_dataset("jdpressman/retro-text-style-transfer-v0.1")

# Choose the relevant split, e.g., 'train'
data = dataset['train']

# Initialize a list to hold the formatted examples
formatted_examples = []

# Extract and format each example
for idx, item in enumerate(data):
    # Check if the item is None
    if item is None:
        print(f"Warning: Item at index {idx} is None and will be skipped.")
        continue

    # Print the current item for inspection (optional, can be commented out later)
    # print(f"Inspecting item at index {idx}: {item}")

    # Safely extract the relevant fields
    task_passage = item.get("task_passage")
    ground_truth = item.get("ground_truth")

    # Check if both fields are None or empty and skip if so
    if task_passage is None and ground_truth is None:
        print(f"Skipping item at index {idx} because both fields are None.")
        continue

    # Strip whitespace and check if they are empty
    if task_passage and task_passage.strip() == "":
        task_passage = None  # Treat empty start_task as None
    if ground_truth and ground_truth.strip() == "":
        ground_truth = None  # Treat empty ground_truth as None

    # If either field is None after the checks, skip the entry
    if task_passage is None or ground_truth is None:
        print(f"Skipping item at index {idx} due to empty fields:")
        print(f"   start_task: {task_passage}")
        print(f"   ground_truth: {ground_truth}")
        continue

    # If both fields are valid, format the example
    example = {
        "input": task_passage.strip(),
        "output": ground_truth.strip()
    }
    formatted_examples.append(example)

# Specify the output file path
output_file_path = './data/example_data.json'  # Change this to your desired output path

# Write the formatted examples to the output JSON file if there are any valid examples
if formatted_examples:
    with open(output_file_path, 'w') as outfile:
        json.dump(formatted_examples, outfile, indent=4)
    print(f'Formatted data has been saved to {output_file_path}')
else:
    print('No valid examples were found.')
