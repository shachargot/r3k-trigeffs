import os
import json

# Function to merge JSON files
def merge_json_files(directory):
    merged_data = {}

    # List JSON files in the directory
    json_files = [f for f in os.listdir(directory) if f.endswith(".json")]

    for file_name in json_files:
        file_path = os.path.join(directory, file_name)

        # Read JSON file
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)

        # Use the file name as the key
        merged_data[file_name] = data

    return merged_data

if __name__ == "__main__":
    directory_path = "/eos/user/j/jodedra/AnalysisWork_2024/BDTScoreadder_FINAL/CMSSW_14_0_1/src/BDTscoreAdder/jsons/2023new/"  # Change this to your directory path
    merged_data = merge_json_files(directory_path)

    # Convert the merged data to a JSON object
    merged_json = json.dumps(merged_data, indent=4)

    # Optionally, save the merged JSON to a file
    with open("merged.json", 'w') as output_file:
        output_file.write(merged_json)

    print("Merged JSON:")
    print(merged_json)