folder = "text"
import os
import unidecode
import json

# Function to process files recursively
def process_files(directory):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        # If it's a directory, process it recursively
        if os.path.isdir(item_path):
            process_files(item_path)
        # If it's a file, process it
        elif os.path.isfile(item_path):
            try:
                with open(item_path, "r") as f:
                    text = f.read()
                    text = text.replace("\n", " ")
                    text = unidecode.unidecode(text)
                
                # Create a JSON structure with the text
                json_data = {"content": text}
                
                # Get new file name with json extension
                file_name = os.path.splitext(item)[0]
                json_file_path = os.path.join(directory, file_name + ".json")
                
                # Write the indented JSON
                with open(json_file_path, "w") as f:
                    json.dump(json_data, f, indent=4)
                
                # Remove the old file if the new file has a different name
                if item_path != json_file_path:
                    os.remove(item_path)
                    print(f"Processed: {item_path} -> {json_file_path}")
            except Exception as e:
                print(f"Error processing {item_path}: {e}")

# Start processing from the root folder
try:
    process_files(folder)
    print("Files have been cleaned, converted to JSON format, and renamed with .json extension")
except Exception as e:
    print(f"Error: {e}")
