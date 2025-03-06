folder = "text"
import os
import unidecode
import json
import re

# Function to unescape unicode characters in text
def unescape_unicode(text):
    # Pattern to find unicode escape sequences like \u00e9
    pattern = r'\\u([0-9a-fA-F]{4})'
    
    def replace_unicode(match):
        # Convert the hex code to an actual character
        return chr(int(match.group(1), 16))
    
    # Replace all occurrences of unicode escape sequences
    return re.sub(pattern, replace_unicode, text)

# Function to process files recursively
def process_files(directory):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        # If it's a directory, process it recursively
        if os.path.isdir(item_path):
            process_files(item_path)
        # If it's a file, process it
        elif os.path.isfile(item_path) and not item_path.endswith('.json'):
            try:
                with open(item_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    text = text.replace("\n", " ")
                    text = unescape_unicode(text)  # Replace Unicode escape sequences
                    text = unidecode.unidecode(text)
                
                # Get new file name with json extension
                file_name = os.path.splitext(item)[0]
                json_file_path = os.path.join(directory, file_name + ".json")
                
                # Write the text as raw JSON content
                with open(json_file_path, "w", encoding="utf-8") as f:
                    f.write(text)
                
                # Remove the old file if the new file has a different name
                if item_path != json_file_path:
                    os.remove(item_path)
                    print(f"Processed: {item_path} -> {json_file_path}")
            except Exception as e:
                print(f"Error processing {item_path}: {e}")

# Start processing from the root folder
try:
    process_files(folder)
    print("Files have been cleaned, unicode escaped characters replaced, and converted to text files with .json extension")
except Exception as e:
    print(f"Error: {e}")
