import os

def reset_json_files(directory="text"):
    """
    Remove all JSON files to prepare for fresh conversion
    """
    count = 0
    # Walk through all subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    count += 1
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
    
    print(f"Removed {count} JSON files. You can now run clean-text.py to create fresh files.")

if __name__ == "__main__":
    print("This script will remove all JSON files in the text directory.")
    confirm = input("Are you sure you want to proceed? (y/n): ")
    if confirm.lower() == 'y':
        reset_json_files()
    else:
        print("Operation cancelled.")
