import os
import re
import argparse

def unescape_unicode(text):
    # Pattern to find unicode escape sequences like \u00e9
    pattern = r'\\u([0-9a-fA-F]{4})'
    
    def replace_unicode(match):
        # Convert the hex code to an actual character
        return chr(int(match.group(1), 16))
    
    # Replace all occurrences of unicode escape sequences
    return re.sub(pattern, replace_unicode, text)

def process_files(directory="text"):
    """
    Process all JSON files to unescape unicode characters
    """
    count = 0
    # Walk through all subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Replace unicode escape sequences
                    processed_content = unescape_unicode(content)
                    
                    # Write back if changes were made
                    if processed_content != content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(processed_content)
                        count += 1
                        print(f"Processed: {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    print(f"Processed {count} files with unicode escape sequences.")

def main():
    parser = argparse.ArgumentParser(description='Replace Unicode escape sequences in JSON files')
    parser.add_argument('--dir', default='text', help='The directory to process (default: text)')
    
    args = parser.parse_args()
    
    print(f"Processing files in {args.dir}...")
    process_files(args.dir)

if __name__ == "__main__":
    main()
