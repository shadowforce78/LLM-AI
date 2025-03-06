import os
import re
import argparse

def search_in_dumps(search_term, directory="text"):
    """
    Search for a term in all files with .json extension
    """
    results = []
    search_term = search_term.lower()
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        if search_term in content:
                            # Find position of the term for context
                            position = content.find(search_term)
                            start = max(0, position - 50)
                            end = min(len(content), position + 100)
                            context = content[start:end]
                            
                            # Try to extract a title for display
                            title_match = re.search(r'"title":\s*"([^"]+)"', content)
                            display_title = title_match.group(1) if title_match else file
                            
                            results.append({
                                "path": file_path,
                                "display_title": display_title,
                                "context": context
                            })
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Search for terms in JSON dump files')
    parser.add_argument('search_term', help='The term to search for')
    parser.add_argument('--dir', default='text', help='The directory to search in (default: text)')
    
    args = parser.parse_args()
    
    print(f"Searching for '{args.search_term}' in {args.dir}...")
    results = search_in_dumps(args.search_term, args.dir)
    
    if results:
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['path']}")
            print(f"   Title: {result['display_title']}")
            print(f"   Context: ...{result['context']}...")
    else:
        print("No results found.")

if __name__ == "__main__":
    main()
