import os

def replace_aira_with_aria(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file == 'rename_aria.py':
                continue
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                if 'AIRA' in content:
                    new_content = content.replace('AIRA', 'ARIA')
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"Updated: {file_path}")
            except Exception as e:
                print(f"Could not process {file_path}: {e}")

if __name__ == "__main__":
    replace_aira_with_aria('.')