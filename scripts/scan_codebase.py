import os
import ast
import hashlib

def scan():
    project_root = "."
    py_files = []
    class_count = 0
    method_count = 0
    classes_list = []
    methods_list = []
    
    # We want to ignore .venv, .git, .pytest_cache, __pycache__, logs, checkpoints
    ignored_dirs = {".venv", ".git", ".pytest_cache", "__pycache__", "logs", "checkpoints", "checkpoints_chat", "checkpoints_ts", "brain"}
    
    module_dirs = set()
    codebase_hasher = hashlib.sha256()
    
    for root, dirs, files in os.walk(project_root):
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                py_files.append(file_path)
                
                # Check module
                rel_dir = os.path.relpath(root, project_root)
                if rel_dir != ".":
                    module_dirs.add(rel_dir.split(os.sep)[0])
                
                # Hash content
                with open(file_path, "rb") as f:
                    content = f.read()
                    codebase_hasher.update(content)
                
                # Parse AST
                try:
                    tree = ast.parse(content, filename=file_path)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            class_count += 1
                            classes_list.append((file_path, node.name))
                        elif isinstance(node, ast.FunctionDef):
                            # Check if it's a method inside a class
                            method_count += 1
                            methods_list.append((file_path, node.name))
                except Exception as e:
                    print(f"Error parsing {file_path}: {e}")

    codebase_hash = codebase_hasher.hexdigest()
    
    print("Codebase Scan Results:")
    print(f"  Python Files: {len(py_files)}")
    print(f"  Classes Count: {class_count}")
    print(f"  Methods Count: {method_count}")
    print(f"  Modules Count: {len(module_dirs)} ({list(module_dirs)})")
    print(f"  Codebase Hash: {codebase_hash}")
    
    print("\nClasses:")
    for file, name in sorted(classes_list):
        print(f"  - {name} in {file}")
        
    print("\nKey Methods:")
    for file, name in sorted(methods_list)[:50]: # Show first 50
        print(f"  - {name} in {file}")

if __name__ == "__main__":
    scan()
