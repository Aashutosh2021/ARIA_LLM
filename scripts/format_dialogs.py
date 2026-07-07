import os

def format_dialogs():
    input_path = os.path.join("data", "dialogs.txt")
    output_path = os.path.join("data", "dialogs.txt")
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return
        
    formatted_lines = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                parts = line.split("\t")
                if len(parts) >= 2:
                    user_msg = parts[0].strip()
                    aira_msg = parts[1].strip()
                    formatted_lines.append(f"USER: {user_msg}\nAIRA: {aira_msg}\n<|endoftext|>\n")
                else:
                    # Single message or unexpected format, skip or treat as USER
                    pass
            else:
                # No tab, skip or log
                pass
                
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(formatted_lines)
        
    print(f"Successfully formatted {len(formatted_lines)} dialog turns in {output_path}.")

if __name__ == "__main__":
    format_dialogs()
