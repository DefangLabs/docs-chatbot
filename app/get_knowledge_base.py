import os
import shutil
import subprocess
import re
import json
from git import Repo

kb_file_path = './data/knowledge_base.json'

def clone_repository(repo_url, local_dir):
    """ Clone or pull the repository based on its existence. """
    if not os.path.exists(local_dir):
        print(f"Cloning repository into {local_dir}")
        Repo.clone_from(repo_url, local_dir, depth=1)
    else:
        print(f"Repository already exists at {local_dir}. Pulling latest changes...")
        repo = Repo(local_dir)
        repo.git.pull()

def setup_repositories():
    tmp_dir = ".tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    # Define repositories and their URLs
    repos = {
        "defang-docs": "https://github.com/DefangLabs/defang-docs.git",
        "defang": "https://github.com/DefangLabs/defang.git"
    }

    # Clone each repository
    for repo_name, repo_url in repos.items():
        clone_repository(repo_url, os.path.join(tmp_dir, repo_name))

def run_prebuild_script():
    """ Run the 'prebuild.sh' script located in the .tmp directory. """
    os.chdir(".tmp")
    script_path = os.path.join("./", "prebuild.sh")  # Ensure the path is correct
    os.chdir("..")
    if os.path.exists(script_path):
        print("Running prebuild.sh...")
        try:
            subprocess.run(["bash", script_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running prebuild.sh: {e}")
    else:
        print("prebuild.sh not found.")

def parse_markdown():
    """ Parse markdown files in the current directory into JSON """
    reset_knowledge_base()  # Reset the JSON database file
    recursive_parse_directory('./.tmp/defang-docs')  # Parse markdown files in the current directory
    print("Markdown parsing completed successfully.")

def reset_knowledge_base():
    """ Resets or initializes the knowledge base JSON file. """
    with open(kb_file_path, 'w') as output_file:
        json.dump([], output_file)

def parse_markdown_file_to_json(json_output, current_id, file_path):
    """ Parses individual markdown file and adds its content to JSON """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Skip the first 5 lines
    markdown_content = "".join(lines[5:])

    # First pass: Determine headers for 'about' section
    sections = []
    current_section = {"about": [], "text": []}
    has_main_header = False

    for line in markdown_content.split('\n'):
        header_match = re.match(r'^(#{1,6}|\*\*+)\s+(.*)', line)  # Match `#`, `##`, ..., `######` and `**`
        if header_match:
            header_level = len(header_match.group(1).strip())
            header_text = header_match.group(2).strip()

            if header_level == 1 or header_match.group(1).startswith('**'):  # Treat `**` as a main header
                if current_section["about"] or current_section["text"]:
                    sections.append(current_section)
                current_section = {"about": [header_text], "text": []}
                has_main_header = True
            else:
                if has_main_header:
                    current_section["about"].append(header_text)
                else:
                    if header_level == 2:
                        if current_section["about"] or current_section["text"]:
                            sections.append(current_section)
                        current_section = {"about": [header_text], "text": []}
                    else:
                        current_section["about"].append(header_text)
        else:
            current_section["text"].append(line.strip())

    if current_section["about"] or current_section["text"]:
        sections.append(current_section)

    # Second pass: Combine text while ignoring headers and discard entries with empty 'about' or 'text'
    for section in sections:
        about = ", ".join(section["about"])
        text = " ".join(line for line in section["text"] if line)

        if about and text:  # Only insert if both 'about' and 'text' are not empty
            json_output.append({
                "id": current_id,
                "about": about,
                "text": text,
                "path": adjust_knowledge_base_entry_path(file_path)  # Adjust path format
            })

def adjust_knowledge_base_entry_path(file_path):
    """ Adjusts the file path format for storage. """
    return re.sub(r'\/(\d{4})-(\d{2})-(\d{2})-', r'/\1/\2/\3/', normalize_docs_path(file_path))

def normalize_docs_path(path):
    """ Normalizes the file path to ensure consistent formatting. """
    return path.replace("./.tmp/defang-docs", "").replace(".mdx", "").replace(".md", "")

def parse_cli_markdown(json_output, current_id, file_path):
    """ Parses CLI-specific markdown files """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    if len(lines) < 5:
        print(f"File {file_path} does not have enough lines to parse.")
        return

    # Extract 'about' from the 5th line (index 4)
    about = lines[4].strip()

    # Combine all remaining lines after the first 5 lines into 'text'
    text_lines = lines[5:]
    text = "".join(text_lines).strip()

    # Only append if both 'about' and 'text' are not empty
    if about and text:
        json_output.append({
            "id": current_id,
            "about": about,
            "text": text,
            "path": normalize_docs_path(file_path)
        })

def recursive_parse_directory(root_dir):
    """ Recursively parses all markdown files in the directory. """
    paths = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            lower_filename = filename.lower()
            if lower_filename.endswith('.md') or lower_filename.endswith('.mdx'):
                paths.append(os.path.join(dirpath, filename))

    with open(kb_file_path, 'r') as kb_file:
        kb_data = json.load(kb_file)

    for id, file_path in enumerate(paths, start=1):
        if 'cli' in dirpath.lower() or 'cli' in filename.lower():
            parse_cli_markdown(kb_data, id, file_path)
        else:
            parse_markdown_file_to_json(kb_data, id, file_path)

    with open(kb_file_path, 'w') as kb_file:
        json.dump(kb_data, kb_file, indent=2)

if __name__ == "__main__":
    setup_repositories()
    run_prebuild_script()
    parse_markdown()  # Start parsing logic after all setups
    print("All processes completed successfully.")
