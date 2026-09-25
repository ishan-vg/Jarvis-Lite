from pathlib import Path


def load_markdown_files(vault_path):
    knowledge = []

    vault = Path(vault_path)

    for file in vault.rglob("*.md"):

        # Ignore Obsidian's internal files
        if ".obsidian" in file.parts:
            continue

        # Ignore Jarvis instructions
        if "06 Instructions" in file.parts:
            continue

        content = file.read_text(encoding="utf-8")

        knowledge.append({
            "file": file.name,
            "path": str(file),
            "content": content
        })

    return knowledge