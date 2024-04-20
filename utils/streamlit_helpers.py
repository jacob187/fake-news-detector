def read_markdown_file(markdown_file):
    with open(markdown_file, "r") as file:
        content = file.read()
    return content
