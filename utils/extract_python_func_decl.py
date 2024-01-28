import re


def extract_python_func_decl(input_string):
    # Define a regular expression pattern to match the function declaration
    pattern = re.compile(r'^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)(?:\s*->\s*(\S+)\s*)?:', re.MULTILINE)

    # Use re.search to find the match in the input string
    match = pattern.search(input_string)

    # Check if a match is found and retrieve the function declaration
    if match:
        function_declaration = match.group(0)
        return function_declaration.strip()
    else:
        return None