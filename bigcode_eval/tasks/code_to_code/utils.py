import re


def remove_py_docstring(input_code):
    # return re.sub(r'\"\"\".*?\"\"\"', '', input_code, flags=re.DOTALL).strip()
    # Remove the docstring from the code and avoid leaving empty lines
    output_code = re.sub(r'\"\"\".*?\"\"\"', '', input_code,
                        flags=re.DOTALL)  # Remove docstring
    # Remove empty lines created by removing docstring
    output_code = re.sub(r'\n\s*\n', '\n', output_code)
    return output_code.strip()  # Remove leading/trailing whitespace


def remove_java_comments_before_first_public_static_func(input_code):
    # Remove single-line comments before the 'public static' function
    code_cleaned = re.sub(
        r'(?<=\n)(\s*//.*?\n)+(?=\s*public static)', '', input_code)

    return code_cleaned.strip()
