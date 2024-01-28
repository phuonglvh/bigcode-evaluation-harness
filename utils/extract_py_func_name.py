import re


def extract_py_func_name(func_code):
    func_sig_pattern = r'def\s+(\w+)\s*\('
    match = re.search(func_sig_pattern, func_code)
    if match:
        return match.group(1).strip()
    else:
        return None
