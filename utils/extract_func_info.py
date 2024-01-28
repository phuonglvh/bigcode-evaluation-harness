import ast

import javalang


def extract_java_func_info(source_code):
    tree = javalang.parse.parse(source_code)

    for path, node in tree:
        if isinstance(node, javalang.tree.MethodDeclaration):
            method_name = node.name
            return_type = node.return_type.name if node.return_type else None
            arguments = [(param.name, param.type.name)
                         for param in node.parameters]
            modifiers = node.modifiers

            return {
                'function_name': method_name,
                'arguments': arguments,
                'return_type': return_type,
                'modifiers': modifiers,
            }


# print(extract_java_func_info(
#     "package javalang.brewtab.com; class Test { public static long strlen(String s) {} }"))

def extract_py_func_info(source_code):
    # Parse the source code into an abstract syntax tree
    tree = ast.parse(source_code)

    # Find function definitions in the AST
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Extract function name
            function_name = node.name

            # Extract arguments and their types
            arguments = [(arg.arg, ast.get_source_segment(
                source_code, arg.annotation)) for arg in node.args.args]

            # Extract return type
            return_type = ast.get_source_segment(
                source_code, node.returns) if node.returns else None

            return {
                'function_name': function_name,
                'arguments': arguments,
                'return_type': return_type
            }


def extract_func_info(source_code, language):
    lang_extractor = {
        'python': extract_py_func_info,
        'java': extract_java_func_info,
    }

    return lang_extractor[language](source_code)
