def build_prompt(prompt_template: str, python_code: str, java_declaration: str) -> str:
    """
    Constructs a prompt by replacing placeholders `<python-code>` and `<java-declaration>` 
    in the given prompt template with the provided Python solution and Java declaration.

    Parameters:
        prompt_template (str): The template string containing instruction and placeholders 
            `<python-code>`, `<java-declaration>` for the Python code and Java declaration.
        python_code (str): The Python code including imports and function code to be inserted into the prompt.
        java_declaration (str): The Java declaration to be inserted into the prompt.

    Returns:
        str: The final prompt with placeholders replaced by the actual Python code and Java declaration.

    Example:
        >>> prompt_template = (
        ...     "code translation\n"
        ...     "Python:\n<python-code>\n"
        ...     "Java:\n<java-declaration>"
        ... )
        >>> python_code = (
        ...     "def add(x: int, y: int):\n"
        ...     "    return x + y"
        ... )
        >>> java_declaration = (
        ...     "class Solution {\n"
        ...     "    public int add(int x, int y) {\n"
        ... )
        >>> print(build_prompt(prompt_template, python_code, java_declaration))
        code translation
        Python:
        def add(x: int, y: int):
            return x + y
        Java:
        class Solution {
            public int add(int x, int y) {
                return x + y;
            }
        }
    """
    return (prompt_template
            .replace('<python-code>', python_code)
            .replace('<java-declaration>', java_declaration))


# Example usage
prompt_template = """
code translation
Python:
<python-code>
Java:
<java-declaration>
"""

python_code = """
def add(x: int, y: int):
    return x + y
"""

java_declaration = """
import java.util.*;
import java.lang.*;

class Solution {
    public int add(int x, int y) {
"""

print(build_prompt(prompt_template, python_code, java_declaration))

