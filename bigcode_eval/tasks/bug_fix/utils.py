import re


def remove_comments(java_code):
    # Regular expression to match single-line comments
    single_line_comment_pattern = r'//.*?$'
    # Regular expression to match multi-line comments
    multi_line_comment_pattern = r'/\*.*?\*/'
    # Regular expression to remove consecutive empty lines
    empty_line_pattern = r'\n\s*\n'

    # Remove single-line comments
    java_code = re.sub(single_line_comment_pattern, '', java_code, flags=re.MULTILINE)
    # Remove multi-line comments
    java_code = re.sub(multi_line_comment_pattern, '', java_code, flags=re.DOTALL)
    # Remove consecutive empty lines
    java_code = re.sub(empty_line_pattern, '\n', java_code)

    return java_code

if __name__ == '__main__':
    # Example Java code with comments
    java_code_with_comments = """import java.util.*;
import java.lang.reflect.*;
import org.javatuples.*;
import java.security.*;
import java.math.*;
import java.io.*;
import java.util.stream.*;

class Problem {
    // Given a string text, replace all spaces in it with underscores,
    // and if a string has more than 2 consecutive spaces,
    // then replace all consecutive spaces with -
    // >>> fixSpaces((" Example"))
    // ("Example")
    // >>> fixSpaces((" Example 1"))
    // ("Example_1")
    // >>> fixSpaces((" Example 2"))
    // ("_Example_2")
    // >>> fixSpaces((" Example 3"))
    // ("_Example-3")
    public static String fixSpaces(String text) {

        
    }
}
"""

    # Remove comments from the Java code
    java_code_without_comments = remove_comments(java_code_with_comments)
    print(f'java_code_without_comments:\n{java_code_without_comments}')
