import os
import re


def read_file(*path_components, full_path: str = None) -> str:
    """
       Reads and returns the content of a file.

       Parameters:
           *path_components (str): Parts of the file path to be joined (if 'full_path' is not provided).
           full_path (str, optional): The complete file path. If provided, it takes precedence over 'path_components'.

       Returns:
           str: The content of the file as a string.

       Behavior:
           - If 'full_path' is provided, the function reads the file from this exact location.
           - If 'full_path' is not provided, the function constructs the file path using 'path_components'.
           - The file is opened in read mode with UTF-8 encoding.

       Example:
           read_file("folder", "subfolder", "file.txt")
           -> Reads content from "folder/subfolder/file.txt".

           read_file(full_path="/absolute/path/to/file.txt")
           -> Reads content from "/absolute/path/to/file.txt".
       """
    if full_path is not None:
        with open(full_path, mode="r", encoding="utf-8") as file:
            content = file.read()
        return content

    with open(os.path.join(*path_components), mode="r", encoding="utf-8") as file:
        content = file.read()
    return content


def save_file(content: str, destination: str, file_name: str):
    """
       Saves the given content to a file at the specified destination.

       Parameters:
           content (str): The text content to be written to the file.
           destination (str): The directory where the file should be saved.
           file_name (str): The name of the file to be created or overwritten.

       Behavior:
           - Constructs the full file path using 'destination' and 'file_name'.
           - Opens the file in write mode ('w') with UTF-8 encoding.
           - Writes the provided 'content' to the file, overwriting any existing content.

       Example:
           save_file("Hello, world!", "/home/user/documents", "greeting.txt")
           -> Creates or overwrites "/home/user/documents/greeting.txt" with "Hello, world!".
       """
    with open(os.path.join(destination, file_name), mode="w", encoding="utf-8") as file:
        file.write(content)


def build_full_path(root, relative_path: str):
    """
       Constructs an absolute file path by combining a root directory with a relative path.

       Parameters:
           root (str): The root directory.
           relative_path (str): The relative path, using forward slashes ("/") as separators.

       Returns:
           str: The full path formed by joining 'root' and 'relative_path'.

       Behavior:
           - Splits 'relative_path' using "/" and joins the segments with 'root' using 'os.path.join'.
           - Ensures compatibility with different operating systems by using the appropriate path separator.

       Example:
           build_full_path("/home/user", "documents/file.txt")
           -> "/home/user/documents/file.txt" (on Linux/macOS)
           -> "C:\\home\\user\\documents\\file.txt" (on Windows)
       """
    return os.path.join(root, *relative_path.split("/"))


def replace_n_occurrences(text: str, old: str, new: str, n: int, reverse: bool = False) -> str:
    """
        Replaces up to 'n' occurrences of a substring 'old' with a new substring 'new' in the given 'text'.

        Parameters:
            text (str): The input string where replacements will be performed.
            old (str): The substring to be replaced.
            new (str): The substring to replace 'old' with.
            n (int): The maximum number of occurrences to replace.
            reverse (bool, optional): If True, replaces the last 'n' occurrences instead of the first 'n'.
                                      Defaults to False (replaces the first 'n' occurrences).

        Returns:
            str: The modified string with replacements applied.

        Notes:
            - If 'n' is greater than the number of occurrences of 'old', all occurrences are replaced.
            - Uses regular expressions to find exact matches of 'old'.
            - Replacement is performed by modifying the string as a list for efficiency.

        Example:
            replace_n_occurrences("hello world, hello universe", "hello", "hi", 1)
            -> "hi world, hello universe"

            replace_n_occurrences("hello world, hello universe", "hello", "hi", 1, reverse=True)
            -> "hello world, hi universe"
        """
    matches = list(re.finditer(re.escape(old), text))

    if len(matches) < n:
        n = len(matches)

    if reverse:
        replace_indices = [m.start() for m in matches[-n:]]  # Last n occurrences
    else:
        replace_indices = [m.start() for m in matches[:n]]  # First n occurrences

    result = list(text)
    for index in replace_indices:
        result[index:index + len(old)] = new

    return "".join(result)
