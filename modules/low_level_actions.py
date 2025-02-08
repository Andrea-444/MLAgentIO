import os


def read_file(*path_components, full_path: str = None) -> str:
    if full_path is not None:
        with open(full_path, mode="r", encoding="utf-8") as file:
            content = file.read()
        return content

    with open(os.path.join(*path_components), mode="r", encoding="utf-8") as file:
        content = file.read()
    return content


def save_file(content: str, destination: str, file_name: str):
    with open(os.path.join(destination, file_name), mode="w", encoding="utf-8") as file:
        file.write(content)
