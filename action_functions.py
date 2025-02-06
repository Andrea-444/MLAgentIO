from typing import Dict
import os
import json
import subprocess
import sys


def list_files(args: Dict) -> str:
    """
    Use this to navigate the file system.
    Usage:
    ‘‘‘
    Action: List Files
    Action Input: {
    "dir_path": [a valid relative path to a directory, such as "." or "
                folder1/folder2"]
    }
    Observation: [The observation will be a list of files and folders in
                dir_path or current directory is dir_path is empty, or an error
                message if dir_path is invalid.]
    ‘‘‘
    """
    try:
        dir_path = args.get('dir_path', '.')
        if not os.path.exists(dir_path):
            return f"Error: Directory '{dir_path}' does not exist"

        files = os.listdir(dir_path)
        return json.dumps(files, indent=2)
    except Exception as e:
        return f"Error listing files: {str(e)}"


def execute_script(args: Dict) -> str:
    """
    Use this to execute the python script. The script must already exist.
    Usage:
    '''
    Action: Execute Script
    Action Input: {
    "script_name": [a valid python script name with relative path to
                    current directory if needed]
    }
    Observation: [The observation will be output of the script or errors.]
    '''
    """
    try:
        script_name = args.get('script_name')
        if not script_name:
            return "Error: No script name provided"

        if not os.path.exists(script_name):
            return f"Error: Script '{script_name}' does not exist"

        python_executable = sys.executable

        try:
            result = subprocess.run(
                [python_executable, script_name],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=os.path.dirname(os.path.abspath(script_name)) or '.'
            )

            output = []
            if result.stdout:
                output.append("Output:")
                output.append(result.stdout)

            if result.stderr:
                output.append("Errors:")
                output.append(result.stderr)

            output.append(f"Process returned: {result.returncode}")

            if not output:
                return "Script executed successfully with no output"

            return "\n".join(output)

        except subprocess.TimeoutExpired:
            return "Error: Script execution timed out after 30 seconds"
        except subprocess.SubprocessError as e:
            return f"Error executing script: {str(e)}"

    except Exception as e:
        return f"Error executing script: {str(e)}"


def final_answer(args: Dict) -> str:
    """
    Use this to provide the final answer to the current task.
    Usage:
    ‘‘‘
    Action: Final Answer
    Action Input: {
    "final_answer": [a detailed description on the final answer]
    }
    Observation: [The observation will be empty.]
    ‘‘‘
    """
    try:
        answer = args.get('final_answer')
        if not answer:
            return "Error: No final answer provided"
        return f"Final answer submitted: {answer}"
    except Exception as e:
        return f"Error submitting final answer: {str(e)}"


def understand_file(args: Dict) -> str:
    """
    Use this to read the whole file and understand certain aspects. You
    should provide detailed description on what to look for and what
    should be returned. To get a better understanding of the file, you
    can use Inspect Script Lines action to inspect specific part of the
    file.
    Usage:
    ‘‘‘
    Action: Understand File
    Action Input: {
    "file_name": [a valid file name with relative path to current
                directory if needed],
    "things_to_look_for": [a detailed description on what to look for and
                        what should be returned]
    }
    Observation: [The observation will be a description of relevant content
                and lines in the file. If the file does not exist, the observation
                will be an error message.]
    ‘‘‘

    """
    try:
        file_name = args.get('file_name')
        things_to_look_for = args.get('things_to_look_for')

        if not file_name or not things_to_look_for:
            return "Error: Missing required parameters"

        if not os.path.exists(file_name):
            return f"Error: File '{file_name}' does not exist"

        with open(file_name, 'r') as f:
            content = f.read()
        # TODO LLM Needed for this action
        return f"Analyzed file '{file_name}' looking for: {things_to_look_for}"
    except Exception as e:
        return f"Error understanding file: {str(e)}"


def inspect_script_lines(args: Dict) -> str:
    """
    Use this to inspect specific part of a python script precisely, or the
    full content of a short script. The number of lines to display is
    limited to 100 lines. This is especially helpful when debugging.
    Usage:
    ‘‘‘
    Action: Inspect Script Lines
    Action Input: {
    "script_name": [a valid python script name with relative path to
                    current directory if needed],
    "start_line_number": [a valid line number],
    "end_line_number": [a valid line number]
    }
    Observation: [The observation will be the content of the script between
                start_line_number and end_line_number . If the script does not exist,
                the observation will be an error message.]
    ‘‘‘

    """
    try:
        script_name = args.get('script_name')
        start_line = args.get('start_line_number')
        end_line = args.get('end_line_number')

        if not all([script_name, start_line, end_line]):
            return "Error: Missing required parameters"

        if not os.path.exists(script_name):
            return f"Error: Script '{script_name}' does not exist"

        with open(script_name, 'r') as f:
            lines = f.readlines()

        if end_line - start_line > 100:
            return "Error: Cannot display more than 100 lines"

        selected_lines = lines[start_line - 1:end_line]
        return ''.join(selected_lines)
    except Exception as e:
        return f"Error inspecting script lines: {str(e)}"


def edit_script_ai(args: Dict) -> str:
    """
    Use this to do a relatively large but cohesive edit over a python script.
    Instead of editing the script directly, you should describe the edit
    instruction so that another AI can help you do this.
    Usage:
    ‘‘‘
    Action: Edit Script (AI)
    Action Input: {
    "script_name": [a valid python script name with relative path to
                    current directory if needed. An empty script will be created if
                    it does not exist.],
    "edit_instruction": [a detailed step by step description on how to
                        edit it.],
    "save_name": [a valid file name with relative path to current
                directory if needed]
    }
    Observation: [The observation will be the edited content of the script.
                If the script does not exist, the observation will be an error
                message. You should always double-check whether the edit is correct.
                If it is far from correct, you can use the Undo Edit Script action to
                undo the edit.]
    ‘‘‘

    """
    try:
        script_name = args.get('script_name')
        edit_instruction = args.get('edit_instruction')
        save_name = args.get('save_name')

        if not all([script_name, edit_instruction, save_name]):
            return "Error: Missing required parameters"

        if os.path.exists(script_name):
            with open(script_name, 'r') as f:
                content = f.read()
        else:
            content = ""  # New file will be created

        # TODO LLM Needed for this action
        edited_content = f"Original content with applied changes based on: {edit_instruction}"

        return edited_content
    except Exception as e:
        return f"Error editing script: {str(e)}"
