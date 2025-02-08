from typing import Dict
import os
import json
import subprocess
import sys
from modules.llm_assistant import LLMAssistant
from modules.MLAgentIO import MLAgentIO

# API_KEY = read_file("../api_keys/open_ai.txt")
API_KEY = '<KEY>'


class ActionExecutioner:
    FINAL_ANSWER_FLAG = 'Final answer submitted'

    def __init__(self, action_mapping: dict, task_dir_path):
        self.action_mapping = action_mapping
        self.task_folder_path = task_dir_path
        # self.llm_assistant = llm_assistant

    def execute(self, action_name: str, action_args: dict) -> str:

        if action_name is None:
            return "Error: Action Name is None"

        if action_args is None:
            return "Error: Action Args is None"

        if action_name not in self.action_mapping:
            return f"Error: Unknown action '{action_name}'"

        action_args["task_folder_path"] = self.task_folder_path

        return self.action_mapping[action_name](action_args)

    @staticmethod
    def build_full_path(root, relative_path: str):
        return os.path.join(root, *relative_path.split("/"))

    @staticmethod
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
            full_dir_path = ActionExecutioner.build_full_path(args["task_folder_path"], dir_path)

            if not os.path.exists(full_dir_path):
                return f"Error: Directory '{full_dir_path}' does not exist"

            files = os.listdir(full_dir_path)
            return json.dumps(files, indent=2)
        except Exception as e:
            return f"Error listing files: {str(e)}"

    @staticmethod
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

            full_script_name = ActionExecutioner.build_full_path(args["task_folder_path"], script_name)

            if not os.path.exists(full_script_name):
                return f"Error: Script '{full_script_name}' does not exist"

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
                    output.append("Script Output: '''")
                    output.append(result.stdout)

                if result.stderr:
                    output.append("Errors:")
                    output.append(result.stderr)

                output.append(f"Process finished with exit code {result.returncode}")
                output.append("'''")

                if not output:
                    return "Script executed successfully with no output"

                return "\n".join(output)

            except subprocess.TimeoutExpired:
                return "Error: Script execution timed out after 30 seconds"
            except subprocess.SubprocessError as e:
                return f"Error executing script: {str(e)}"

        except Exception as e:
            return f"Error executing script: {str(e)}"

    @staticmethod
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
            return f"{ActionExecutioner.FINAL_ANSWER_FLAG}: {answer}"
        except Exception as e:
            return f"Error submitting final answer: {str(e)}"

    @staticmethod
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

            full_file_path = ActionExecutioner.build_full_path("", file_name)

            if not os.path.exists(full_file_path):
                return f"Error: File '{full_file_path}' does not exist"

            with open(full_file_path, 'r') as f:
                content = f.read()
            # TODO LLM Needed for this action
            supporting_assistant = LLMAssistant(api_key=API_KEY,
                                        starting_instructions=MLAgentIO.build_instructions(
                                            MLAgentIO.SUPPORTING_LLM_INSTRUCTIONS_DIR))

            llm_instruction = things_to_look_for

            llm_response = supporting_assistant.consult_once(content, llm_instruction)
            return llm_response
        except Exception as e:
            return f"Error understanding file: {str(e)}"

    @staticmethod
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

            if not all([script_name, start_line]):
                return "Error: Missing required parameters"

            full_script_name = ActionExecutioner.build_full_path(args["task_folder_path"], script_name)

            if not os.path.exists(full_script_name):
                return f"Error: Script '{full_script_name}' does not exist"

            with open(full_script_name, 'r') as f:
                lines = f.readlines()

            if end_line is not None:
                selected_lines = lines[start_line - 1:end_line]
            else:
                selected_lines = lines[start_line - 1:]
            return ''.join(selected_lines)
        except Exception as e:
            return f"Error inspecting script lines: {str(e)}"

    @staticmethod
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

            full_script_path = ActionExecutioner.build_full_path("", script_name)
            if os.path.exists(full_script_path):
                with open(full_script_path, 'r') as f:
                    content = f.read()
            else:
                content = ""  # New file will be created

            # TODO LLM Needed for this action
            supporting_assistant = LLMAssistant(api_key=API_KEY,
                                               starting_instructions=MLAgentIO.build_instructions(
                                                   MLAgentIO.SUPPORTING_LLM_INSTRUCTIONS_DIR))

            llm_instruction = f"Edit the following script based on the instruction: {edit_instruction}"
            edited_content = supporting_assistant.consult_once(script_content=content, instructions=llm_instruction)
            full_save_path = ActionExecutioner.build_full_path(args["task_folder_path"], save_name)
            with open(full_save_path, 'w') as f:
                f.write(edited_content)

            return edited_content

        except Exception as e:
            return f"Error editing script: {str(e)}"
