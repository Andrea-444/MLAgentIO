import json
from typing import Dict, Optional, Tuple
from action_functions import (
    list_files,
    execute_script,
    final_answer,
    understand_file,
    inspect_script_lines,
    edit_script_ai
)


def parse_message(message: str) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Parse the LLM message to extract Action and Action Input.

    Args:
        message (str): The full message from the LLM

    Returns:
        Tuple[Optional[str], Optional[Dict]]: Tuple containing the action name and parsed action input
    """
    try:
        # Find the action line
        action_start = message.find('Action: ')
        if action_start == -1:
            return None, None

        # Extract action name
        action_end = message.find('\n', action_start)
        action = message[action_start + 8:action_end].strip()

        # Find the action input
        input_start = message.find('Action Input: ', action_end)
        if input_start == -1:
            return None, None

        # Extract action input json
        input_end = message.find('\n', input_start)
        if input_end == -1:
            input_end = len(message)

        action_input_str = message[input_start + 14:input_end].strip()
        action_input = json.loads(action_input_str)

        return action, action_input

    except Exception as e:
        print(f"Error parsing message: {e}")
        return None, None


class LLMActionParser:
    def __init__(self):
        self.action_mapping = {
            'List Files': list_files,
            'Execute Script': execute_script,
            'Final Answer': final_answer,
            'Understand File': understand_file,
            'Inspect Script Lines': inspect_script_lines,
            'Edit Script (AI)': edit_script_ai
        }

    def process_action(self, message: str) -> str:
        """
        Process the message and route to appropriate function based on action.

        Args:
            message (str): The full message from the LLM

        Returns:
            str: Result of the action execution
        """
        action, action_input = parse_message(message)

        if action is None or action_input is None:
            return "Error: Could not parse action and input from message"

        if action not in self.action_mapping:
            return f"Error: Unknown action '{action}'"

        return self.action_mapping[action](action_input)


# Example
if __name__ == "__main__":
    # Example messages
    list_files_message = '''
        Reflection: The edit is correct.
        Action: List Files
        Action Input: {"dir_path": "C:/Users/Asus/Desktop/not_so_imp_pidp_proekt/diagrams_graphs"}
        Observation: The script executed successfully
        '''

    example_message_inspect_lines = '''
        Reflection: The edit is correct.
        Action: Inspect Script Lines
        Action Input: {"script_name": "./example-file.py", "start_line_number": 2, "end_line_number": 5}
        Observation: The script executed successfully
        '''

    execute_script_message = '''
        Action: Execute Script
        Action Input: {"script_name": "./example-file.py"}
        '''

    understand_file_message = '''
        Action: Understand File
        Action Input: {"file_name": "example.py", "things_to_look_for": "main function definition"}
        '''

    edit_script_message = '''
        Action: Edit Script (AI)
        Action Input: {"script_name": "example.py", "edit_instruction": "Add error handling", "save_name": "example_updated.py"}
        '''
    parser = LLMActionParser()
    result = parser.process_action(execute_script_message)
    print(f"Result: {result}")
