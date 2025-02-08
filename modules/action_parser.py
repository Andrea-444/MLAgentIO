import json
import re
from typing import Dict, Optional, Tuple
from action_functions import ActionExecutioner


class LLMActionParser:
    DEFAULT_ACTION_NAME_DICT = {
        'List Files': ActionExecutioner.list_files,
        'Execute Script': ActionExecutioner.execute_script,
        'Final Answer': ActionExecutioner.final_answer,
        'Understand File': ActionExecutioner.understand_file,
        'Inspect Script Lines': ActionExecutioner.inspect_script_lines,
        'Edit Script (AI)': ActionExecutioner.edit_script_ai
    }

    @staticmethod
    def replace_n_occurrences(text: str, old: str, new: str, n: int, reverse: bool = False) -> str:
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

    @staticmethod
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
            action_start = message.find('Action:')
            if action_start == -1:
                return None, None

            # Extract action name
            action_end = message.find('\n', action_start)
            action = message[action_start + 7:action_end].strip()

            # Find the action input
            input_start = message.find('Action Input:', action_end)
            if input_start == -1:
                return None, None

            # Extract action input json
            input_end = message.find(':', input_start)
            if input_end == -1:
                input_end = len(message)

            input_args_end = message.find("}", input_end)

            action_input_str = message[input_end + 1:input_args_end + 1].strip()
            action_input_str = LLMActionParser.replace_n_occurrences(action_input_str, "\n", " ", n=3)
            action_input_str = LLMActionParser.replace_n_occurrences(action_input_str, "\n", " ", n=3, reverse=True)
            action_input_str = action_input_str.replace("\n", "\\n")
            action_input_str = action_input_str.replace("None", "null")
            action_input = json.loads(action_input_str)

            return action, action_input

        except Exception as e:
            print(f"Error parsing message: {e}")
            return None, None



