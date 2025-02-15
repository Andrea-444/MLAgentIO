import json
from typing import Dict, Optional, Tuple
from action_executioner import ActionExecutioner
from low_level_actions import replace_n_occurrences


class ActionParser:
    DEFAULT_ACTION_MAPPING = {
        'List Files': ActionExecutioner.list_files,
        'Execute Script': ActionExecutioner.execute_script,
        'Final Answer': ActionExecutioner.final_answer,
        'Understand File': ActionExecutioner.understand_file,
        'Inspect Script Lines': ActionExecutioner.inspect_script_lines,
        'Edit Script (AI)': ActionExecutioner.edit_script_ai
    }

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
            action_start = message.find('Action:')
            if action_start == -1:
                return None, None

            action_end = message.find('\n', action_start)
            action = message[action_start + 7:action_end].strip()

            input_start = message.find('Action Input:', action_end)
            if input_start == -1:
                return None, None

            input_end = message.find(':', input_start)
            if input_end == -1:
                input_end = len(message)

            input_args_end = message.find("}", input_end)

            action_input_str = message[input_end + 1:input_args_end + 1].strip()
            action_input_str = replace_n_occurrences(action_input_str, "\n", " ", n=3)
            action_input_str = replace_n_occurrences(action_input_str, "\n", " ", n=3, reverse=True)
            action_input_str = action_input_str.replace("\n", "\\n")
            action_input_str = action_input_str.replace("None", "null")
            action_input = json.loads(action_input_str)

            return action, action_input

        except Exception as e:
            print(f"Error parsing message: {e}")
            return None, None

    @staticmethod
    def parse_final_message(message: str) -> bool | None:
        """
           Parses a message to determine if the goal was achieved.

           Parameters:
               message (str): The input message containing the goal status.

           Returns:
               bool | None:
                   - True if the message indicates that the goal was achieved.
                   - False if the message explicitly states the goal was not achieved.
                   - None if the goal status could not be determined.

           Behavior:
               - Searches for the phrase 'Goal Achieved:' in the message.
               - Extracts the status following this phrase.
               - Returns True if the extracted status is 'True', False otherwise.
               - Handles exceptions gracefully and returns None in case of errors.
        """
        try:
            goal_start = message.find('Goal Achieved:')
            if goal_start == -1:
                return None

            goal_end = message.find('\n', goal_start)
            status = message[goal_start + len('Goal Achieved:'):goal_end].strip()
            task_status = status == 'True'
            return task_status
        except Exception as e:
            print(f"Error parsing final message: {e}")
            return None
