import os
from openai import OpenAI
from modules.action_functions import ActionExecutioner
from modules.action_parser import LLMActionParser
from datetime import datetime
from modules.llm_assistant import LLMAssistant
from modules.low_level_actions import read_file, save_file
from modules.MLAgentIO import *


class AgentLogger:
    LOGS_DEFAULT_DIR = "../logs"

    def __init__(self, task: Task, logs_dir_path: str = None, ):
        if logs_dir_path is None:
            logs_dir_path = AgentLogger.LOGS_DEFAULT_DIR

        log_file_name = f"log_{task.name}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.txt"
        log_file_path = os.path.join(logs_dir_path, log_file_name)

        self.logs_file = open(log_file_path, mode="w", encoding="utf-8")
        self.step = 0

    def initial_log(self, instructions: str, research_problem: str):
        self.logs_file.write(instructions)
        self.logs_file.write(research_problem)

    def save_log(self, output: str, observation: str):
        self.logs_file.write(f"\n\nStep {self.step}:\n\n")
        self.logs_file.write(output)
        self.logs_file.write("\nObservation: \n'''\n")
        self.logs_file.write(observation)
        self.logs_file.write("\n'''")
        self.step += 1

    def close(self):
        self.logs_file.close()


def test_agent():
    task_name = "sarcasm_lstm"

    ml_agent_io = MLAgentIO(task_name=None)

    main_assistant = LLMAssistant(api_key=API_KEY,
                                  starting_instructions=ml_agent_io.get_instructions())

    parser = LLMActionParser()

    executioner = ActionExecutioner(action_mapping=parser.DEFAULT_ACTION_NAME_DICT,
                                    task_dir_path=ml_agent_io.active_task.get_dir_path())

    logger = AgentLogger(ml_agent_io.get_active_task())

    output_index = 0
    output = None
    observation = None
    while True:
        if output_index == 0:
            output = main_assistant.initiate_conversation(research_problem=ml_agent_io.get_research_problem())
            logger.initial_log(instructions=ml_agent_io.get_instructions(),
                               research_problem=ml_agent_io.get_research_problem())
        else:
            output = main_assistant.consult(observation)

        print(f"\n=======Output ({output_index})=======:\n", output)
        print("=" * 10)

        action_name, action_args, = parser.parse_message(output)
        print("Action and action args:\n", action_name, action_args)
        print("=" * 10)

        observation = executioner.execute(action_name, action_args)
        print("Observation:\n", observation)

        logger.save_log(output, observation)

        if ActionExecutioner.FINAL_ANSWER_FLAG in observation:
            main_assistant.end_conversation()
            logger.close()
            break

        output_index += 1

    main_assistant.print_history()


def test_supporting_assistant_andrea():
    ml_agent_io = MLAgentIO(task_name=None)
    parser = LLMActionParser()
    # script_content = read_file(full_path="../test/dummy_script.py")
    executor = ActionExecutioner(action_mapping=parser.DEFAULT_ACTION_NAME_DICT,
                                 task_dir_path=ml_agent_io.active_task.get_dir_path())
    message = read_file(full_path="../test/dummy_output_4.txt")
    # message = read_file(full_path="../test/dummy_output_4.txt")
    action_name, action_args = parser.parse_message(message)
    output = executor.execute(action_name=action_name, action_args=action_args)
    print("=" * 15, "OUTPUT", "=" * 15)
    print(output)
    save_file(output, "../test", "edited_dummy_script.py")


def test_supporting_assistant():
    supporting_assistant = LLMAssistant(api_key=API_KEY,
                                        starting_instructions=MLAgentIO.build_instructions(
                                            MLAgentIO.SUPPORTING_LLM_INSTRUCTIONS_DIR))

    script_content = read_file(full_path="../test/dummy_script.py")
    understand_file_instructions = read_file(full_path="../test/dummy_instructions_understand.txt")
    edit_script_instructions = read_file(full_path="../test/dummy_instructions_edit.txt")
    output = supporting_assistant.consult_once(script_content, edit_script_instructions)
    print("=" * 15, "OUTPUT", "=" * 15)
    print(output)
    save_file(output, "../test", "edited_dummy_script.py")


if __name__ == '__main__':
    # test_agent()
    # test_supporting_assistant()
    test_supporting_assistant_andrea()
