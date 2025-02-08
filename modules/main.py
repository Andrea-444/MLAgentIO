from modules.action_functions import ActionExecutioner
from modules.action_parser import LLMActionParser
from modules.llm_assistant import LLMAssistant
from modules.ml_agent_io import *
from logger import AgentLogger


def test_agent(task_name: str | None, api_key: str):
    ml_agent_io = MLAgentIO(task_name=task_name)

    print(
        "Since this is a test environment, you will be prompted for confirmation "
        "before proceeding with each step. Follow the on-screen instructions carefully.")

    main_assistant = LLMAssistant(api_key=api_key,
                                  starting_instructions=ml_agent_io.get_instructions())

    parser = LLMActionParser()

    executioner = ActionExecutioner(action_mapping=parser.DEFAULT_ACTION_NAME_DICT,
                                    task_dir_path=ml_agent_io.active_task.get_dir_path(),
                                    api_key=api_key)

    logger = AgentLogger(task=ml_agent_io.get_active_task())

    observation_index = 1
    output = None
    observation = None
    while True:
        if observation_index == 1:
            output = main_assistant.initiate_conversation(research_problem=ml_agent_io.get_research_problem())
            logger.initial_log(instructions=ml_agent_io.get_instructions(),
                               research_problem=ml_agent_io.get_research_problem())
        else:
            output = main_assistant.consult(observation, observation_index)

        print(f"\n=======Output ({observation_index})=======:\n", output)
        print("=" * 10)

        action_name, action_args, = parser.parse_message(output)
        print("Action:\n", action_name, "\nAction Inputs:\n", action_args)
        print("=" * 10)

        print(
            "Should the stated action be executed? "
            "\nType 't' to force termination on the conversation."
            "\nType 'end' to forcefully end the agent process."
            "\nType anything else to proceed".upper())
        command = input()
        command = command.lower()

        if command.lower() == "end":
            break

        if command == "t":
            output = main_assistant.consult("Terminate", observation_index + 1)
            action_name, action_args, = parser.parse_message(output)
            print("Action:\n", action_name, "\nAction Inputs:\n", action_args)
            print("=" * 10)

            observation = executioner.execute(action_name, action_args)
            print("Observation:\n", observation)

            logger.save_log(output, observation)

            main_assistant.end_conversation()
            logger.close()
            break

        observation = executioner.execute(action_name, action_args)
        print("Observation:\n", observation)

        logger.save_log(output, observation)

        if ActionExecutioner.FINAL_ANSWER_FLAG in observation:
            main_assistant.end_conversation()
            logger.close()
            break

        print(
            "Should the agent process continue executing?"
            "\nType 'end' to forcibly terminate the agent."
            "\nType anything else to proceed.".upper())
        command = input()
        command = command.lower()
        if command == "end":
            break

        observation_index += 1

    main_assistant.print_history()


def test_output(output_path: str):
    output = read_file(output_path)
    parse = LLMActionParser()
    action_name, action_args = parse.parse_message(output)
    print("Action:\n", action_name, "\nAction Inputs:\n", action_args)
    command = input()
    if command == "end":
        return
    task = Task(name="sarcasm_lstm")
    executioner = ActionExecutioner(action_mapping=LLMActionParser.DEFAULT_ACTION_NAME_DICT,
                                    task_dir_path=task.get_dir_path(), api_key=API_KEY)
    observation = executioner.execute(action_name, action_args)
    print("Observation:\n", observation)


if __name__ == '__main__':
    API_KEY = read_file("../api_keys/open_ai.txt")

    # Leave task_name = None to choose from a given list
    test_agent(task_name=None,
               api_key=API_KEY)
