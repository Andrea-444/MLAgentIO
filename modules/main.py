from modules.action_functions import ActionExecutioner
from modules.action_parser import LLMActionParser
from modules.llm_assistant import LLMAssistant
from modules.ml_agent_io import MLAgentIO, Task
from modules.logger import AgentLogger
from modules.low_level_actions import read_file


def test_agent(task_name: str | None, api_key: str, assistant_model: str | None):
    print(
        "Since this is a test environment, you will be prompted for confirmation "
        "before proceeding with each step. Follow the on-screen instructions carefully.")

    ml_agent_io = MLAgentIO(task_name=task_name)

    main_assistant = LLMAssistant(api_key=api_key,
                                  starting_instructions=ml_agent_io.get_main_instructions(),
                                  model=assistant_model
                                  )

    supporting_assistant = LLMAssistant(api_key=api_key,
                                        starting_instructions=ml_agent_io.get_supporting_instructions(),
                                        model=assistant_model
                                        )

    parser = LLMActionParser()

    executioner = ActionExecutioner(action_mapping=parser.DEFAULT_ACTION_MAPPING,
                                    task_dir_path=ml_agent_io.get_task_env_dir_path(),
                                    assistant=supporting_assistant)

    logger = AgentLogger(task_name=ml_agent_io.get_active_task().name,
                         log_timestamp=ml_agent_io.get_activation_timestamp())

    iteration_index = 1
    output = None
    observation = None
    while True:
        if iteration_index == 1:
            output = main_assistant.initiate_conversation(research_problem=ml_agent_io.get_research_problem())
            logger.initial_log(instructions=ml_agent_io.get_main_instructions(),
                               research_problem=ml_agent_io.get_research_problem())
        else:
            output = main_assistant.consult(observation, iteration_index)

        print(f"\n=======Output ({iteration_index})=======:\n", output)
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
            output = main_assistant.consult("Terminate", iteration_index + 1)
            print(f"\n=======Output ({iteration_index + 1})=======:\n", output)
            print("=" * 10)
            action_name, action_args, = parser.parse_message(output)
            print("Action:\n", action_name, "\nAction Inputs:\n", action_args)
            print("=" * 10)

            observation = executioner.execute(action_name, action_args)
            print("Observation:\n", observation)

            logger.save_log(output, observation)

            main_assistant.end_conversation()
            logger.close()
            executioner.shutdown()
            break

        observation = executioner.execute(action_name, action_args)
        print("Observation:\n", observation)

        logger.save_log(output, observation)

        if ActionExecutioner.FINAL_ANSWER_FLAG in observation:
            main_assistant.end_conversation()
            logger.close()
            executioner.shutdown()
            break

        print(
            "Should the agent process continue executing?"
            "\nType 'end' to forcefully terminate the agent."
            "\nType anything else to proceed.".upper())
        command = input()
        command = command.lower()
        if command == "end":
            break

        iteration_index += 1


def test_output(output_path: str, task_name: str):
    output = read_file(output_path)
    parse = LLMActionParser()
    action_name, action_args = parse.parse_message(output)
    print("Action:\n", action_name, "\nAction Inputs:\n", action_args)
    print("Proceed ('end' to end, anything else to proceed):\n")
    command = input()
    if command.lower() == "end":
        return
    task = Task(name=task_name)
    executioner = ActionExecutioner(action_mapping=LLMActionParser.DEFAULT_ACTION_MAPPING,
                                    task_dir_path=task.get_dir_path(), api_key=API_KEY)
    observation = executioner.execute(action_name, action_args)
    print("Observation:\n", observation)


if __name__ == '__main__':
    # Load the API key from a file
    # API_KEY = read_file("{a path to an OpenAi api key}")
    API_KEY = read_file("../api_keys/open_ai.txt")

    # Specify the assistant model to use; if None, the default model: 'gpt-4o-mini' will be used
    assistant_model = None

    # Set the task name; if None, the user will be prompted to choose a task
    task_name = None

    test_agent(task_name=task_name,
               api_key=API_KEY,
               assistant_model=assistant_model)
