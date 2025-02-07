import os
from openai import OpenAI
from modules.action_functions import ActionExecutioner
from modules.action_parser import LLMActionParser
from datetime import datetime


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


API_KEY = read_file("../api_keys/open_ai.txt")


class Task:
    MAIN_DIR = "../tasks"
    SUBMISSION_FILE_NAME = "submission.txt"

    def __init__(self, name: str):
        self.name = name
        self.description = read_file(Task.MAIN_DIR, name, "description.txt")

    def get_dir_path(self):
        return os.path.join(self.MAIN_DIR, self.name, "setup")

    @staticmethod
    def list_all_tasks():
        return os.listdir(Task.MAIN_DIR)


class MLAgentIO:
    MAIN_LLM_INSTRUCTIONS_DIR = "../assistant_instructions/main_llm"
    SUPPORTING_LLM_INSTRUCTIONS_DIR = "../assistant_instructions/supporting_llm"

    def __init__(self, task_name: str | None = None):
        self.main_instructions: str = self.build_instructions(self.MAIN_LLM_INSTRUCTIONS_DIR)
        self.all_tasks: [Task] = Task.list_all_tasks()
        self.active_task: Task = self.__choose_task(task_name)

    @staticmethod
    def build_instructions(instructions_dir) -> str:
        instructions_file_names = os.listdir(instructions_dir)
        sorted_instructions_file_names = sorted(instructions_file_names,
                                                key=lambda file_name: int(file_name.split("_")[0]))
        instructions = ""
        for instruction_file_name in sorted_instructions_file_names:
            instructions += read_file(instructions_dir, instruction_file_name)
            instructions += "\n\n"
        return instructions

    def __choose_task(self, task_name: str | None) -> Task:
        if task_name is not None:
            return Task(task_name)

        print("Input the number before the task you want to choose:")
        for i, task in enumerate(self.all_tasks):
            print(i, task)
        task_index = int(input())
        return Task(self.all_tasks[task_index])

    def get_instructions(self):
        return self.main_instructions

    def get_research_problem(self):
        return f"Research Problem: {self.active_task.description}"

    def get_active_task(self):
        return self.active_task


class LLMAssistant:
    tmp_index = 0
    dummy_output_names = ["../test/dummy_output.txt", "../test/dummy_output_1.txt", "../test/dummy_output_2.txt"]

    def __init__(self, api_key: str, starting_instructions: str):
        self.starting_instructions = self.to_developer_message(starting_instructions)
        self.history = []
        self.client = OpenAI(api_key=api_key)

    @staticmethod
    def to_developer_message(instruction: str) -> dict:
        return {"role": "developer", "content": instruction}

    @staticmethod
    def to_user_message(message: str) -> dict:
        return {"role": "user", "content": message}

    @staticmethod
    def to_assistant_message(response: str) -> dict:
        return {"role": "assistant", "content": response}

    def build_context(self, observation: str) -> list:
        observation_message = self.to_user_message(observation)
        context = [self.starting_instructions] + self.history + [observation_message]
        self.history.append(observation_message)
        return context

    def __ask_assistant(self, context: list, model="gpt-4o-mini", max_tokens=None) -> str:
        dummy_output = read_file(self.dummy_output_names[self.tmp_index])
        self.tmp_index += 1
        return dummy_output

        # response = self.client.chat.completions.create(
        #     model=model,
        #     messages=context,
        #     max_tokens=max_tokens,
        #     n=1,
        #     store=True
        # )
        #
        # return response.choices[0].message.content

    def initiate_conversation(self, research_problem: str):
        research_problem_message = self.to_user_message(research_problem)
        initial_context = [self.starting_instructions, research_problem_message]
        self.print_context(initial_context)
        self.history.append(research_problem_message)
        output = self.__ask_assistant(initial_context)
        self.history.append(self.to_assistant_message(output))
        return output

    def consult(self, observation: str) -> str:
        context = self.build_context(observation)
        output = self.__ask_assistant(context)
        self.history.append(self.to_assistant_message(output))
        return output

    def consult_once(self, script_content: str, instructions: str) -> str:
        message = self.to_user_message(f"{instructions}\n\nScript Content:\n{script_content}")
        context = [self.starting_instructions, message]
        self.print_context(context)
        output = self.__ask_assistant(context)
        return output

    def print_history(self):
        print("=" * 50, "HISTORY", "=" * 50)
        for message in self.history:
            self.print_message(message)

    @staticmethod
    def print_message(message: dict):
        print("ROLE:", message["role"])
        print("CONTENT:\n", message["content"])
        print("=" * 100)

    @staticmethod
    def print_context(context: list):
        for message in context:
            print(message["content"])

    def end_conversation(self):
        self.client.close()


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
    test_agent()
