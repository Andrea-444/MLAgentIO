import os
from openai import OpenAI
from modules.action_functions import ActionExecutioner
from modules.action_parser import LLMActionParser


def read_file(*path_components) -> str:
    with open(os.path.join(*path_components), mode="r", encoding="utf-8") as file:
        content = file.read()
    return content


API_KEY = read_file("../api_keys/open_ai.txt")


class Task:
    MAIN_DIR = "../tasks"
    SUBMISSION_FILE_NAME = "submission.txt"

    def __init__(self, name: str):
        self.name = name
        self.description = read_file(Task.MAIN_DIR, name, "description.txt")

    def get_folder_path(self):
        return os.path.join(self.MAIN_DIR, self.name, "setup")

    @staticmethod
    def list_all_tasks():
        return os.listdir(Task.MAIN_DIR)


class MLAgentIO:
    INSTRUCTIONS_DIR = "../instructions"

    def __init__(self, task_name: str | None = None):
        self.instructions = self.build_instructions()
        self.all_tasks = Task.list_all_tasks()
        self.active_task: Task = self.__choose_task(task_name)
        self.starting_prompt = self.get_instructions_string()

    def build_instructions(self) -> dict:
        instructions_file_names = os.listdir(self.INSTRUCTIONS_DIR)
        instructions = dict()
        for instruction_file_name in instructions_file_names:
            instruction_name = instruction_file_name.split(".")[0]
            instructions[instruction_name] = read_file(self.INSTRUCTIONS_DIR, instruction_file_name)
        return instructions

    def __choose_task(self, task_name: str | None) -> Task:

        if task_name is not None:
            return Task(task_name)

        print("Input the number before the task you want to choose:")
        for i, task in enumerate(self.all_tasks):
            print(i, task)
        task_index = int(input())
        return Task(self.all_tasks[task_index])

    def change_task(self, task_name: str):
        if self.active_task is not None and self.active_task.name == task_name:
            return

        self.active_task = Task(task_name)
        self.starting_prompt = self.get_instructions_string()

    def get_instructions_string(self) -> str:
        instructions_string = (f"{self.instructions["actions_description"]} "
                               f"\n\n{self.instructions["introduction"]}"
                               f"\n\n{self.instructions["response_format"]}")
        return instructions_string

    def get_research_problem(self):
        return f"Research Problem: {self.active_task.description}"

    def get_active_task_folder_path(self):
        return self.active_task.get_folder_path()


class AssistantOutput:

    def __init__(self, reflection: str, research_plan: str, fact_check: str, thought: str, action: str,
                 action_inputs: list):
        self.research_plan = research_plan
        self.reflection = reflection
        self.fact_check = fact_check
        self.thought = thought
        self.action = action
        self.action_inputs = action_inputs

    def to_string(self):
        return (f"Reflection:{self.reflection}\n\n"
                f"Research Plan and Status: {self.research_plan}\n\n"
                f"Fact Check: {self.fact_check}"
                f"Thought: {self.thought}\n\n"
                f"Action: {self.action}\n\n"
                f"Action Inputs: {self.action_inputs}\n\n")


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

    def build_context(self, assistant_output: str, observation: str) -> list:
        previous_assistant_message = self.to_assistant_message(assistant_output)
        self.history.append(previous_assistant_message)
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

    def initial_ask(self, research_problem: str):
        research_problem_message = self.to_user_message(research_problem)
        initial_context = [self.starting_instructions, research_problem_message]
        self.print_context(initial_context)
        self.history.append(research_problem_message)
        return self.__ask_assistant(initial_context)

    def ask(self, output: str, observation: str) -> str:
        context = self.build_context(output, observation)
        output = self.__ask_assistant(context)
        return output

    def ask_once(self, script_content:str, instructions:str) -> str:
        pass

    def print_history(self):
        print("="*50,"HISTORY", "="*50)
        for message in self.history:
            self.print_message(message)

    def print_message(self, message):
        print("ROLE:", message["role"])
        print("CONTENT:\n", message["content"])
        print("=" * 100)

    def print_context(self, context):
        for message in context:
            print(message["content"])


if __name__ == '__main__':
    ml_agent_io = MLAgentIO(task_name="sarcasm_lstm")
    llm_assistant = LLMAssistant(api_key=API_KEY,
                                 starting_instructions=ml_agent_io.get_instructions_string())
    parser = LLMActionParser()
    executioner = ActionExecutioner(action_mapping=parser.DEFAULT_ACTION_NAME_DICT,
                                    task_folder_path=ml_agent_io.get_active_task_folder_path())
    output_index = 0
    output = None
    observation = None
    while True:
        if output_index == 0:
            output = llm_assistant.initial_ask(ml_agent_io.get_research_problem())
        else:
            output = llm_assistant.ask(output, observation)


        print(f"\n=======Output ({output_index})=======:\n", output)
        print("=" * 10)

        action_name, action_args, = parser.parse_message(output)
        print("Action and action args:\n", action_name, action_args)
        print("=" * 10)

        observation = executioner.execute(action_name, action_args)
        print("Observation:\n", observation)

        if ActionExecutioner.FINAL_ANSWER_FLAG in observation:
            break

        output_index += 1

    llm_assistant.print_history()

