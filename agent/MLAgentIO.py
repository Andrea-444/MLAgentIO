import os
from openai import OpenAI


def read_file(*path_components) -> str:
    with open(os.path.join(*path_components), mode="r", encoding="utf-8") as file:
        content = file.read()
    return content


class Task:
    MAIN_DIR = "../tasks"
    SUBMISSION_FILE_NAME = "submission.txt"

    def __init__(self, name: str):
        self.name = name
        self.description = read_file(Task.MAIN_DIR, f"{name}", "description.txt")

    @staticmethod
    def list_all_tasks():
        return os.listdir(Task.MAIN_DIR)


class MLAgentIO:
    INSTRUCTIONS_DIR = "../instructions"

    def __init__(self, task_name: str | None = None):
        self.instructions = self.build_instructions()
        self.all_tasks = Task.list_all_tasks()
        self.active_task: Task = self.__choose_task(task_name)
        self.starting_prompt = self.build_starting_prompt()

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
        self.starting_prompt = self.build_starting_prompt()

    def build_starting_prompt(self) -> str:
        prompt = (f"{self.instructions["tools_description"]} "
                  f"\n\n{self.instructions["introduction"]}"
                  f"\n\n{self.instructions["response_format"]}")
        return prompt

    def get_research_problem(self):
        return f"Research Problem: {self.active_task.description}"


API_KEY = read_file("../api_keys/open_ai.txt")


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
        response = self.client.chat.completions.create(
            model=model,
            messages=context,
            max_tokens=max_tokens,
            n=1,
            store=True
        )

        return response.choices[0].message.content

    def ask(self, output: str, observation: str) -> str:
        context = self.build_context(output, observation)
        output = self.__ask_assistant(context)
        return output


if __name__ == '__main__':
    ml_agent_io = MLAgentIO(task_name="sarcasm_lstm")
    prompt = ml_agent_io.starting_prompt

