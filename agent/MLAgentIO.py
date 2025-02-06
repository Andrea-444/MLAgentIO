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
                  f"\n\nResearch Problem: {self.active_task.description}"
                  f"\n\n{self.instructions["introduction"]}"
                  f"\n\n{self.instructions["response_format"]}")
        return prompt


API_KEY = read_file("../api_keys/open_ai.txt")

if __name__ == '__main__':
    ml_agent_io = MLAgentIO(task_name="sarcasm_lstm")
    prompt = ml_agent_io.starting_prompt

    # Testing OpenAi connection
    client = OpenAI(api_key=API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": "Tell me a joke about AI"
        }
        ],
        max_tokens=50,
        n=1,
        store=True
    )

    print(response.choices[0].message.content)
