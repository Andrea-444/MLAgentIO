import os
from modules.low_level_actions import read_file

API_KEY = '<KEY>'


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
