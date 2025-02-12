import os
import shutil
from datetime import datetime
import pandas as pd

from modules.low_level_actions import read_file


class Task:
    MAIN_DIR = "../tasks"
    SUBMISSION_FILE_NAME = "submission.txt"

    def __init__(self, name: str):
        self.name = name
        self.description = read_file(Task.MAIN_DIR, name, "description.txt")

    def get_dir_path(self):
        """
           Constructs and returns the directory path for the task setup.

           Returns:
               str: The full path to the task's setup directory.

           Behavior:
               - Joins the main directory (`MAIN_DIR`), the task's name, and the "setup" subdirectory.
       """
        return os.path.join(self.MAIN_DIR, self.name, "setup")

    @staticmethod
    def list_all_tasks():
        """
           Lists all tasks available in the main task directory.

           Returns:
               list: A list of directory names representing the available tasks.

           Behavior:
               - Retrieves a list of all entries in `Task.MAIN_DIR`.
               - Returns the names of the directories (tasks).
       """
        return os.listdir(Task.MAIN_DIR)


class MLAgentIO:
    MAIN_LLM_INSTRUCTIONS_DIR = "../assistants_instructions/main"
    SUPPORTING_LLM_INSTRUCTIONS_DIR = "../assistants_instructions/supporting"
    ENVIRONMENT_DIR = "../environment"

    def __init__(self, task_name: str | None = None):
        self.activation_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.main_instructions: str = self.build_instructions(self.MAIN_LLM_INSTRUCTIONS_DIR)
        self.supporting_instructions = self.build_instructions(self.SUPPORTING_LLM_INSTRUCTIONS_DIR)
        self.all_tasks: [Task] = Task.list_all_tasks()
        self.active_task: Task = self.__choose_task(task_name)
        self.task_env_dir_path = self.__setup_task()

    @staticmethod
    def build_instructions(instructions_dir) -> str:
        """
            Constructs a complete instruction set by reading and combining all instruction files in a directory.

            Parameters:
                instructions_dir (str): The directory containing instruction files.

            Returns:
                str: The combined instructions as a single string.

            Behavior:
                - Retrieves all instruction file names from the specified directory.
                - Sorts the files numerically based on the prefix before the underscore.
                - Reads and concatenates the content of each file with spacing in between.
        """
        instructions_file_names = os.listdir(instructions_dir)
        sorted_instructions_file_names = sorted(instructions_file_names,
                                                key=lambda file_name: int(file_name.split("_")[0]))
        instructions = ""
        for instruction_file_name in sorted_instructions_file_names:
            instructions += read_file(instructions_dir, instruction_file_name)
            instructions += "\n\n"
        return instructions

    def __choose_task(self, task_name: str | None) -> Task:
        """
            Selects a task either by name or by user input.

            Parameters:
                task_name (str | None): The name of the task to be chosen, or None to prompt the user.

            Returns:
                Task: The selected task instance.

            Behavior:
                - If a task name is provided, it creates a `Task` instance for it.
                - If no name is given, it lists all available tasks and prompts the user to select one.
        """
        if task_name is not None:
            return Task(task_name)

        print("Input the number before the task you want to choose:")
        for i, task in enumerate(self.all_tasks):
            print(i, task)
        task_index = int(input())
        chosen_task = Task(self.all_tasks[task_index])
        return chosen_task

    def __setup_task(self) -> str:
        """
            Sets up the task environment by copying task-related files to a dedicated directory.

            Returns:
                str: The path to the newly created task environment directory.

            Behavior:
                - Creates a unique environment directory for the active task.
                - Copies files and directories from the task's setup directory into the environment directory.
        """
        env_task_dir_name = f"{self.active_task.name}_{self.activation_timestamp}"
        env_task_dir_path = os.path.join(MLAgentIO.ENVIRONMENT_DIR, env_task_dir_name)
        os.makedirs(env_task_dir_path, exist_ok=True)

        content = os.listdir(self.active_task.get_dir_path())
        for file_name in content:
            source_file_path = os.path.join(self.active_task.get_dir_path(), file_name)
            destination_file_path = os.path.join(env_task_dir_path, file_name)
            if os.path.isfile(source_file_path):
                shutil.copy(source_file_path, destination_file_path)
            else:
                shutil.copytree(source_file_path, destination_file_path)

        return env_task_dir_path

    def get_main_instructions(self) -> str:
        """
            Retrieves the instructions for the main assistant.

            Returns:
                str: The main assistant's instructions as a string.
        """
        return self.main_instructions

    def get_supporting_instructions(self):
        """
           Retrieves the instructions for the supporting assistant.

           Returns:
               Any: The supporting assistant's instructions.
       """
        return self.supporting_instructions

    def get_research_problem(self) -> str:
        """
            Retrieves the research problem description for the active task.

            Returns:
                str: The research problem description formatted as "Research Problem: {description}".
        """
        return f"Research Problem: {self.active_task.description}"

    def get_active_task(self) -> Task:
        """
            Retrieves the currently active task.

            Returns:
                Task: The active `Task` instance.
        """
        return self.active_task

    def get_task_env_dir_path(self) -> str:
        """
           Retrieves the directory path of the task environment.

           Returns:
               str: The task environment directory path.
       """
        return self.task_env_dir_path

    def get_activation_timestamp(self):
        """
            Retrieves the activation timestamp of the current task.

            Returns:
                str: The timestamp when the task was activated.
        """
        return self.activation_timestamp

    @staticmethod
    def create_task(task_name: str) -> bool:
        """
        Creates a new ML task with the required directory structure and files.

        Parameters:
            task_name (str): Name of the task to be created

        Returns:
            bool: True if task was created successfully, False otherwise

        Directory structure created:
        ../tasks/
        └── task_name/
            ├── description.txt
            └── setup/
                ├── train.py
                └── data/
        """
        try:
            task_dir = os.path.join(Task.MAIN_DIR, task_name)
            if os.path.exists(task_dir):
                print(f"Error: Task '{task_name}' already exists")
                return False

            os.makedirs(task_dir)

            setup_dir = os.path.join(task_dir, "setup")
            data_dir = os.path.join(setup_dir, "data")
            os.makedirs(setup_dir)
            os.makedirs(data_dir)

            train_path = os.path.join(setup_dir, "train.py")
            with open(train_path, 'w') as f:
                f.write("""# Enter your training script here""")

            guidelines_message = """
    Please provide a description for your ML task.

    Guidelines for writing a good task description:
    --------------------------------------------
    1. Clear Objective: State what the model needs to achieve
    2. Performance Targets: Specify required improvements (e.g., 5% accuracy increase)
    3. Constraints: Mention any limitations (e.g., max epochs, memory constraints)
    4. Data Context: Briefly describe the dataset's nature if relevant
    5. Submission Requirements: Specify what and how it needs to be saved
    
    You can also enter it manually in the file, for which you will have to press enter twice (2 empty lines)
    to finish this input and leave the description empty.

    Example description:
    ------------------
    You are given a training script (train.py) for a machine learning model on a specific dataset.
    The model is already implemented using the current hyperparameters in train.py.
    Your objective is to enhance the model's performance by improving the baseline accuracy
    by at least 5% while keeping the number of training epochs within 10 to optimize efficiency.
    You do not know what is the baseline accuracy of this model.
    After training, you must generate a classification report and a confusion matrix for the test set,
    saving them to submission.txt as specified in train.py.

    Enter your task description (press Enter twice to finish):"""

            print(guidelines_message)

            lines = []
            while True:
                line = input()
                if line == "":
                    if lines and lines[-1] == "":
                        break
                lines.append(line)

            description = "\n".join(lines).strip()

            description_path = os.path.join(task_dir, "description.txt")
            with open(description_path, 'w') as f:
                f.write(description)

            completion_message = f"""
    Task '{task_name}' created successfully!
    Location: {task_dir}

    Next steps:
    1. Add your dataset files to the 'data' directory
    2. Implement your training script in train.py"""

            print(completion_message)

            return True

        except Exception as e:
            print(f"Error creating task: {str(e)}")
            return False

    @staticmethod
    def save_performance_metrics(assistant_model: str, requests: int, tokens_spent: int,
                                 money_spent: float, goal_achieved: bool) -> bool:
        """
        Saves agent performance metrics to evaluation/agent_performance.csv.
        Creates the file and directory if they don't exist.

        Parameters:
            assistant_model (str): Name/version of the assistant model
            requests (int): Number of requests made
            tokens_spent (int): Total tokens used
            money_spent (float): Total cost in dollars
            goal_achieved (bool): Whether the task goal was achieved

        Returns:
            bool: True if metrics were saved successfully, False otherwise
        """
        try:
            eval_dir = "../evaluation"
            if not os.path.exists(eval_dir):
                os.makedirs(eval_dir)

            file_path = os.path.join(eval_dir, "agent_performance.csv")

            dtypes = {
                'assistant_model': 'str',
                'requests': 'int64',
                'tokens_spent': 'int64',
                'money_spent': 'float64',
                'goal_achieved': 'bool'
            }

            new_data = {
                'assistant_model': [assistant_model],
                'requests': [requests],
                'tokens_spent': [tokens_spent],
                'money_spent': [money_spent],
                'goal_achieved': [goal_achieved]
            }

            new_df = pd.DataFrame(new_data).astype(dtypes)

            if not os.path.exists(file_path):
                new_df.to_csv(file_path, index=False)
            else:
                df = pd.read_csv(file_path, dtype=dtypes)
                df = pd.concat([df, new_df], ignore_index=True)
                df.to_csv(file_path, index=False)

            return True

        except Exception as e:
            print(f"Error saving performance metrics: {str(e)}")
            return False
