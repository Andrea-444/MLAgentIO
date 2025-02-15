import os
import shutil
from datetime import datetime

from modules.action_executioner import ActionExecutioner
from modules.action_parser import ActionParser
from modules.evaluator import AgentEvaluator
from modules.llm_assistant import LLMAssistant
from modules.logger import AgentLogger
from modules.low_level_actions import read_file


class Task:
    MAIN_DIR = "../tasks"

    def __init__(self, name: str):
        if name not in tuple(Task.list_all_tasks()):
            raise Exception("Invalid task name")

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
               - Returns the names of the tasks.
       """
        return os.listdir(Task.MAIN_DIR)


class MLAgentIO:
    MAIN_LLM_INSTRUCTIONS_DIR = "../assistants_instructions/main"
    SUPPORTING_LLM_INSTRUCTIONS_DIR = "../assistants_instructions/supporting"
    ENVIRONMENT_DIR = "../environment"

    def __init__(self, api_key: str, assistant_model: str | None = None):
        self.main_instructions: str = self.__build_instructions(self.MAIN_LLM_INSTRUCTIONS_DIR)
        self.supporting_instructions = self.__build_instructions(self.SUPPORTING_LLM_INSTRUCTIONS_DIR)
        self.main_assistant = LLMAssistant(api_key=api_key,
                                           starting_instructions=self.main_instructions,
                                           model=assistant_model
                                           )
        self.supporting_assistant = LLMAssistant(api_key=api_key,
                                                 starting_instructions=self.supporting_instructions,
                                                 model=assistant_model
                                                 )

        self.parser = ActionParser()

        self.executioner = ActionExecutioner(action_mapping=self.parser.DEFAULT_ACTION_MAPPING,
                                             assistant=self.supporting_assistant)

        self.logger = AgentLogger()
        self.evaluator = AgentEvaluator()

    @staticmethod
    def __build_instructions(instructions_dir) -> str:
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

    @staticmethod
    def __choose_task(task_name: str | None) -> Task:
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
        try:
            if task_name is not None:
                return Task(task_name)
            else:
                return MLAgentIO.__choose_task_from_list()
        except Exception as e:
            print(f"Error: {e}")
            return MLAgentIO.__choose_task_from_list()

    @staticmethod
    def __choose_task_from_list():
        """
            Prompts the user to select a task from a list of available tasks.

            Behavior:
                - Retrieves all available tasks.
                - Displays a numbered list of tasks for the user to choose from.
                - Continuously prompts the user until a valid task index is entered.
                - Returns the selected task as a `Task` instance.

            Returns:
                Task: The task chosen by the user.
        """
        all_tasks = Task.list_all_tasks()
        print("Input the number before the task you want to choose:")

        while True:
            for i, task in enumerate(all_tasks):
                print(i, task)
            task_index = int(input())
            if 0 <= task_index < len(all_tasks):
                break

            print("Invalid number. Choose again:")

        chosen_task = Task(all_tasks[task_index])
        return chosen_task

    @staticmethod
    def __setup_task(active_task: Task, run_timestamp_str: str) -> str:
        """
            Sets up the task environment by copying task-related files to a dedicated directory.

            Returns:
                str: The path to the newly created task environment directory.

            Behavior:
                - Creates a unique environment directory for the active task.
                - Copies files and directories from the task's setup directory into the environment directory.
        """
        env_task_dir_name = f"{active_task.name}_{run_timestamp_str}"
        env_task_dir_path = os.path.join(MLAgentIO.ENVIRONMENT_DIR, env_task_dir_name)
        os.makedirs(env_task_dir_path, exist_ok=True)

        content = os.listdir(active_task.get_dir_path())
        for file_name in content:
            source_file_path = os.path.join(active_task.get_dir_path(), file_name)
            destination_file_path = os.path.join(env_task_dir_path, file_name)
            if os.path.isfile(source_file_path):
                shutil.copy(source_file_path, destination_file_path)
            else:
                shutil.copytree(source_file_path, destination_file_path)

        return env_task_dir_path

    @staticmethod
    def __get_research_problem(active_task: Task) -> str:
        """
            Retrieves the research problem description for the active task.

            Returns:
                str: The research problem description formatted as "Research Problem: {description}".
        """
        return f"Research Problem: {active_task.description}"

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
            if task_name in tuple(Task.list_all_tasks()):
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

    def run_task(self, task_name: str | None = None, auto: bool = False, terminate_after: int = 30):
        """
            Runs a task with specified parameters and iterates through multiple steps to achieve the goal.

            Parameters:
                task_name (str | None): The name of the task to execute. If None, the user is prompted to choose a task.
                auto (bool): If True, automatically proceeds with the task. If False, the user is prompted for decisions during execution.
                terminate_after (int): The iteration after which the task should automatically terminate (only relevant if `auto` is True).

            Behavior:
                - Chooses a task if `task_name` is not provided.
                - Sets up the task environment and logs.
                - Initiates the conversation with the assistant and logs the instructions.
                - Iterates through multiple steps, executing actions and capturing observations.
                - Provides an option to manually control whether to continue or terminate the task.
                - Records output and observation logs, including performance metrics and statistics.
                - Automatically terminates if the goal is achieved or if the `auto` flag is set to True and `terminate_after` is reached.
                - Evaluates the agent performance and saves the metrics at the end.

            Returns:
                None
            """
        active_task = self.__choose_task(task_name=task_name)
        run_timestamp_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        task_env_dir_path = self.__setup_task(active_task=active_task, run_timestamp_str=run_timestamp_str)

        self.executioner.setup(task_dir_path=task_env_dir_path)
        self.logger.setup(task_name=active_task.name, log_timestamp_str=run_timestamp_str)

        iteration_index = 1
        output = None
        observation = None
        goal_achieved = False
        while True:
            if iteration_index == 1:
                output = self.main_assistant.initiate_conversation(
                    research_problem=self.__get_research_problem(active_task=active_task))
                self.logger.initial_log(instructions=self.main_instructions,
                                        research_problem=self.__get_research_problem(active_task=active_task))
            else:
                output = self.main_assistant.consult(observation, iteration_index)

            print(f"\n=======Output ({iteration_index})=======:\n", output)
            print("=" * 10)

            action_name, action_args, = self.parser.parse_message(output)
            print("Action:\n", action_name, "\nAction Inputs:\n", action_args)
            print("=" * 10)
            if not auto:
                print(
                    "Should the stated action be executed? "
                    "\nType 't' to force termination on the conversation."
                    "\nType 'end' to forcefully end the agent process."
                    "\nType anything else to proceed".upper())

            if auto:
                if iteration_index == terminate_after:
                    command = 'T'
                else:
                    command = "Proceed"
            else:
                command = input()
            command = command.lower()

            if command.lower() == "end":
                break

            if command == "t":
                output = self.main_assistant.consult("Terminate", iteration_index + 1)
                print(f"\n=======Output ({iteration_index + 1})=======:\n", output)
                print("=" * 10)
                action_name, action_args, = self.parser.parse_message(output)
                print("Action:\n", action_name, "\nAction Inputs:\n", action_args)
                print("=" * 10)

                observation = self.executioner.execute(action_name, action_args)
                print("Observation:\n", observation)

                self.logger.save_log(output, observation)

                goal_achieved = self.parser.parse_final_message(observation)

                self.logger.close()
                break

            observation = self.executioner.execute(action_name, action_args)
            print("Observation:\n", observation)

            self.logger.save_log(output, observation)

            if ActionExecutioner.FINAL_ANSWER_FLAG in observation:
                goal_achieved = self.parser.parse_final_message(observation)
                self.logger.close()
                break

            if not auto:
                print(
                    "Should the agent process continue executing?"
                    "\nType 'end' to forcefully terminate the agent."
                    "\nType anything else to proceed.".upper())

            command = "Proceed" if auto else input()
            command = command.lower()
            if command == "end":
                break

            iteration_index += 1

        self.evaluator.save_performance_metrics(
            task_name=active_task.name,
            main_usage_statistics=self.main_assistant.get_and_reset_usage_statistics(),
            supporting_usage_statistics=self.supporting_assistant.get_and_reset_usage_statistics(),
            goal_achieved=goal_achieved)

    def terminate(self):
        """
            Terminates the MLAgentIO by ending the conversation and shutting down the executioner.

            Behavior:
                - Ends the current conversation with the main assistant.
                - Shuts down the executioner, stopping any ongoing operations.

            Returns:
                None
        """
        self.main_assistant.end_conversation()
        self.executioner.shutdown()
