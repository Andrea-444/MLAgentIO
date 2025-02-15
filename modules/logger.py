import os


class AgentLogger:
    LOGS_DEFAULT_DIR = "../logs"

    def __init__(self, logs_dir_path: str = None):
        if logs_dir_path is None:
            logs_dir_path = AgentLogger.LOGS_DEFAULT_DIR

        self.logs_dir_path = logs_dir_path
        self.logs_file = None
        self.step = 0

    def setup(self, task_name: str, log_timestamp_str: str):
        """
           Initializes the logging setup for a specific task.

           Parameters:
               task_name (str): The name of the task for which logs are being created.
               log_timestamp_str (str): A timestamp string to ensure unique log file names.

           Behavior:
               - Constructs a log file name using the task name and timestamp.
               - Creates the default logs directory if it does not exist.
               - Opens a new log file in write mode and sets it as the active log file.
               - Initializes the step counter to track log entries.
       """
        log_file_name = f"log_{task_name}_{log_timestamp_str}.txt"
        log_file_path = os.path.join(self.logs_dir_path, log_file_name)

        os.makedirs(self.LOGS_DEFAULT_DIR, exist_ok=True)

        self.logs_file = open(log_file_path, mode="w", encoding="utf-8")
        self.step = 0

    def initial_log(self, instructions: str, research_problem: str):
        """
            Writes the initial log containing instructions and the research problem.

            Parameters:
                instructions (str): The instructions for the research or task.
                research_problem (str): The research problem or topic being addressed.

            Behavior:
                - Writes the provided instructions and research problem to the log file.
        """
        if self.logs_file is None:
            print(f"Error initiating logger: {self.__class__.__name__} not properly setup. Missing log file")

        try:
            self.logs_file.write(instructions)
            self.logs_file.write(research_problem)
        except Exception as e:
            print(f"Error initiating logger: {e}")

    def save_log(self, output: str, observation: str):
        """
            Saves a step-by-step log entry with the assistant's output and the user's observation.

            Parameters:
                output (str): The assistant's response or generated content.
                observation (str): The observation made during this step.

            Behavior:
                - Writes a formatted step entry to the log file.
                - Includes the assistant's output and observation in the log.
                - Increments the step counter.
        """
        self.logs_file.write(f"\n\nStep {self.step}:\n\n")
        self.logs_file.write(output)
        self.logs_file.write("\nObservation: \n'''\n")
        self.logs_file.write(observation)
        self.logs_file.write("\n'''")
        self.step += 1

    def close(self):
        """
            Closes the log file.

            Behavior:
                - Ensures that all written content is saved and releases the file resource.
        """
        self.logs_file.close()
