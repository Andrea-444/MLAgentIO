import os
from datetime import datetime
from ml_agent_io import Task


class AgentLogger:
    LOGS_DEFAULT_DIR = "../logs"

    def __init__(self, task_name: str, log_timestamp: datetime, logs_dir_path: str = None, ):
        if logs_dir_path is None:
            logs_dir_path = AgentLogger.LOGS_DEFAULT_DIR

        log_file_name = f"log_{task_name}_{log_timestamp}.txt"
        log_file_path = os.path.join(logs_dir_path, log_file_name)

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
        self.logs_file.write(instructions)
        self.logs_file.write(research_problem)

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
