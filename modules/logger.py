import os
from datetime import datetime
from ml_agent_io import Task


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
