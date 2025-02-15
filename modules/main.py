from modules.ml_agent_io import MLAgentIO
from modules.low_level_actions import read_file

if __name__ == '__main__':
    # ====================== SETUP ======================

    # Load the OpenAI API key from a file.
    # This key is required for interacting with OpenAI's API and using the assistant model.
    # API_KEY = read_file("{a path to an OpenAI API key}")
    API_KEY = read_file("../api_keys/open_ai.txt")

    # Specify the assistant model to use.
    # The model determines how well the agent will respond during the task.
    # If set to None, the default model 'gpt-4o-mini' will be used.
    assistant_model = None

    # Initialize the MLAgentIO object with the provided API key and assistant model.
    ml_agent_io = MLAgentIO(api_key=API_KEY, assistant_model=assistant_model)

    # ====================== EXECUTION ======================
    # Set the task name.
    # If task_name is set to None, the user will be prompted to choose from available tasks.
    task_name = "{the task name}"

    # Run the chosen task with the 'auto' flag set to True and terminate after 12 iterations.
    # This will allow the agent to execute the task without manual intervention and stop after 12 iterations.
    ml_agent_io.run_task(task_name=task_name, auto=True, terminate_after=12)

    #     OR

    # Alternatively, you can create and set up a new task
    ml_agent_io.create_task(task_name=task_name)
    # and then run it.
    ml_agent_io.run_task(task_name=task_name, auto=True, terminate_after=12)

    # ====================== TERMINATION =====================

    # At the end of the task, terminate the agent.
    # This will gracefully shut down the conversation and execution process.
    ml_agent_io.terminate()

