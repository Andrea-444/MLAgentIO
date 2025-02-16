# MLAgentIO

## Overview

MLAgentIO is an AI-powered agent designed to automate the machine learning experimentation process. It assists users in improving existing ML models or building new models from scratch. The process is fully automated, requiring the user to create a 'Task'.

### What is a Task?

A Task is a structured set of instructions for the agent. It consists of:

- A **task name**: A unique identifier used to reference and run the task.
- A **task description**: A detailed prompt introducing the ML problem and the expected results (either improving an existing model or creating a new one).
- A **set of files**: Datasets, starting scripts, and other necessary resources for the ML problem.

### Features

- **Task Creation**: Users can create custom tasks using `.create_task()`.
- **Full Automation**: The agent can run in full auto mode with `auto=True`, eliminating the need for manual confirmations.
- **Iteration Control**: Users can define a maximum number of iterations before forced termination.
- **Evaluation**: The agent assesses each task run and stores results in `evaluation/agent_performance.txt`.
- **Test-Agnostic Environment**: Each task runs in a separate test environment (`environment/{task_name}_{execution_date}_{execution_time}`), ensuring reproducibility and preventing modifications to the original files.

## How to Use MLAgentIO

1. **Clone the repository**
   ```sh
   git clone https://github.com/Itonkdong/MLAgentIO
   cd MLAgentIO
   ```

2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

3. **Run the agent**
   - Open `main.py` in the `modules` directory.
   - Load an OpenAI API key (required for running the agent).
   - Specify the assistant model (e.g., `'gpt-4o-mini'`).
   - Create a new task with `.create_task()` or run an existing one using `.run_task(task_name)`.
   - Enable full automation with `auto=True` and set iteration limits as needed.

4. **Execute the agent**
   ```sh
   python modules/main.py
   ```

5. **Review the logs**
   A log file is generated in the `logs` folder:
   ```
   log_{task_name}_{execution_date}_{execution_time}.log
   ```
   This contains a detailed record of the agent’s actions.

## Example Tasks

MLAgentIO includes three example tasks that demonstrate its core capabilities:

- **`sarcasm_lstm`**  
  This task focuses on improving an existing model. The goal is to enhance the baseline accuracy of the provided LSTM-based classification model by **5%**. It showcases the agent’s ability to refine and optimize pre-existing architectures.

- **`toxic_bert`**  
  This task demonstrates the agent’s ability to **create a model from scratch**. It involves building a BERT-based classification model using the `transformers` library, showcasing MLAgentIO's feature of generating models from zero.

- **`informal_multi_modal`**  
  This task highlights the agent's ability to **compare different architectures**. It involves testing two approaches—one using **LSTM and convolutional layers**, and another based on a **pretrained BERT model** from the `transformers` library. The agent evaluates both and determines the better-performing architecture.

These example tasks serve as a starting point for users to understand MLAgentIO’s capabilities in **improving**, **creating**, and **comparing** machine learning models.


## Future Plans

- Extend support for additional LLMs beyond OpenAI.

## License

This project is developed by Viktor Kostadinoski and Andrea Stevanoska from the Faculty of Computer Science and Engineering in Skopje, under the guidance of Professor Sonja Gievska and Teaching and Research Assistant Martina Toshevska. Redistribution and use of this code, with or without modification, are permitted for educational and research purposes, provided proper credit is given to the original authors.

MLAgentIO is inspired by MLAgentBench from Stanford: [MLAgentBench](https://github.com/snap-stanford/MLAgentBench) and the research paper: [MLAgentBench Paper](https://arxiv.org/abs/2310.03302).

## Contributors

- **Professor Sonja Gievska** – Supervisor
- **Teaching and Research Assistant Martina Toshevska** – Supervisor
- **Andrea Stevanoska (Andrea-444)** – Developer
- **Viktor Kostadinoski (Itonkdong)** – Developer

---

For further inquiries or contributions, feel free to open an issue or submit a pull request!

