# MLAgentIO

## Overview
MLAgentIO is an AI-powered agent designed to automate the machine learning experimentation process. It assists users in improving existing ML models or building new models from scratch. The process is fully automated, requiring the user to create a 'Task'.

### What is a Task?
A Task is a structured set of instructions for the agent. It consists of:
- A **task description**: A detailed prompt introducing the ML problem and the expected results (either improving an existing model or creating a new one).
- A **set of files**: Datasets, starting scripts, and other necessary resources for the ML problem.

### Current Features
Since the project is in its early stages, there are **three predefined tasks** available for testing. Users **cannot yet create their own tasks** but can experiment with the existing ones to understand how the agent operates.

## How to Use MLAgentIO
To test the agent, follow these steps:

1. **Clone the repository**
   ```sh
   git clone https://github.com/Itonkdong/MLAgentIO
   cd MLAgentIO
   ```

2. **Install dependencies**
   Ensure all required packages are installed:
   ```sh
   pip install -r requirements.txt
   ```

3. **Run the agent**
   - Open `main.py` located in the `modules` directory.
   - Load an OpenAI API key (required for running the agent).
   - Specify the assistant model to use (e.g., `'gpt-4o-mini'`).
   - Specify the task to run. If unsure, set `task_name = None` to be prompted with a list of available tasks.
   
4. **Execute the agent**
   Run the main module and follow on-screen instructions to confirm execution. This ensures transparency and user control during testing. In the release version, the agent will execute fully automatically.
   ```sh
   python modules/main.py
   ```

5. **Review the logs**
   After execution, a log file is created in the `logs` folder:
   ```
   log_{task_name}_{execution_date}_{execution_time}.log
   ```
   This file contains a detailed record of the agent’s operations.

## Future Plans
- Enable users to create and add their own personalized tasks.
- Expand the use of various LLMs beyond just those from OpenAI.
- Conduct performance testing to assess and optimize the agent's capabilities.
- Eliminate the confirmation commands currently embedded within the agent, which are useful for testing but hinder full automation. Removing these commands will allow for complete automation of the agent.

## License
This project is developed as part of a team effort by Viktor Kostadinoski and Andrea Stevanoska from the Faculty of Computer Science in Skopje, under the guidance of Professor Sonja Gievska and Teaching and Research Assistant Martina Toshevska. All rights reserved. Redistribution and use of this code, with or without modification, are permitted for educational and research purposes, provided proper credit is given to the original authors.

This project was highly inspired by MLAgentBench from Stanford, which can be found on GitHub: [MLAgentBench](https://github.com/snap-stanford/MLAgentBench). It is based on the research presented in the following paper: [MLAgentBench Paper](https://arxiv.org/abs/2310.03302)

## Contributors
- **Professor Sonja Gievska** – Supervisor
- **Teaching and Research Assistant Martina Toshevska** – Supervisor
- **Andrea Stevanoska (Andrea-444)** – Developer
- **Viktor Kostadinoski (Itonkdong)** – Developer


---
For further inquiries or contributions, feel free to open an issue or submit a pull request!

