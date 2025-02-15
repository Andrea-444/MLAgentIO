from openai import OpenAI
from evaluator import UsageStatistics


class LLMAssistant:

    def __init__(self, api_key: str, starting_instructions: str, model=None):
        self.starting_instructions = self.to_developer_message(starting_instructions)
        self.history = []
        self.model = model if model else "gpt-4o-mini"
        self.client = OpenAI(api_key=api_key)
        self.usage_statistics = UsageStatistics(self.model)

    @staticmethod
    def to_developer_message(instruction: str) -> dict:
        """
            Creates a structured message representing a developer's input.

            Parameters:
                instruction (str): The instruction or message from the developer.

            Returns:
                dict: A dictionary with the role set to "developer" and the content as the provided instruction.
        """
        return {"role": "developer", "content": instruction}

    @staticmethod
    def to_user_message(message: str) -> dict:
        """
            Creates a structured message representing a user's input.

            Parameters:
                message (str): The message from the user.

            Returns:
                dict: A dictionary with the role set to "user" and the content as the provided message.
        """
        return {"role": "user", "content": message}

    @staticmethod
    def to_assistant_message(response: str) -> dict:
        """
            Creates a structured message representing the assistant's response.

            Parameters:
                response (str): The response from the assistant.

            Returns:
                dict: A dictionary with the role set to "assistant" and the content as the provided response.
        """
        return {"role": "assistant", "content": response}

    @staticmethod
    def print_message(message: dict):
        """
            Displays the conversation history in a readable format.

            Behavior:
                - Prints a separator followed by the stored conversation history.
                - Calls 'print_message' for each message in the history.
        """
        print("ROLE:", message["role"])
        print("CONTENT:\n", message["content"])
        print("=" * 100)

    @staticmethod
    def print_context(context: list):
        """
            Prints the content of all messages in a given conversation context.

            Parameters:
                context (list): A list of messages to be printed.

            Behavior:
                - Iterates through the context and prints each message's content.
        """
        for message in context:
            print(message["content"])

    def __build_context(self, observation: str) -> list:
        """
            Constructs the conversation context by incorporating past interactions.

            Parameters:
                observation (str): The latest observation or input from the user.

            Returns:
                list: A list of messages representing the conversation context, including the starting instructions,
                      conversation history, and the new observation.

            Behavior:
                - Converts the observation into a user message.
                - Includes starting instructions and past interactions.
                - Appends the new observation to the conversation history.
        """
        observation_message = self.to_user_message(observation)
        context = [self.starting_instructions] + self.history + [observation_message]
        self.history.append(observation_message)
        return context

    def __ask_assistant(self, context: list, max_tokens=None) -> str:
        """
            Sends the conversation context to the assistant model and retrieves a response.

            Parameters:
                context (list): A list of messages forming the conversation history.
                max_tokens (int, optional): The maximum number of tokens the response can contain.

            Returns:
                str: The assistant's response to the provided context.

            Behavior:
                - Uses the assistant client to generate a response based on the given context.
                - Stores the response for future interactions.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=context,
            max_tokens=max_tokens,
            n=1,
            store=True
        )

        self.usage_statistics.update(response)

        return response.choices[0].message.content

    def initiate_conversation(self, research_problem: str):
        """
            Starts a new conversation with the assistant based on a research problem.

            Parameters:
                research_problem (str): The research question or topic to discuss.

            Returns:
                str: The assistant's initial response.

            Behavior:
                - Converts the research problem into a user message.
                - Initializes the conversation context with the starting instructions.
                - Calls the assistant for a response.
                - Updates the conversation history with the new interaction.
        """
        research_problem_message = self.to_user_message(research_problem)
        initial_context = [self.starting_instructions, research_problem_message]
        self.print_context(initial_context)
        self.history.append(research_problem_message)
        output = self.__ask_assistant(initial_context)
        self.history.append(self.to_assistant_message(output))
        return output

    def consult(self, observation: str, observation_index: int) -> str:
        """
            Engages the assistant in a consultation session using an observation.

            Parameters:
                observation (str): The input or finding to be analyzed.
                observation_index (int): The iteration number of the observation.

            Returns:
                str: The assistant's response to the observation.

            Behavior:
                - Formats the observation with its iteration index.
                - Builds the conversation context and sends it to the assistant.
                - Stores the response in the conversation history.
        """
        full_observation = (f"Iteration: {observation_index}\n"
                            f"Observation:\n{observation}")
        context = self.__build_context(full_observation)
        output = self.__ask_assistant(context)
        self.history.append(self.to_assistant_message(output))
        return output

    def consult_once(self, script_content: str, instructions: str) -> str:
        """
           Performs a single consultation with the assistant using script content and instructions.

           Parameters:
               script_content (str): The script or code to be analyzed.
               instructions (str): The guidelines or instructions for processing the script.

           Returns:
               str: The assistant's response.

           Behavior:
               - Constructs a user message combining the instructions and script content.
               - Calls the assistant with the generated context.
       """
        message = self.to_user_message(f"{instructions}\n\nScript Content:\n{script_content}")
        context = [self.starting_instructions, message]
        output = self.__ask_assistant(context)
        return output

    def print_history(self):
        """
            Displays the conversation history in a readable format.

            Behavior:
                - Prints a separator followed by the stored conversation history.
                - Calls 'print_message' for each message in the history.
        """
        print("=" * 50, "HISTORY", "=" * 50)
        for message in self.history:
            self.print_message(message)

    def get_usage_statistics(self) -> UsageStatistics:
        """
            Retrieves the current usage statistics.

            Returns:
                UsageStatistics: The current usage statistics instance.

            Behavior:
                - Provides access to tracked values such as input tokens, output tokens, and request count.
        """
        return self.usage_statistics

    def reset_usage_statistics(self):
        """
            Resets the usage statistics.

            Behavior:
                - Reinitializes the `UsageStatistics` instance with the model.
                - Clears all tracked values (input tokens, output tokens, and request count).
        """
        self.usage_statistics = UsageStatistics(self.model)

    def get_and_reset_usage_statistics(self) -> UsageStatistics:
        """
            Retrieves and resets the usage statistics.

            Returns:
                UsageStatistics: The usage statistics before resetting.

            Behavior:
                - Retrieves the current usage statistics.
                - Resets the statistics immediately after retrieval.
                - Ensures that new tracking sessions start fresh.
        """
        statistics = self.get_usage_statistics()
        self.reset_usage_statistics()
        return statistics

    def end_conversation(self):
        """
            Closes the assistant's client connection, ending the conversation.

            Behavior:
                - Calls the 'close' method of the assistant client.
        """
        self.client.close()
