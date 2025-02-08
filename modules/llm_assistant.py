from openai import OpenAI


class LLMAssistant:
    def __init__(self, api_key: str, starting_instructions: str):
        self.starting_instructions = self.to_developer_message(starting_instructions)
        self.history = []
        self.client = OpenAI(api_key=api_key)

    @staticmethod
    def to_developer_message(instruction: str) -> dict:
        return {"role": "developer", "content": instruction}

    @staticmethod
    def to_user_message(message: str) -> dict:
        return {"role": "user", "content": message}

    @staticmethod
    def to_assistant_message(response: str) -> dict:
        return {"role": "assistant", "content": response}

    def build_context(self, observation: str) -> list:
        observation_message = self.to_user_message(observation)
        context = [self.starting_instructions] + self.history + [observation_message]
        self.history.append(observation_message)
        return context

    def __ask_assistant(self, context: list, model="gpt-4o-mini", max_tokens=None) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=context,
            max_tokens=max_tokens,
            n=1,
            store=True
        )

        return response.choices[0].message.content

    def initiate_conversation(self, research_problem: str):
        research_problem_message = self.to_user_message(research_problem)
        initial_context = [self.starting_instructions, research_problem_message]
        self.print_context(initial_context)
        self.history.append(research_problem_message)
        output = self.__ask_assistant(initial_context)
        self.history.append(self.to_assistant_message(output))
        return output

    def consult(self, observation: str, observation_index: int) -> str:
        full_observation = (f"Iteration: {observation_index}\n"
                            f"Observation:\n{observation}")
        context = self.build_context(full_observation)
        output = self.__ask_assistant(context)
        self.history.append(self.to_assistant_message(output))
        return output

    def consult_once(self, script_content: str, instructions: str) -> str:
        message = self.to_user_message(f"{instructions}\n\nScript Content:\n{script_content}")
        context = [self.starting_instructions, message]
        output = self.__ask_assistant(context)
        return output

    def print_history(self):
        print("=" * 50, "HISTORY", "=" * 50)
        for message in self.history:
            self.print_message(message)

    @staticmethod
    def print_message(message: dict):
        print("ROLE:", message["role"])
        print("CONTENT:\n", message["content"])
        print("=" * 100)

    @staticmethod
    def print_context(context: list):
        for message in context:
            print(message["content"])

    def end_conversation(self):
        self.client.close()
