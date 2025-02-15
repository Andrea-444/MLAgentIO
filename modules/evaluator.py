import os
import pandas as pd
from openai.types.chat import ChatCompletion


class UsageStatistics:
    def __init__(self, model):
        self.model = model
        self.input_tokens = 0
        self.output_tokens = 0
        self.requests = 0

    def update(self, response: ChatCompletion):
        """
            Updates usage statistics with data from a model response.

            Parameters:
                response (ChatCompletion): The response object containing token usage details.

            Behavior:
                - Extracts the number of input (prompt) and output (completion) tokens from the response.
                - Increments the total input and output token counts.
                - Increments the total request count.
        """
        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens

        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.requests += 1


class AgentEvaluator:

    # As of Feb, 2025
    PRICING_PER_MILLION_TOKENS = {
        "gpt-3.5-turbo": {"input": 3.00, "output": 6.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "o1-preview": {"input": 15.00, "output": 60.00},
        "o1-mini": {"input": 5.00, "output": 20.00}
    }
    EVALUATION_DIR = "../evaluation"

    @staticmethod
    def save_performance_metrics(task_name: str, main_usage_statistics: UsageStatistics,
                                 supporting_usage_statistics: UsageStatistics,
                                 goal_achieved: bool) -> bool:
        """
        Saves agent performance metrics to evaluation/agent_performance.csv.
        Creates the file and directory if they don't exist.

        Parameters:
            task_name (str): Task name
            main_usage_statistics (UsageStatistics): Main Assistant usage statistics
            supporting_usage_statistics (UsageStatistics): Supporting Assistant usage statistics
            goal_achieved (bool): Whether the task goal was achieved

        Returns:
            bool: True if metrics were saved successfully, False otherwise
        """
        try:
            assistant_model = main_usage_statistics.model
            total_requests = 0
            money_spent = 0
            tokens_spent = 0

            for usage_statistic in [main_usage_statistics, supporting_usage_statistics]:
                model = usage_statistic.model
                total_requests += usage_statistic.requests
                input_tokens = usage_statistic.input_tokens
                output_tokens = usage_statistic.output_tokens

                input_cost = (input_tokens / 1_000_000) * AgentEvaluator.PRICING_PER_MILLION_TOKENS[model]["input"]
                output_cost = (output_tokens / 1_000_000) * AgentEvaluator.PRICING_PER_MILLION_TOKENS[model]["output"]

                money_spent += input_cost + output_cost
                tokens_spent += input_tokens + output_tokens

            os.makedirs(AgentEvaluator.EVALUATION_DIR, exist_ok=True)

            file_path = os.path.join(AgentEvaluator.EVALUATION_DIR, "agent_performance.csv")

            dtypes = {
                'task_name': 'str',
                'assistant_model': 'str',
                'requests': 'int64',
                'tokens_spent': 'int64',
                'money_spent': 'float64',
                'goal_achieved': 'bool'
            }

            new_data = {
                'task_name': [task_name],
                'assistant_model': [assistant_model],
                'requests': [total_requests],
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
