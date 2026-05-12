from abc import ABC, abstractmethod

import openai

from config import LLMConfig


class BaseChatModel(ABC):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]

    @abstractmethod
    def generate_response(self, user_input: str) -> str:
        raise NotImplementedError


class OpenAILLM(BaseChatModel):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = openai.OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=300.0,
        )

    def generate_response(self, user_input: str) -> str:
        try:
            self.messages.append({"role": "user", "content": user_input})
            response = self.client.chat.completions.create(
                model=self.config.api_model,
                messages=self.messages,
                temperature=self.config.api_temperature,
            )
            assistant_response = response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": assistant_response})
            return assistant_response
        except Exception as e:
            print(f"OpenAI API call failed: {e}")
            if self.messages and self.messages[-1]["role"] == "user":
                self.messages.pop()
            return f"Failed to generate response: {e}"


class Chatbot:
    def __init__(self, config: LLMConfig):
        self.config = config
        if self.config.use_api_model:
            self.llm_instance = OpenAILLM(config)
        else:
            print("Warning: use_api_model is False, no LLM instance created")
            self.llm_instance = None

    def chat(self, user_input: str) -> str:
        if self.llm_instance is None:
            print("Error: LLM instance is None, cannot generate response")
            return "Error: LLM instance not initialized"
        return self.llm_instance.generate_response(user_input)

    def new_chat(self, user_input: str) -> str:
        self.llm_instance = OpenAILLM(self.config)
        return self.llm_instance.generate_response(user_input)
