import json
from openai import OpenAI
from typing import List, Dict, Generator


class VllmServer:
    def __init__(
        self, 
        api_key: str = "EMPTY", 
        api_base: str = "http://localhost:8001/v1", 
        model_name: str = None,
        **kwargs):
        """
        Initialize the ChatWithTools class.
        """
        self.client: OpenAI = OpenAI(api_key=api_key, base_url=api_base)
        self.model: str = model_name or self._get_default_model()

    def _get_default_model(self) -> str:
        """
        Get the default model from the API.
        """
        models = self.client.models.list()
        return models.data[0].id if models.data else None


    def stream_chat(self, messages: List[Dict], **kwargs) -> Generator:
        """
        Stream chat completions from the server.

        :param messages: A list of message dictionaries for the chat.
        :param kwargs: Optional parameters including tools for specific models.
        :yield: A stream of chat completion chunks.
        """
        max_token = kwargs.get("max_token",2048)
        temperature = kwargs.get("temperature",0.7)
        stream_completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            max_tokens=max_token,
            temperature=temperature,
            frequency_penalty=1.05,
            response_format={'type':'json_schema'}
        )
            
        # Yield chunks of streamed responses
        for chunk in stream_completion:
            yield chunk

    def chat(self, messages: List[Dict], **kwargs) -> str:
        """
        Chat completions from the server.

        :param messages: A list of message dictionaries for the chat.
        :param kwargs: Optional parameters including tools for specific models.
        """
        max_token = kwargs.get("max_token",1024)
        temperature = kwargs.get("temperature",0.7)
        frequency_penalty = kwargs.get("frequency_penalty",0.7)
        response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_token,
                temperature=temperature,
                frequency_penalty=frequency_penalty,
        )
        return response.choices[0].message.content
        