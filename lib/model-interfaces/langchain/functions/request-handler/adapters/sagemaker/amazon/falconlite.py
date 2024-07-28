import json
import os
from typing import Dict, Any

from langchain.llms.sagemaker_endpoint import LLMContentHandler, SagemakerEndpoint
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from ...base import ModelAdapter
from genai_core.registry import registry

class FalconLiteContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: Dict[str, Any]) -> bytes:
        input_str = json.dumps({
            "inputs": prompt,
            "parameters": model_kwargs
        })
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        # Remove "Bot:" prefix if present
        response = response_json[0]["generated_text"]
        return response.lstrip("Bot: ") if response.startswith("Bot: ") else response

content_handler = FalconLiteContentHandler()

class SMFalconLiteAdapter(ModelAdapter):
    def __init__(self, model_id: str, **kwargs):
        self.model_id = model_id
        super().__init__(**kwargs)

    def get_llm(self, model_kwargs: Dict[str, Any] = {}):
        params = {
            "temperature": model_kwargs.get("temperature", 0.9),
            "max_new_tokens": model_kwargs.get("maxTokens", 10000),
            "do_sample": model_kwargs.get("do_sample", True),
            "return_full_text": model_kwargs.get("return_full_text", False),
            "typical_p": model_kwargs.get("typical_p", 0.2),
            "use_cache": model_kwargs.get("use_cache", True),
            "seed": model_kwargs.get("seed", None),
            "top_k": model_kwargs.get("top_k", 50),
            "top_p": model_kwargs.get("top_p", 0.95),
        }

        return SagemakerEndpoint(
            endpoint_name=self.model_id,
            region_name=os.environ["AWS_REGION"],
            content_handler=content_handler,
            model_kwargs=params,
            callbacks=[self.callback_handler],
            streaming=True
        )

    def get_prompt(self):
        template = """You are an AWS Principal Security Engineer AI assistant named Falcon and expert in cybersecurity, AI, and cloud.
        Your responses should be friendly, concise, and natural.
        <|prompter|>Chat history: {chat_history}
        Human: {input}
        <|endoftext|><|assistant|>"""
        input_variables = ["chat_history", "input"]
        return PromptTemplate(template=template, input_variables=input_variables)

    def get_chain(self, model_kwargs: Dict[str, Any] = {}):
        llm = self.get_llm(model_kwargs)
        prompt = self.get_prompt()
        memory = ConversationBufferMemory()
        
        return ConversationChain(
            llm=llm,
            prompt=prompt,
            memory=memory,
            verbose=True
        )

    def generate(self, prompt: str, context: str = "", **kwargs):
        chain = self.get_chain(kwargs)
        return chain.predict(input=prompt)

# Register the adapter
registry.register(r"(?i)sagemaker\..*amazon-FalconLite.", SMFalconLiteAdapter)