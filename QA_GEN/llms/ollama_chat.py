import os
import requests
import json

from .base import LLM
from .utils import retry_with_exponential_backoff

class OllamaChat(LLM):
    TOP_LOGPROBS = 1

    def __init__(self, model_name='llama3.2', base_url = None) -> None:
        if base_url is None:
            self.base_url = os.getenv('OLLAMA_API_URL', 'http://localhost:11434')
        else:
            self.base_url = base_url
        self.model_name = model_name
        # Ensure the model is available
        self._check_model_availability()

    def _check_model_availability(self):
        """Check if the model is available, if not try to pull it"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'].split(':')[0] for model in models]
                if self.model_name not in model_names:
                    print(f"Model {self.model_name} not found locally. Attempting to pull...")
                    self._pull_model()
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not check model availability: {e}")

    def _pull_model(self):
        """Pull the model if it's not available locally"""
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model_name},
                stream=True
            )
            if response.status_code == 200:
                print(f"Successfully pulled model {self.model_name}")
            else:
                print(f"Failed to pull model {self.model_name}")
        except requests.exceptions.RequestException as e:
            print(f"Error pulling model: {e}")

    def _count_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters for most models)"""
        return len(text) // 4

    @retry_with_exponential_backoff
    def __call__(self, prompt: str, max_tokens: int = 1024, temperature=0.0, **kwargs) -> tuple[str, dict]:
        payload = {
            "model": self.model_name,
            "prompt": prompt,  # 使用 prompt 而不是 messages
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
            }
        }
        
        # Add any additional options from kwargs
        if kwargs:
            payload["options"].update(kwargs)

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",  # 使用 generate 而不是 chat
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            
            result = response.json()
            res_text = result["response"]  # generate API 回傳的是 "response" 欄位
            
            # Estimate token counts since Ollama doesn't provide exact counts
            num_input_tokens = self._count_tokens(prompt)
            num_output_tokens = self._count_tokens(res_text)
            
            # Create mock logprobs structure to match OpenAI format
            # Ollama doesn't provide logprobs by default, so we create a placeholder
            mock_logprobs = []
            if res_text:
                # Split into rough tokens (words and punctuation)
                tokens = res_text.replace(' ', ' ').split()
                for token in tokens:
                    if token:  # Skip empty tokens
                        mock_logprobs.append([{
                            "token": token,
                            "logprob": -1.0  # Placeholder logprob
                        }])
            
            res_info = {
                "input": prompt,
                "output": res_text,
                "num_input_tokens": num_input_tokens,
                "num_output_tokens": num_output_tokens,
                "logprobs": mock_logprobs
            }
            
            return res_text, res_info
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama API request failed: {e}")
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse Ollama response: {e}")

if __name__ == "__main__":
    from pprint import pprint
    llm = OllamaChat()
    res_text, res_info = llm(prompt="Say apple!")
    print(res_text)
    print()
    pprint(res_info)