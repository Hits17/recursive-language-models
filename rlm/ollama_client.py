"""
Ollama Client for RLM

A client wrapper that provides OpenAI-compatible API interface
for local Ollama models, specifically optimized for Qwen3.
"""

import os
import json
import requests
from typing import Optional, List, Dict, Any, Generator
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class CompletionResponse:
    """Response from the Ollama completion."""
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    raw_response: Dict[str, Any]


@dataclass
class Message:
    """A chat message."""
    role: str  # 'system', 'user', or 'assistant'
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


class OllamaClient:
    """
    Client for interacting with local Ollama service.
    
    Provides a consistent interface that mirrors OpenAI's API structure,
    making it easy to swap between local and cloud-based models.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        host: Optional[str] = None,
        timeout: int = 300,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 0.9,
        **kwargs
    ):
        """
        Initialize the Ollama client.

        Args:
            model_name: Name of the Ollama model to use (default: from env or 'qwen3:latest')
            host: Ollama API host URL (default: from env or 'http://localhost:11434')
            timeout: Request timeout in seconds
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
        """
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "qwen3:latest")
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.extra_params = kwargs

        # Validate connection on initialization
        self._validate_connection()

    def _validate_connection(self) -> None:
        """Validate that Ollama is running and the model is available."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=10)
            response.raise_for_status()
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            if self.model_name not in model_names:
                # Try without tag
                base_name = self.model_name.split(":")[0]
                matching = [m for m in model_names if m.startswith(base_name)]
                if matching:
                    print(f"[OllamaClient] Using model: {matching[0]}")
                    self.model_name = matching[0]
                else:
                    available = ", ".join(model_names) if model_names else "None"
                    raise ValueError(
                        f"Model '{self.model_name}' not found. Available: {available}"
                    )
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.host}. "
                "Make sure Ollama is running: `ollama serve`"
            )

    def completion(
        self,
        messages: List[Message] | List[Dict[str, str]] | str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> CompletionResponse:
        """
        Generate a completion using the Ollama model.

        Args:
            messages: Either a list of Message objects, list of dicts, or a single string
            model: Override the default model
            temperature: Override the default temperature
            max_tokens: Override the default max tokens
            stream: Whether to stream the response
            **kwargs: Additional parameters to pass to Ollama

        Returns:
            CompletionResponse with the generated content
        """
        # Convert messages to proper format
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif messages and isinstance(messages[0], Message):
            messages = [m.to_dict() for m in messages]

        payload = {
            "model": model or self.model_name,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens,
                "top_p": self.top_p,
                **kwargs.get("options", {}),
            },
        }

        # Add any extra parameters
        for key, value in {**self.extra_params, **kwargs}.items():
            if key not in ["options"]:
                payload[key] = value

        try:
            response = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=self.timeout,
                stream=stream,
            )
            response.raise_for_status()

            if stream:
                return self._handle_stream(response)

            result = response.json()
            return CompletionResponse(
                content=result.get("message", {}).get("content", ""),
                model=result.get("model", self.model_name),
                prompt_tokens=result.get("prompt_eval_count", 0),
                completion_tokens=result.get("eval_count", 0),
                total_tokens=(
                    result.get("prompt_eval_count", 0) +
                    result.get("eval_count", 0)
                ),
                raw_response=result,
            )

        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Request timed out after {self.timeout}s. "
                "Try increasing the timeout or reducing max_tokens."
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to get completion: {e}")

    def _handle_stream(self, response) -> Generator[str, None, None]:
        """Handle streaming response."""
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "message" in data:
                    yield data["message"].get("content", "")
                if data.get("done", False):
                    break

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        **kwargs
    ) -> CompletionResponse:
        """
        Simple generate endpoint (non-chat format).
        Useful for simple completion tasks.

        Args:
            prompt: The prompt to complete
            model: Override the default model
            system: Optional system prompt
            **kwargs: Additional parameters

        Returns:
            CompletionResponse with the generated content
        """
        payload = {
            "model": model or self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.pop("temperature", self.temperature),
                "num_predict": kwargs.pop("max_tokens", self.max_tokens),
                "top_p": self.top_p,
            },
            **kwargs,
        }

        if system:
            payload["system"] = system

        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()

            return CompletionResponse(
                content=result.get("response", ""),
                model=result.get("model", self.model_name),
                prompt_tokens=result.get("prompt_eval_count", 0),
                completion_tokens=result.get("eval_count", 0),
                total_tokens=(
                    result.get("prompt_eval_count", 0) +
                    result.get("eval_count", 0)
                ),
                raw_response=result,
            )

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to generate: {e}")

    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models in Ollama."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=10)
            response.raise_for_status()
            return response.json().get("models", [])
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to list models: {e}")

    def check_health(self) -> bool:
        """Check if Ollama is running and healthy."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


# Convenience function for quick testing
def chat_with_qwen(message: str, system: Optional[str] = None) -> str:
    """
    Quick helper function to chat with Qwen3 via Ollama.

    Args:
        message: User message
        system: Optional system prompt

    Returns:
        Assistant's response content
    """
    client = OllamaClient()
    messages = []
    
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": message})
    
    response = client.completion(messages)
    return response.content


if __name__ == "__main__":
    # Quick test
    print("Testing Ollama client with Qwen3...")
    
    client = OllamaClient()
    print(f"Connected to: {client.host}")
    print(f"Using model: {client.model_name}")
    
    response = client.completion("What is 2 + 2? Reply with just the number.")
    print(f"Response: {response.content}")
    print(f"Tokens used: {response.total_tokens}")
