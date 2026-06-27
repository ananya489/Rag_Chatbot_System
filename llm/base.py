from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """
    Abstract interface that every LLM provider must implement.

    The rest of the application — RAG pipeline, prompt builder,
    Streamlit UI — only ever sees this class. They call generate()
    and receive a string back. They have no idea whether that string
    came from Ollama, Groq, OpenAI, or a mock in a test.

    This is the Strategy Pattern: the algorithm (which LLM to call)
    is separated from the code that uses it.

    To add a new provider:
      1. Create llm/your_provider.py
      2. Subclass BaseLLM
      3. Implement generate() and stream()
      4. Update config.py — nothing else changes
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Send a prompt, receive the full response as a string.
        Used when you want the complete answer before displaying it.
        """
        ...

    @abstractmethod
    def stream(self, prompt: str):
        """
        Send a prompt, receive a generator that yields response chunks.
        Used for streaming token-by-token display in Streamlit.

        Yields:
            str: Each token or chunk as it arrives from the model.
        """
        ...

    def health_check(self) -> bool:
        """
        Optional: verify the LLM backend is reachable before use.
        Default implementation returns True (assume available).
        Subclasses should override this with a real connectivity check.
        """
        return True