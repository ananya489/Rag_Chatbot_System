from llm.ollama_llm import OllamaLLM


llm=OllamaLLM()


print("Health:",llm.health_check())


print(
    llm.generate(
        "Explain machine learning in simple words"
    )
)