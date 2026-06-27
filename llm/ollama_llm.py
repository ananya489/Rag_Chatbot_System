import requests
import json

from llm.base import BaseLLM
import config


class OllamaLLM(BaseLLM):

    def __init__(self):
        self.base_url = config.OLLAMA_BASE_URL.rstrip("/")
        self.model = config.OLLAMA_MODEL
        self.temperature = config.LLM_TEMPERATURE
        self.max_tokens = config.LLM_MAX_TOKENS


    def _payload(self, prompt, stream=False):

        return {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }


    def generate(self, prompt: str):

        url = f"{self.base_url}/api/generate"

        try:

            r = requests.post(
                url,
                json=self._payload(prompt, False),
                timeout=120
            )


            if r.status_code != 200:
                raise RuntimeError(
                    f"Ollama error {r.status_code}: {r.text}"
                )


            data = r.json()

            return data.get("response","")


        except requests.exceptions.ConnectionError:

            raise RuntimeError(
                "Ollama is not running"
            )


    def stream(self,prompt):

        url=f"{self.base_url}/api/generate"


        with requests.post(
            url,
            json=self._payload(prompt,True),
            stream=True
        ) as r:


            for line in r.iter_lines():

                if line:

                    data=json.loads(line)

                    if data.get("done"):
                        break

                    yield data.get("response","")



    def health_check(self):

        try:

            r=requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )


            if r.status_code != 200:
                return False


            models=[
                x["name"]
                for x in r.json()["models"]
            ]


            return self.model in models


        except Exception:

            return False