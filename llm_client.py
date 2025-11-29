import base64
from openai import OpenAI

class LLMClient:
    def __init__(self, api_key, base_url, model_name="gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def chat_completion(self, messages, temperature=0, max_tokens=500):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()

    def describe_image(self, image_bytes: bytes, max_tokens=300) -> str:
        """
        Sends image bytes to the LLM for description.
        Handles base64 encoding internally.
        """
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail so that the output is less than 200 words."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ],
            }
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()