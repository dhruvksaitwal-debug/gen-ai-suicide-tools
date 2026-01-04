from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

print(client.embeddings.create(
    model="text-embedding-3-small",
    input="hello world"
))