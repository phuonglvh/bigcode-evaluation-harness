import os
import requests
import openai
from openai import OpenAI
import json

openai.api_key = os.environ.get("OPENAI_API_KEY")

client = OpenAI(
    organization=os.environ.get('OPENAI_ORGANIZATION_ID'),
    project=os.environ.get('OPENAI_PROJECT_ID'),
)


def openai_chat_completions(messages, model="gpt-4o-mini"):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False
    ).choices[0].message.content
