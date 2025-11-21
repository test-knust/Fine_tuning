import ollama
from ollama import ResponseError

# Ensure you have run 'ollama pull llama3' (or your chosen model) in terminal first

print("--- 1. Simple Generation ---")
response = ollama.generate(model='llama3', prompt='Why is the sky blue?')
print(response['response'])

print("\n--- 2. Chat with History ---")
messages = [
    {'role': 'user', 'content': 'Hi, I am a student named Alex.'},
    {'role': 'assistant', 'content': 'Hello Alex! How can I help you with your studies today?'},
    {'role': 'user', 'content': 'What is my name?'}
]

chat_response = ollama.chat(model='llama3', messages=messages)
print(f"AI: {chat_response['message']['content']}")

print("\n--- 3. Using a Custom Model (from Lab Part 2) ---")
# Only works if you completed the 'grumpy-coder' step in the lab
try:
    grumpy_response = ollama.chat(
        model='grumpy-coder',
        messages=[{'role': 'user', 'content': 'Help me install Python.'}]
    )
    print(f"Grumpy AI: {grumpy_response['message']['content']}")
except ResponseError:
    print("Could not find model 'grumpy-coder'. Did you run the 'ollama create' step?")
