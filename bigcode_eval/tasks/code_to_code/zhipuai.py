from zhipuai import ZhipuAI
client = ZhipuAI(api_key="")
response = client.chat.completions.create(
    model="codegeex-4",
    messages=[
        {
            "role": "system",
            "content": "You are an intelligent programming assistant named CodeGeeX. You will answer any questions related to programming, code, and computers, providing well-formatted, executable, accurate, and safe code, and detailed explanations when necessary. Task: Please provide well-formatted comments for the input code, including both multi-line and single-line comments. Please ensure not to modify the original code, only add comments. Please respond in Chinese."
        },
        {
            "role": "user",
            "content": "Write a quicksort function"
        }
    ],
    top_p=0.7,
    temperature=0.9,
    max_tokens=1024,
    stop=["<|endoftext|>", "<|user|>", "<|assistant|>", "<|observation|>"]
)
print(response.choices[0].message)
