from openai import OpenAI

api_key = "your-api-key"
base_url= "your-base-url"

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "how are you?",
        }
    ],
    model="gpt-3.5-turbo",
)

print(chat_completion.choices[0].message.content)
# ok
