from openai import OpenAI

api_key = "sk-7kVvoLKWePIdvvirGUXY80MVqAsofr3f37kqqHUw7Pnmil3u"

client = OpenAI(
    api_key=api_key,
    base_url="https://api.chatanywhere.org/v1"
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
