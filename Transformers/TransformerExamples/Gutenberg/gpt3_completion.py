import os
from openai import OpenAI

# Get API key from environment
my_api_key = os.getenv('OPENAI_API_KEY')

# Print the API key for verification (ensure this is removed in production for security)
print(f"API_KEY : {my_api_key}")



client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Original: She no went to the market.\nStandard American English:"
        }
    ],
    # model="gpt-3.5-turbo",
    model="gpt-3.5-turbo-16k"
)
print(chat_completion)

#https://medium.com/@Doug-Creates/nightmares-and-client-chat-completions-create-29ad0acbe16a
#https://platform.openai.com/docs/examples