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

def dialog(uinput):
   #preparing the prompt for OpenAI
   role="user"

   #prompt="Where is Tahiti located?" #maintenance or if you do not want to use a microphone
   line = {"role": role, "content": uinput}

   #creating the mesage
   assert1={"role": "system", "content": "You are a Natural Language Processing Assistant."}
   assert2={"role": "assistant", "content": "You are helping viewers analyze social medial better."}
   assert3=line
   iprompt = []
   iprompt.append(assert1)
   iprompt.append(assert2)
   iprompt.append(assert3)

   #sending the message to ChatGPT
   response = client.chat.completions.create(model="gpt-4",messages=iprompt) #ChatGPT dialog
   text=response.choices[0].message.content

   return text

if False:
    uinput="Provide the list of labels for Semantic Role Labeling"
    text=dialog(uinput) #preparing the messages for ChatGPT
    print("Viewer request",uinput)
    print("ChatGPT Sentiment Analysis:",text)

    uinput="Perform Semantic Role Labeling on the following sentence:Did Bob really think he could prepare a meal for 50 people in only a few hours?"
    text=dialog(uinput) #preparing the messages for ChatGPT
    print("Viewer request",uinput)
    print("ChatGPT Sentiment Analysis:",text)

uinput="Perform Semantic Role Labeling on the following sentence:Alice, whose husband went jogging every Sunday, liked to go to a dancing class in the meantime."
text=dialog(uinput) #preparing the messages for ChatGPT
print("Viewer request",uinput)
print("ChatGPT Sentiment Analysis:",text)    