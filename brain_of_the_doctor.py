#Step 1 : Setup groq API key
from dotenv import load_dotenv
load_dotenv()

import os
GROQ_API_KEY= os.environ.get("GROQ_API_KEY")

#Step 2: Convert Image into Required Format
import base64              # converts bits/bytes to string # basic encoding/decoding


#image_path="acne.jpeg"
def encode_image(image_path):
    image_file=open(image_path,"rb")
    return base64.b64encode(image_file.read()).decode('utf-8')


#Step 3: Setup Multimodal LLM

from groq import Groq
query= "is there something wrong with my face"
model="meta-llama/llama-4-scout-17b-16e-instruct"

def analyze_image_with_query(query,model,encoded_image):
    client=Groq()
    
    messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": query
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}",
                        },
                    },
                ],
            }]
    chat_completion=client.chat.completions.create(
        messages=messages,
        model=model
    )
    #print(chat_completion)

    return chat_completion.choices[0].message.content
