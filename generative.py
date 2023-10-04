import openai
import os
import time  
import pickle
from tqdm import tqdm

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
os.environ["OPENAI_API_BASE"] = "https://airesidency-grasp-anything.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "2da25ecd69a040bf874bb6921b412606
openai.api_type = "azure"
openai.api_base = "https://airesidency-grasp-anything.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = "2da25ecd69a040bf874bb6921b412606"

def get_completion(prompt, history):
    history.append({"role": "user", "content": prompt})
    response = openai.ChatCompletion.create(engine = "IvyChat_demo", messages=history, temperature = 1.0, max_tokens = 4097, top_p = 0.5, frequency_penalty = 1.0, presence_penalty = 1.0, stop = None)
    message_response = response["choices"][0]["message"]["content"]
    history.append({"role": "system", "content": message_response})
    return message_response, history

def get_objects(response):
  prompt = f"""
Your task is to seperate the objects in a simple description of a image. 

The objects that can be grasped should be small objects such as pencil, watch, spoon, knife, scissors, key, spoon,... The objects must be converted to singular and have 1 word length. For example, you must convert 'keys' to 'key'.

Some common containers are corner table, cabinet, shelf, desk,... You should return only the objects, not the container. 

The results should be in the form [<object 1>, <object 2>,...]. The response must be the list containing the objects, nothing else.

Example:```A metal cabinet contains a pair of scissors and a set of keys -> [scissors, key]
A table contains a set of keys and a knife -> [key, knife]
A set of keys and a pair of sunglasses are lying on top of a wooden nightstand. -> [key, sunglasses]
A red mug on a kitchen counter holds a teaspoon and a sugar packet. -> [teaspoon, packet]
A wooden nightstand contains a phone and a pair of glasses. -> [phone, glasses]
A glass jar containing a set of colored pencils is on top of a wooden table. -> [pencil]```

Description: {response}
Response: """
  history = []
  response,_ = get_completion(prompt, history)
  return response

def checkLength(string_to_dump):
    string_to_dump = (string_to_dump.split("\n"))[:-1] 
    print("Check length: ", len(string_to_dump))
    return len(string_to_dump)

k = 10000
string_to_dump = ""
i = 48
while True:
    if checkLength(string_to_dump) > k:
        with open(f"./data/batch_{i}.pkl", "wb") as f:
            string = pickle.dump("\n".join(string_to_dump.split("\n")[:k]), f)
        string_to_dump = "\n".join(string_to_dump.split("\n")[k:])
        print(f"Writing to ./data/batch_{i}.pkl")
        i += 1
        print("Left: ", checkLength(string_to_dump))
    try:
        samples = ["A desk contains a pen and an apple", "A table contains a set of keys and a knife", "A vase lies on the table", "A silver spoon and a wristwatch rest on a wooden nightstand", "A pair of scissors and a ruler rest on a wooden desk next to an open notebook."]
        samples_text = "\n".join(samples)
        prompt = f"""
Your task is to create a simple description of an image used for training a robot to grasp.

The description must be in one sentence. The description should contain 1-3 objects. You must use all small objects that can be grasped with the same probability, except the objects in previously generated descriptions. 

The description MUST BE DIVERSE AND NOT THE SAME STRUCTURE AS THE GENERATED ONES. You can also use synonyms of objects to increase diversity, for example, mug, cup, and bottle can be considered synonymous. 

You must return only the description, nothing else. 
"""
        history = []
        for _ in range(9):
          response, history = get_completion(prompt, history)
          print(response, get_objects(response))
          string_to_dump += response + " " + get_objects(response) + "\n"
          # plot(response)
        time.sleep(15)
    except Exception as e:
        print(e)
        time.sleep(15)
    new = (string_to_dump.split("\n"))[:-1] 
    print("Generated: ", len(new), "\n")
