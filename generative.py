import openai
import os
import time  
import pickle
from tqdm import tqdm

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
os.environ["OPENAI_API_BASE"] = "https://aic-ivychat.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "77ef8d32367c4a11bf4f71d7e2d5726b"
openai.api_type = "azure"
openai.api_base = "https://aic-ivychat.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = "77ef8d32367c4a11bf4f71d7e2d5726b"

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
  Response:
  """
  history = []
  response,_ = get_completion(prompt, history)
  return response

def checkLength(string_to_dump):
    string_to_dump = (string_to_dump.split("\n"))[:-1] 
    print("Generate: ", len(string_to_dump))
    return len(string_to_dump)

string_to_dump = ""
while True:
    i = 1
    if checkLength(string_to_dump) > 10000:
        with open(f"./data/batch_{i}.pkl", "wb") as f:
            string = pickle.dump("\n".join(string_to_dump.split("\n")[:10000]))
        string_to_dump = "\n".join(string_to_dump[10000:])
        print(f"Writing to ./data/batch_{i}.pkl")
        i += 1
    try:
        samples = ["A desk contains a pen and an apple", "A table contains a set of keys and a knife", "A vase lies on the table", "A silver spoon and a wristwatch rest on a wooden nightstand", "A pair of scissors and a ruler rest on a wooden desk next to an open notebook."]
        samples_text = "\n".join(samples)
        prompt = f"""
You'll need to create a simple description of an image used for training a robot to grasp.

The description has the chance of 80% to generate in the form of 'A <container> contains <object 1>, <object 2>,...'.
The description must be in one sentence and based on the following samples.
Example:```\n{samples_text}\n```

The objects that can be grasped should be small objects such as pencils, watches, spoons, knives, scissors, keys, spoons,... The description should contain 1-3 objects.
Some common containers are corner tables, cabinets, shelves, desks,...

The description MUST BE DIVERSE AND NOT THE SAME AS THE GENERATED ONES. You can use synonyms of objects to increase diversity. For example, mug can be replaced with cup, bottle...
"""
        history = []
        for _ in range(10):
          response, history = get_completion(prompt, history)
          print(response, get_objects(response))
          string_to_dump += response + " " + get_objects(response) + "\n"
          # plot(response)
        time.sleep(30)
    except Exception as e:
        print(e)
        time.sleep(30)
    string_to_dump = (string_to_dump.split("\n"))[:-1] 
    print("Generated: ", len(string_to_dump.split("\n")))
