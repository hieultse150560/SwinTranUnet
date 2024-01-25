import subprocess
import re
import time  
import pickle
from tqdm import tqdm

def get_completion():
    command = """./main -m ./models/13B/ggml-model-q4_0.bin --n-gpu-layers 15000 -p "Your task is to create a simple description of an image. The description must be simple, in one sentence length. The description should contain 1-3 objects separately placed on the container. Objects must be graspable, do not reuse generated objects in previously generated descriptions. The description only shows the name of objects, don't describe the objects too much. You can also use synonyms of objects to increase diversity, for example, mug, cup, and bottle can be considered synonymous. You must return only the description, nothing else. Don't include any explanation.\nAnswer:" -n 400 -e --temp 1.0"""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    output = result.stdout
    description = re.findall(r"Answer:\s+(.*?)(?:\.|\]|$)", output, flags=re.IGNORECASE)
    return description[0]

def get_objects(response):
    command = """./main -m ./models/13B/ggml-model-q4_0.bin --n-gpu-layers 15000 -p f""" Your task is to seperate the objects in a simple description of a image. The objects that can be grasped should be small objects such as pencil, watch, spoon, knife, scissors, key, spoon,... The objects must be converted to singular and have 1 word length. For example, you must convert 'keys' to 'key'. Some common containers are corner table, cabinet, shelf, desk,... You should return only the objects, not the container. The results should be in the form [<object 1>, <object 2>,...]. The response must be the list containing the objects, nothing else. Example:A metal cabinet contains a pair of scissors and a set of keys -> [scissors, key] A table contains a set of keys and a knife -> [key, knife] A set of keys and a pair of sunglasses are lying on top of a wooden nightstand. -> [key, sunglasses] A red mug on a kitchen counter holds a teaspoon and a sugar packet. -> [teaspoon, packet] A wooden nightstand contains a phone and a pair of glasses. -> [phone, glasses] A glass jar containing a set of colored pencils is on top of a wooden table. -> [pencil] Description: {description}  Response: """ -n 400 -e
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    output = result.stdout
    objects = re.findall(r"Response:\s+(.*?)(?:\.|\]|$)", output, flags=re.IGNORECASE)
    return objects[0]

def checkLength(string_to_dump):
    string_to_dump = (string_to_dump.split("\n"))[:-1] 
    print("Check length: ", len(string_to_dump))
    return len(string_to_dump)

k = 10000
string_to_dump = ""
i = 68
while True:
    if checkLength(string_to_dump) > k:
        with open(f"./data/batch_{i}.pkl", "wb") as f:
            string = pickle.dump("\n".join(string_to_dump.split("\n")[:k]), f)
        string_to_dump = "\n".join(string_to_dump.split("\n")[k:])
        print(f"Writing to ./data/batch_{i}.pkl")
        i += 1
        print("Left: ", checkLength(string_to_dump))
    try:
        response = get_completion()
        objects = get_objects(response)
        print(response, objects)
        string_to_dump += response + " " + objects + "\n"
    except Exception as e:
        print(e)
    new = (string_to_dump.split("\n"))[:-1] 
    print("Generated: ", len(new), "\n")
