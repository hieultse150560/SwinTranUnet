import subprocess
import re

command = """./main -m ./models/13B/ggml-model-q4_0.bin --n-gpu-layers 15000 -p "Your task is to create a simple description of an image. The description must be simple, in one sentence length. The description should contain 1-3 objects separately placed on the container. Objects must be graspable, do not reuse generated objects in previously generated descriptions. The description only shows the name of objects, don't describe the objects too much. You can also use synonyms of objects to increase diversity, for example, mug, cup, and bottle can be considered synonymous. You must return only the description, nothing else. Don't include any explanation.\nAnswer:" -n 400 -e --temp 1.0"""

result = subprocess.run(command, shell=True, capture_output=True, text=True)
output = result.stdout

description = re.findall(r"Answer:\s+(.*?)(?:\.|\]|$)", output, flags=re.IGNORECASE)

command = """./main -m ./models/13B/ggml-model-q4_0.bin --n-gpu-layers 15000 -p f""" Your task is to seperate the objects in a simple description of a image. The objects that can be grasped should be small objects such as pencil, watch, spoon, knife, scissors, key, spoon,... The objects must be converted to singular and have 1 word length. For example, you must convert 'keys' to 'key'. Some common containers are corner table, cabinet, shelf, desk,... You should return only the objects, not the container. The results should be in the form [<object 1>, <object 2>,...]. The response must be the list containing the objects, nothing else. Example:A metal cabinet contains a pair of scissors and a set of keys -> [scissors, key] A table contains a set of keys and a knife -> [key, knife] A set of keys and a pair of sunglasses are lying on top of a wooden nightstand. -> [key, sunglasses] A red mug on a kitchen counter holds a teaspoon and a sugar packet. -> [teaspoon, packet] A wooden nightstand contains a phone and a pair of glasses. -> [phone, glasses] A glass jar containing a set of colored pencils is on top of a wooden table. -> [pencil] Description: {description}  Response: """ -n 400 -e

result = subprocess.run(command, shell=True, capture_output=True, text=True)
output = result.stdout

objects = re.findall(r"Response:\s+(.*?)(?:\.|\]|$)", output, flags=re.IGNORECASE)

print(output)
print(description[0])
print(objects[0])
