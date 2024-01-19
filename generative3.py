import subprocess

command = """./main -m ./models/13B/ggml-model-q4_0.bin --n-gpu-layers 15000 -p "Your task is to create a simple description of an image. The description must be simple, in one sentence length. The description should contain 1-3 objects separately placed on the container. Objects must be graspable, do not reuse generated objects in previously generated descriptions. The description only shows the name of objects, don't describe the objects too much. You can also use synonyms of objects to increase diversity, for example, mug, cup, and bottle can be considered synonymous. You must return only the description, nothing else. Don't include any explanation.\nAnswer:" -n 400 -e --temp 1.0"""

result = subprocess.run(command, shell=True, capture_output=True, text=True)
output = result.stdout

print(output)
