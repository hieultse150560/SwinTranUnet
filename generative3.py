import subprocess

command = "./main -m ./models/13B/ggml-model-q4_0.bin -p 'Building a website can be done in 10 simple steps:\nStep 1:' -n 400 -e"

result = subprocess.run(command, shell=True, capture_output=True, text=True)
output = result.stdout

print(output)
