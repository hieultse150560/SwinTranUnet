import pickle

for i in range(56,57):
  with open(f"batch{i}.pkl", "rb") as f:
    data = pickle.load(f)
    sens = data.split(".")
    print(len(sens))
