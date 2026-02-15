import pickle

from dataset import JAAD

with open("datagen/data/jaad_dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

# Top-level keys
print(dataset.keys())

# Train/test split
print("Train videos:", len(dataset["split"]["train_ID"]))
print("Test videos:", len(dataset["split"]["test_ID"]))

# Peek at one video's annotations
vid = list(dataset["annotations"].keys())[0]
print(f"\nVideo: {vid}")
print("Num frames:", dataset["annotations"][vid]["num_frames"])
peds = dataset["annotations"][vid]["ped_annotations"]
print("Pedestrians:", list(peds.keys()))

# The script itself also has a get_stats() method that does exactly this — prints dataset statistics (sequence counts, crossing vs.
# not-crossing label distribution, train/test split). You can run it by adding a call like:

DS = JAAD(data_path=".")
DS.get_stats()
