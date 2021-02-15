import json

file_path = './confs/training_conf.json'

with open(file_path, 'r') as j:
    json_file = json.loads(j.read())

training_conf = json_file["mcd_training_conf"] 

print(training_conf["path"])
print(training_conf["model"])
print(training_conf["batch_size"])
print(training_conf["epochs"])
print(training_conf["lrs"])
print(training_conf["ps"])

print(type(training_conf["path"]))
print(type(training_conf["model"]))
print(type(training_conf["batch_size"]))
print(type(training_conf["epochs"]))
print(type(training_conf["lrs"]))
print(type(training_conf["ps"]))