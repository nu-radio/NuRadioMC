import pickle

with open("lookup_table_60.p", "rb") as f:
    data = pickle.load(f)

print(data)
