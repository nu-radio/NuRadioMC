import pickle

with open("lookup_table_greenland_80.p", "rb") as f:
    data = pickle.load(f)

print(data)
