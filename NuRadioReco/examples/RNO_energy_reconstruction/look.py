import pickle
lut_file = "./lookup_tables_old/lookup_table_96.p"
with open(lut_file, "rb") as f:
    lut = pickle.load(f)
print(lut['header'])