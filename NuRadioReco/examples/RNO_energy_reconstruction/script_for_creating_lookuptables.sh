# mkdir Final_lookups

# python create_lookup_table.py 96 --d_r 2 --d_z 2 --output_path ./lookup_tables_old_2

# python create_lookup_table.py 97 --d_r 2 --d_z 2 --output_path ./lookup_tables_old_2

# python create_lookup_table.py 95 --d_r 2 --d_z 2 --output_path ./lookup_tables_old_2

# python create_lookup_table.py 94 --d_r 2 --d_z 2 --output_path ./lookup_tables_old_2

# python create_lookup_table.py 80 --d_r 2 --d_z 2 --output_path ./lookup_tables_old_2

# python create_lookup_table.py 60 --d_r 2 --d_z 2 --output_path ./lookup_tables_old_2

# python create_lookup_table.py 40 --d_r 2 --d_z 2 --output_path ./lookup_tables_old_2

# # -97,-96,-95,-94,-80,-60,-40,


#!/bin/bash

# Create the new lookup table folder
mkdir -p Final_lookups

# Exact z-values for vertex channels
z_values=(96.215 95.174 94.183 93.155 59.131 39.357 92.177 82.95 81.94 94.66 95.66)

# Loop over each z-value and create lookup table
for z in "${z_values[@]}"
do
    python create_lookup_table.py $z --d_r 2 --d_z 2 --output_path ./Final_lookups
done