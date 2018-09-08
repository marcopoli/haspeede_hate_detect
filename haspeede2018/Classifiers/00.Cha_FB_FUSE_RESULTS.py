import sys
import csv
import pandas as pd

reader1 = pd.read_csv("final_models/TW/TASK3.2_1_twfb.csv", header=None, delimiter=',', names = ["id", "res"])
reader2 = pd.read_csv("final_models/TW/TASK3.2_2_twfb.csv", header=None, delimiter=',', names = ["id", "res"])



fullRes = reader1.append(reader2)

sortedlist =fullRes.sort_values(axis=0, by='id')


print(sortedlist)

file = open("final_models/TW/TASK3.2_final_twfb.tsv","w")

for index, row  in sortedlist.iterrows():
    file.write(str(row['id']) +"	"+str(row['res'])+"\n")

file.close()