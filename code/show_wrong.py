from shutil import copyfile
import os, shutil


folder = '../datasets/wrong_assignments'
for f in os.listdir(folder):
    fp = os.path.join(folder, f)
    os.unlink(fp)
    
with open('../datasets/wrong_predictions.csv') as was:
    for line in was:
        line = line[:-1]
        if (line[-1] == "1"):
            copyfile("../datasets/img_pos/"+line[:-2], "../datasets/wrong_assignments/"+line[:-2])
        elif (line[-1] == "0"):
            copyfile("../datasets/img_neg/"+line[:-2], "../datasets/wrong_assignments/"+line[:-2])
        