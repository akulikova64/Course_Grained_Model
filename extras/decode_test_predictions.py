import numpy as np
import sys
import csv

# decodes the predictions during testing

# loading data
wt_array = np.load("../testing_2/centers_test.npy", allow_pickle = True)
pred_array = np.load("../output/26.npy", allow_pickle = True)

# dictionary that decodes the 20 amino acids
aa_dict = {0:'H', 1:'E', 2:'D', 3:'R', 4:'K', 5:'S', 6:'T', 7:'N', 8:'Q', 9:'A', 10:'V', 11:'L', 12:'I', 13:'M', 14:'F', 15:'Y', 16:'W', 17:'P', 18:'G', 19:'C'}

# decode answers (wt) list
answers = []
for wt_aa in wt_array:
    answers.append(aa_dict[wt_aa])

# make and decode predictions list 
predictions = []
for pred in pred_array:
    max_prob = 0
    max_prob_i = 0
    for i in range(0, 20):
        if pred[i] > max_prob:
            max_prob = pred[i]
            max_prob_i = i
    predictions.append(aa_dict[max_prob_i])

# saving data in a CSV file
model_id = "26"
path = "../output/predictions_model_" + model_id + ".csv"

with open(path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['wt_aa', 'pred_aa', 'H', 'E', 'D', 'R', 'K', 'S', 'T', 'N', 'Q', 'A', 'V', 'L', 'I', 'M', 'F', 'Y', 'W', 'P', 'G', 'C'])
    
    for i in range(0, len(predictions)):
        row = []
        row.append(answers[i])
        row.append(predictions[i])
        for pred in pred_array[i]:
            row.append(pred)
        writer.writerow(row)

print("Finished making csv.")



