import csv
import numpy as np
import matplotlib.pyplot as plt


with open(losses_file.csv, 'r') as losses_f:
    data_iter = csv.reader(losses_f, delimiter=', ')
    losses = [data for data in data_iter]


plt.style.use(['science'])
plt.plot(losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Loss after given epoch')
plt.show()
