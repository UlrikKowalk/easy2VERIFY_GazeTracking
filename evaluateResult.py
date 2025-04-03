import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('easyTest.csv')
list_targets = np.array(df['Target'])
list_predictions = np.array(df['Prediction'])

list_predictions = np.convolve(list_predictions, np.ones(3)/3, mode='same')

combine = np.zeros(shape=(len(list_predictions),2))
combine[:, 0] = list_targets
combine[:, 1] = list_predictions
combine = np.sort(combine, axis=0)


plt.plot(list_predictions)
plt.plot(list_targets)
plt.plot(list_targets)
# plt.plot(combine[:, 0], combine[:, 1])
# plt.plot([-0.3, 0.3],[-0.3, 0.3], 'r--')
plt.show()