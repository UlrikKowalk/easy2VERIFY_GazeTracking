import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

df = pd.read_csv('easyTest.csv')
list_targets = np.array(df['Target'])
list_predictions = np.array(df['Prediction'])

# list_predictions = np.convolve(list_predictions, [0.25, 0.5, 0.25], mode='same') #np.ones(3)/3
list_filtered = []

kalman = KalmanFilter (dim_x=2, dim_z=1)
kalman.F = np.array([[1.,1.],[0.,1.]])
kalman.H = np.array([[1.,0.]])
kalman.P *= 10
kalman.R = 50
kalman.Q = Q_discrete_white_noise(dim=2, dt=1/30, var=0.5)
kalman.x = np.array([0, 0.])

for item in list_predictions:
    kalman.predict()
    kalman.update(item)
    list_filtered.append(kalman.x[0])


combine = np.zeros(shape=(len(list_predictions), 2))
combine[:, 0] = list_targets
combine[:, 1] = list_predictions
combine = np.sort(combine, axis=0)


plt.plot(list_predictions, ':')
plt.plot(list_filtered)
plt.plot(list_targets)

# plt.plot(combine[:, 0], combine[:, 1])
# plt.plot([-0.3, 0.3],[-0.3, 0.3], 'r--')
plt.show()