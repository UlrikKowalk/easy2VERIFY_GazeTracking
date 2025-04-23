import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

from apply_regression import apply_regression

df = pd.read_csv('easyTest.csv')
list_targets = np.array(df['Target'])
list_predictions = np.array(df['Prediction'])

p = np.polyfit(list_predictions, list_targets, 5, rcond=None, full=False, w=None, cov=False)
print(p)

list_eq = p[0]*list_predictions**5 + p[1]*list_predictions**4 + p[2]*list_predictions**3 + p[3]*list_predictions**2 + p[4]*list_predictions + p[5]
# list_eq = apply_regression(list_predictions)

# list_predictions = np.convolve(list_predictions, [0.25, 0.5, 0.25], mode='same') #np.ones(3)/3
list_filtered = []


kalman = KalmanFilter (dim_x=2, dim_z=1)
kalman.F = np.array([[1.,1.],[0.,1.]])
kalman.H = np.array([[1.,0.]])
kalman.P *= 10
kalman.R = 1
kalman.Q = Q_discrete_white_noise(dim=2, dt=1/30, var=1)
kalman.x = np.array([0, 0.])

for item in list_predictions:
    kalman.predict()
    kalman.update(item)
    list_filtered.append(kalman.x[0])

list_eq_filtered = []
kalman = KalmanFilter (dim_x=2, dim_z=1)
kalman.F = np.array([[1.,1.],[0.,1.]])
kalman.H = np.array([[1.,0.]])
kalman.P *= 10
kalman.R = 1
kalman.Q = Q_discrete_white_noise(dim=2, dt=1/30, var=1)
kalman.x = np.array([0, 0.])

for item in list_eq:
    kalman.predict()
    kalman.update(item)
    list_eq_filtered.append(kalman.x[0])

combine = np.zeros(shape=(len(list_predictions), 3))
combine[:, 0] = list_targets
combine[:, 1] = list_predictions
combine[:, 2] = list_eq
combine = np.sort(combine, axis=0)


print('\nError:', 1000 * np.std(np.array(list_predictions) - np.array(list_targets)))
print('Kalman filtered:', 1000 * np.std(np.array(list_filtered) - np.array(list_targets)))
print('EQ:', 1000 * np.std(np.array(list_eq) - np.array(list_targets)))
print('EQ, Kalman filtered:', 1000 * np.std(np.array(list_eq_filtered) - np.array(list_targets)))


plt.plot(combine[:, 0], combine[:, 1], label='prediction')
plt.plot(combine[:, 0], combine[:, 2], label='prediction, EQ')
plt.legend()
plt.plot([-0.25, 0.25],[-0.25, 0.25], 'r--')

# plt.plot(list_predictions, ':')
# plt.plot(list_filtered)
# plt.plot(list_targets)
# plt.plot(list_eq)

plt.show()

