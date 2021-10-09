import numpy as np
import matplotlib.pyplot as plt

results = list(np.load("temp.npy"))
new = []
decay = 5
for i in range(decay, len(results)):
    result = 0
    for j in range(decay):
        result += results[i - j]/decay
    new.append(result)


plt.plot(new)
plt.show()