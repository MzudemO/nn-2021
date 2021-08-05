import numpy as np
import itertools

"""
    Run on the 02. of July 2021, using SplitMNIST GWR with config:
    config = {
        "a_threshold": [0.5, 0.5],
        "beta": 0.7,
        "learning_rates": [0.2, 0.001],
        "context": True,
        "num_context": 2,
        "train_replay": True,
        "e_labels": [10, 10],
        "s_labels": [10],
    }

    Gridsearch run on a_threshold values
"""

forgettings = [
    0.26319678087933945,
    0.09535245639368758,
    0.28876113670283,
    0.24443989629854113,
    0.3228252172820466,
    0.11737636982527538,
    0.11709542627111182,
    0.07809788303958931,
    0.25133339371847035,
    0.1922920593458638,
    0.10813461162463003,
    0.03027384388829167,
    0.17626009604784965,
    0.0852426184766219,
    0.16551667185779684,
    0.0885085955712176,
    0.22710494192383457,
    0.10975836425803744,
    0.27524275910213813,
    0.02855894858118992,
    0.08604811683739391,
    0.07531305964969842,
    0.09485346751486207,
    0.17868570738874065,
    0.31889132833020206,
]

accuracies = [
    29.09,
    8.89,
    53.16,
    54.03,
    45.16,
    8.12,
    31.669999999999998,
    49.559999999999995,
    34.06,
    36.35,
    9.84,
    32.58,
    39.96,
    36.309999999999995,
    32.269999999999996,
    20.16,
    23.630000000000003,
    37.53,
    33.19,
    50.71,
    18.19,
    26.5,
    43.29,
    30.42,
    32.910000000000004,
]

forgettings = np.array(forgettings)
accuracies = np.array(accuracies)

grid = itertools.product(np.arange(0.1, 1.0, 0.2), np.arange(0.1, 1.0, 0.2))
grid = np.array(list(grid))

print(
    f"Best forgetting: {np.min(forgettings)} (Run {np.argmin(forgettings)} with parameters {grid[np.argmin(forgettings)]})"
)
print(
    f"Best accuracy: {np.max(accuracies)} (Run {np.argmax(accuracies)} with parameters {grid[np.argmax(accuracies)]})"
)


print("Top three accuracies (index, values, params)")
max_3_accs = (-accuracies).argsort()[:3]
print(max_3_accs)
print(accuracies[max_3_accs])
print(grid[max_3_accs])

print("Top three forgetting (index, values, params)")
min_3_forgettings = (forgettings).argsort()[:3]
print(min_3_forgettings)
print(forgettings[min_3_forgettings])
print(grid[min_3_forgettings])

