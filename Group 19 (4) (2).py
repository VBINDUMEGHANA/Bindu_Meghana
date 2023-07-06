import numpy as np

# Define the data for the BRT and LRT systems
# Each row represents a criterion, and each column represents a performance measure
# The order of the criteria is: travel time, passenger capacity, safety, cost, and environmental impact
BRT_data = np.array([[20, 1500, 5, 10000000, 2000],
                     [25, 1200, 7, 15000000, 2500],
                     [18, 1800, 4, 9000000, 1800],
                     [30, 2000, 3, 12000000, 2200],
                     [210, 1400, 5, 11000000, 2300]])

LRT_data = np.array([[15, 1200, 3, 12000000, 1800],
                     [20, 1000, 4, 18000000, 2200],
                     [10, 1500, 2, 10000000, 1700],
                     [25, 1800, 5, 15000000, 2100],
                     [180, 1000, 2, 13000000, 2400]])

# Normalize the data for each criterion and performance measure
max_BRT = np.max(BRT_data, axis=1)
min_BRT = np.min(BRT_data, axis=1)
max_LRT = np.max(LRT_data, axis=1)
min_LRT = np.min(LRT_data, axis=1)

BRT_norm = (BRT_data - min_BRT[:, np.newaxis]) / (max_BRT[:, np.newaxis] - min_BRT[:, np.newaxis])
LRT_norm = (LRT_data - min_LRT[:, np.newaxis]) / (max_LRT[:, np.newaxis] - min_LRT[:, np.newaxis])

# Define the weights for each criterion (all weights are assumed to be equal)
weights = np.array([1/5, 1/5, 1/5, 1/5, 1/5])

# Calculate the Grey Relational Coefficient (GRC) for each criterion and performance measure
delta_BRT = np.abs(BRT_norm[:, np.newaxis, :] - LRT_norm)
GRC_BRT = np.min(delta_BRT, axis=2)
delta_LRT = np.abs(LRT_norm[:, np.newaxis, :] - BRT_norm)
GRC_LRT = np.min(delta_LRT, axis=2)

# Calculate the Grey Relational Grade (GRG) for each alternative
GRG_BRT = np.sum(weights * GRC_BRT, axis=1)
GRG_LRT = np.sum(weights * GRC_LRT, axis=1)

# Rank the alternatives based on their GRG values
if np.mean(GRG_BRT) > np.mean(GRG_LRT):
    print("BRT is the better alternative.")
else:
    print("LRT is the better alternative.")

