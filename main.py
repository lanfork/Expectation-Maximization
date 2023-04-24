import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Algorithm for learning parameters for a simple Bayesian network from missing data.
# Estimate the missing data using the current complete model (E-step),
# Learn a new set of parameters using the data set “completed” with the missing
# data just estimated (M-step).

class BayesianNetwork:
    def __init__(self):
        self.graph = {'Gender': ['Weight', 'Height']}
        self.variables = ['Gender', 'Weight', 'Height']
        self.values = {'Gender': [0, 1], 'Weight': [0, 1], 'Height': [0, 1]}
        self.parents = {'Gender': [], 'Weight': ['Gender'], 'Height': ['Gender']}
        self.CPT = {'Gender': np.array([0.5, 0.5]),
                    'Weight': np.array([0.5, 0.5, 0.5, 0.5]),
                    'Height': np.array([0.5, 0.5, 0.5, 0.5])}


    def fit(self, data, max_iters=100, threshold=0.001):
        # Prompt user to choose starting parameters
        print("Please choose a starting point for the following parameters:")
        P_G_M = float(input("P(gender=M)= "))
        P_W_G_M = float(input("P(weight=greater_than_130|gender=M)= "))
        P_W_G_F = float(input("P(weight=greater_than_130|gender=F)= "))
        P_H_G_M = float(input("P(height= greater_than_55|gender=M)= "))
        P_H_G_F = float(input("P(height= greater_than_55|gender=F)= "))

        # Initialize with the chosen starting parameters
        P_G = np.array([P_G_M, 10 - P_G_M])
        P_W_G = np.array([[1 - P_W_G_M, P_W_G_M], [1 - P_W_G_F, P_W_G_F]])
        P_H_G = np.array([[1 - P_H_G_M, P_H_G_M], [1 - P_H_G_F, P_H_G_F]])

        # Convert the input data to a pandas dataframe
        data_copy = pd.DataFrame(data, columns=['Gender', 'Weight', 'Height'])

        # Initialize the list to store the log-likelihood for each iteration
        likelihoods = []

        # Run EM algorithm until convergence
        prev_likelihood = None
        for i in range(max_iters):
            # E-step
            expected_counts = {'Gender': np.zeros_like(P_G),
                               'Weight': np.zeros_like(P_W_G),
                               'Height': np.zeros_like(P_H_G)}
            total_likelihood = 0

            # Compute the posterior probabilities for each data point using the current parameters
            for _, row in data_copy.iterrows():
                gender = row['Gender']
                weight = row['Weight']
                height = row['Height']
                P_G_W_H = P_G * P_W_G[weight, :] * P_H_G[height, :]
                P_G_W_H /= P_G_W_H.sum()

                # Increment the expected counts using the posterior probabilities
                expected_counts['Gender'] += P_G_W_H
                expected_counts['Weight'][weight, :] += P_G_W_H
                expected_counts['Height'][height, :] += P_G_W_H

                # Compute the log-likelihood of the data using the current parameters
                gender = int(gender)
                total_likelihood += np.log(P_G_W_H[gender])

            # Add the log-likelihood to the list
            likelihoods.append(total_likelihood)
            # M-step
            P_G = expected_counts['Gender'] / len(data_copy)
            P_W_G = expected_counts['Weight'] / expected_counts['Gender'][np.newaxis, :]
            P_H_G = expected_counts['Height'] / expected_counts['Gender'][np.newaxis, :]

            # Check for convergence
            if prev_likelihood is not None and abs(total_likelihood - prev_likelihood) < threshold:
                break
            prev_likelihood = total_likelihood

        # Plot the likelihood vs number of iterations
        plt.plot(range(1, len(likelihoods) + 1), likelihoods)
        plt.xlabel('Iteration')
        plt.ylabel('Log-likelihood')
        plt.title('EM algorithm convergence')
        plt.show()

        # Return the learned parameters
        return P_G, P_W_G, P_H_G, total_likelihood


def read_files_and_fit():
    # Read in the txt files and concatenate the data
    dfs = []
    for i in range(1, 6):
        print(f"Data {i}")
        df = pd.read_csv(f"data{i}.txt", delimiter="\t", header=None, skiprows=1)
        df.columns = ['Gender', 'Weight', 'Height']
        dfs.append(df)
        data = pd.concat(dfs, ignore_index=True)

        # Replace missing values with a random 1 or 0
        data = data.applymap(lambda x: np.random.randint(2) if x == '-' else x)

        # Convert the data to a list of tuples
        data = [tuple(x) for x in data.to_numpy()]

        # Create a Bayesian network and fit it to the data
        bn = BayesianNetwork()
        P_G, P_W_G, P_H_G, total_likelihood = bn.fit(data)

    # Print the learned parameters and total log-likelihood of the data
    print(f"P(G) = {P_G}")
    print(f"P(W|G) = {P_W_G}")
    print(f"P(H|G) = {P_H_G}")
    print(f"Total log-likelihood: {total_likelihood}")


read_files_and_fit()