# Expectation-Maximization

Bayesian network for missing data

Implementation of the EM algorithm for learning the parameters for five data sets with missing
rates being 10%, 30%, 50%, 70%, and 100% respectively are provided.
- Estimates the missing data using the current complete model (E-step)
- Learns a new set of parameters using the data set “completed” with the missing
data just estimated (M-step).


# To run

User needs to enter in the the starting point for the prompted parameters for each dataset. 
Each dataset will then be plotted for comparison of Iteration to likelihood.
Then the following stats will be printed:  P_G, P_W_G, P_H_G, total_likelihood

