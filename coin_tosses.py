import numpy as np
import matplotlib.pyplot as plt

# init data
data = np.random.binomial(1, 0.25, (100000, 1000))

epsilon = [0.5, 0.25, 0.1, 0.01, 0.001]
colors = ['b', 'y', 'g', 'k', 'r']
n_rows = data.shape[0]
n_cols = data.shape[1]
xsis = np.arange(1, n_cols + 1)

# 5.2.a
plt.figure()
for i in range(5):
    cumsum = np.cumsum(data[i, :])
    norm_cumsum = 1.0 * cumsum / xsis
    label_seq = 'seq ' + str(i)
    plt.plot(xsis, norm_cumsum, colors[i], label=label_seq)

    plt.title('p estimator')
    plt.xlabel('m')
    plt.ylabel('p estimator')
plt.legend(loc='upper right')
plt.show()

# 5.2.(b+c)
for eps in epsilon:
    chebyshev = 1.0 / (4 * xsis * (eps ** 2))
    hofding = 2 * np.exp(-2 * xsis * (eps ** 2))

    plt.plot(xsis, chebyshev, 'b', label='Chebyshev')
    plt.plot(xsis, hofding, 'g', label='Hoffding')

    plt.title('epsilon = {0}'.format(eps))
    plt.xlabel('m')
    plt.ylabel('upper bound')

    cumsum = np.cumsum(data, 1)
    norm_cumsum = 1.0 * cumsum / xsis
    norm_cumsum_minus_expectation = norm_cumsum - 0.25
    greater_than_eps = np.abs(norm_cumsum_minus_expectation) >= eps
    greater_than_eps = greater_than_eps.astype(int)
    perc = 1.0 * np.sum(greater_than_eps, 0) / n_rows

    plt.plot(xsis, perc, 'r', label='P[x_m-E[x]>eps')

    plt.ylim([0, 1])  # if I wish to see only the error graph more clearly
    if eps == 0.001:
        plt.legend(loc='lower right')
    else:
        plt.legend(loc='upper right')
    plt.show()
