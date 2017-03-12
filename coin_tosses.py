import numpy as np
import matplotlib.pyplot as plt

data = np.random.binomial(1, 0.25, (100000, 1000))

epsilon = [0.5, 0.25, 0.1, 0.01, 0.001]
# epsilon1 = [0.5]

colors = ['b', 'y', 'g', 'k', 'r']

n_rows = data.shape[0]
n_cols = data.shape[1]

# todo check if the following lines needs to be done for each eps since it not related to epse and will result the same all the time
plt.figure()
for i in range(5):
    avg = np.sum(data[i, :])/n_cols
    plt.plot([avg, avg], colors[i])
plt.show()

delta = np.linspace(0.001, 0.999, 100)

for eps in epsilon:
    eps_square = np.square(eps)
    plt.figure()
    plt.title('epsilon = {0}'.format(eps))
    plt.xlabel('delta')
    plt.ylabel('m')
    m1 = (1/(4*eps_square))*(1/delta)
    m2 = (1/(2*eps_square))*(np.log(2/delta))
    plt.plot(delta, m1, 'b')
    plt.plot(delta, m2, 'g')

    # count = 0
    # f_eps = 0.25
    # for i in range(n_rows):
    #     avg = np.sum(data[i, :]) / n_cols
    #     if np.abs(avg - f_eps) >= eps:
    #         count += 1
    # perc = count/n_rows
    # print(perc)
    # plt.plot(perc, 'r')

    plt.show()




