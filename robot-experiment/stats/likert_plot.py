import numpy as np
import matplotlib.pyplot as plt

IRL = np.asarray([
    [3.5, 2.5, 3, 1.5, 3],
    [4.5, 4.5, 3, 3, 3.5],
    [2.5, 3, 2.5, 1, 2.5],
    [1., 1., 1., 1, 1.],
    [1.5, 1.5, 1., 1., 1],
    [3., 2., 5., 1., 2],
    [3.5, 3, 4, 1.5, 3],
    [2, 2.5, 2, 2, 2.5],
    [2, 2.5, 5, 1.5, 2],
    [1.5, 4, 3.5, 1, 3]
])

NOISE = np.asarray([
    [5, 4.5, 2.5, 2, 4],
    [4, 2.5, 4, 2, 3.5],
    [6, 2.5, 6, 4, 6.5],
    [5.5, 5, 5.5, 4.5, 5.5],
    [4.5, 3, 1.5, 5, 5],
    [3, 3, 3, 1.5, 2],
    [2.5, 1, 1.5, 1, 1],
    [6, 4.5, 5.5, 3.5, 5.5],
    [4.5, 4.5, 4.5, 2.5, 6],
    [4, 4.5, 4.5, 1.5, 4]
])


OURS = np.asarray([
    [7, 7, 7, 6, 7],
    [7, 5, 3, 4.5, 7],
    [7, 4.5, 7, 3.5, 7],
    [7, 6.5, 7, 7, 7],
    [6, 4.5, 4.5, 4.5, 5.5],
    [7, 7, 7, 6, 4],
    [6, 5, 6.5, 4, 6.5],
    [6, 5, 5.5, 6.5, 6],
    [6, 6.5, 6, 5.5, 6.5],
    [6.5, 4, 6.5, 6, 7]
])

IRL_mean = np.mean(IRL,axis=0)
IRL_sem = np.std(IRL,axis=0) / np.sqrt(10)
NOISE_mean = np.mean(NOISE,axis=0)
NOISE_sem = np.std(NOISE,axis=0) / np.sqrt(10)
OURS_mean = np.mean(OURS,axis=0)
OURS_sem = np.std(OURS,axis=0) / np.sqrt(10)


criteria = ['learned', 'intuitive', 'extrapolate', 'deploy', 'prefer']
plt.bar(np.arange(len(criteria))-0.2, IRL_mean, width=0.175, yerr=IRL_sem, color='#b9b9b9', label='IRL')
plt.bar(np.arange(len(criteria))+0.0, NOISE_mean, width=0.175, yerr=NOISE_sem, color='#595959', label='NOISE')
plt.bar(np.arange(len(criteria))+0.2, OURS_mean, width=0.175, yerr=OURS_sem, color='orange', label='OURS')


plt.ylim([0, 7.0])
plt.ylabel("Rating")
plt.xticks(np.arange(len(criteria)), criteria)
plt.legend()
plt.yticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
plt.show()
