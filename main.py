import math
import random
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt


def fitness(c, w, agent):
    f = 0
    owerstack = 0
    mas = 0
    for j in range(len(c)):
        if agent[j] > 0:
            f = f + c[j]
            mas = mas + w[j]
    if mas > w_all:
        owerstack = abs(w_all - mas)
    f = f - shtraf * owerstack
    return (f, owerstack)


def generate(mu):
    xi = []
    for j in range(mu):
        xi.append([])
        for i in range(N):
            if random.random() < exp_pers_items:
                xi[j].append(1)
            else:
                xi[j].append(0)  
    return xi



def select(DET, X):
    w = np.zeros(len(X), dtype=int)
    for i in range(mu):
        w[i] = round(DET[i]*100)
    population = np.arange(0, len(X), 1)
    par = random.choices(population, w, k=lam)
    parents = []
    for k in range(lam):
        parents.append(X[par[k]])
    return  (parents)


def crossover(parents): #uniform
    if random.random() <= P_cross:
        return (parents[0], parents[1])
    child_1 = []
    child_2 = []
    iterrations = 0
    if len(parents[0]) > len(parents[1]):
        iterrations = len(parents[1])
    else:
        iterrations = len(parents[0])
    for i in range(iterrations):
        child_1.append(parents[1][i] if random.randint(0,1) == 1 else parents[0][i])
        child_2.append(parents[0][i] if random.randint(0,1) == 1 else parents[1][i])
    return (child_1, child_2)


def mutation(children, P_mut, mut_level):
    Y = []
    for j in range(len(children)):
        d = children[j]
        if random.random() < P_mut:
            for i in range(mut_level):
                num_mut = random.randint(0, len(d) - 1)
                d[num_mut] = 1 - d[num_mut]
        Y.append(d)
    return Y


def selection(X):
    P_det = np.zeros(len(X), dtype=float)
    f = np.zeros(len(X), dtype=int); W_fit = P_det
    for i in range (mu):
        f[i] = X[i][N]
    for i in range (mu):
        f[i] = f[i] + 1 - min(f)
    avg_f = sum(f) / len(f)
    for i in range (mu):
        W_fit[i] = f[i] / avg_f
    s_w_fit = sum(W_fit)
    for i in range (mu):
        P_det[i] = W_fit[i] / s_w_fit
    return P_det


def new_population (Z, Y):
    W = []
    for z in range(mu-2):
        W.append(Z[z][0:N])
    for z in range(lam):
        W.append(Y[z])
    return W
    


data = []
with open("knapPI_5.txt") as f:
    for line in f:
        data.append([int(x) for x in line.split()])
size_k = data[0][0] + 1
w_all = data[0][1]
G_opt = data[0][2]
c = []
w = []
for i in range(size_k):
    if i > 0:
        c.append(data[i][0])
        w.append(data[i][1])
exp_pers_items = 0.05  # w_all / sum(w)
mu = 100  # int(input('Number of parents in each age is '))
lam = 2  # int(input('Number of children in each age is '))
mut_level_0 = 5  # float(input('Mutation level is '))
P_mut_min = 0.4
P_mut_max = 0.95  # float(input('Mutation probability is '))
P_cross = 0.6  # float(input('Crossover probability is '))
best = []
tmax = int(input('Number of evolution age '))
shtraf = 3
N = len(c)
X = generate(mu)
for t in range(tmax):
    mut_level = math.ceil(mut_level_0 * (tmax - t + 1) / tmax)
    P_mut = P_mut_min + (P_mut_max - P_mut_min) * t / tmax
    F = []
    for i in range(mu):
        agent = X[i]
        (fit, ow) = fitness(c, w, agent)
        F.append(fit)
    Z = []
    for i in range(mu):
        Zi = X[i]
        Zi.append(F[i])
        Z.append(Zi)
    Z.sort(key = itemgetter(N), reverse=True)
    best.append(Z[0])
    X = Z[:][0:N]
    DET = selection(X)
    parents = select(DET, X)
    children = crossover(parents)
    Y = mutation(children, P_mut, mut_level)
    X = new_population(Z, Y)
q = len(best)
x = np.arange(0, q, 1)
y = []
for i in range(q):
    k = len(best[i])
    y.append(best[i][k - 1])
print(best[q - 1])
er = (G_opt - best[q - 1][size_k - 1]) / G_opt * 100
print('Error from general optimum is', er, ' percents')
plt.style.use('classic')
plt.plot(x, y)
plt.show()
