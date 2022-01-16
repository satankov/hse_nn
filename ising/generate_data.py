# Uses python3

import numpy as np
import pandas as pd
import sys
import cy


# ====================== all functions ======================

def coord(site):
    """get coordinate i of vector"""
    x = site // L
    y = site - x * L
    return (x, y)


def get(i):
    """fixin' boundary"""
    if i < 0:
        return i
    else:
        return i % L


def get_neigh():
    """get neighbour's arr"""
    s = np.arange(L ** 2).reshape(L, L)
    nei = []
    for site in range(L * L):
        i, j = coord(site)
        nei += [s[get(i - 1), get(j)], s[get(i), get(j + 1)], s[get(i + 1), get(j)], s[get(i), get(j - 1)]]
    return np.array(nei, dtype=np.int32).reshape(L * L, 4)


#################################################################

def gen_state():
    """generate random start state with lenght L*L and q components"""
    state = np.array([np.random.choice([-1, 1]) for _ in range(L * L)], dtype=np.int32)
    return state


################################################################################

def model(T, path, N_avg=10, N_mc=10, Relax=10):
    """Моделируем Ising"""

    state = gen_state()
    nei = get_neigh()

    size = state.shape[0]
    cols = ['T'] + [_ for _ in range(size)]
    df_init = pd.DataFrame(data=[], columns=cols, )
    df_init.to_csv(path, index=None, header=True)

    # relax $Relax times be4 AVG
    for __ in range(Relax):
        cy.mc_step(state, nei, T)
    # AVG every $N_mc steps
    for _ in range(N_avg):
        for __ in range(N_mc):
            cy.mc_step(state, nei, T)
        df = pd.DataFrame(data=[state], )
        df.insert(0, 'T', T)
        df.to_csv(path, index=None, header=False, mode='a')


def t_range(tc):
    t1 = np.arange(0.0005, 0.01, 0.0005)  # 19
    t2 = np.arange(0.01, 0.2, 0.005)  # 38
    t3 = np.arange(0.2, 0.5, 0.05)  # 6
    t_ = np.concatenate([t1, t2, t3])
    t_low = np.round(-t_ + tc, 7)  # low
    t_high = np.round(t_ + tc, 7)  # high
    t = np.concatenate((t_low, t_high), axis=None)
    t.sort()
    return t


#################################################################

if __name__ == '__main__':
    global L
    L = 12

    dot = int(sys.argv[1])
    seed = 1 + dot
    np.random.seed(seed)  # np.random.seed(int(sys.argv[1]))

    tc = 1 / (np.log(2 ** 0.5 + 1) / 2)  # 2.269185314213022
    T = t_range(tc)[dot]

    z = 2.15
    Relax = 12 * L ** z
    N_avg = 1500
    N_mc = int(2 * L ** z) if abs(T - tc) <= 0.02 else int(2 * L ** 2)

    title = f"ising_L{L}_relax{Relax}_dot{dot}"
    path = f"samples/samples_{L}/{title}.csv"
    model(T, path, N_avg, N_mc, Relax)