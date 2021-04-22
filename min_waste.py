import numpy as np
import matplotlib.pyplot as plt


class MinimizeWaste:
    def __init__(self, w, demand):
        self.w = w
        self.demand = demand
        self.w_opt= []
        self.minimal_waste = []
        self.idx_minimal_waste = []
        self.n_orders = []

    def get_opt_w(self, n_orders, verbose=False):
        # allocate one width less, since last width must always be set
        self.n_orders = n_orders
        M = self.n_orders - 1
        N = len(self.w) - 1
        c = [[[np.infty] for i in range(M + 1)] for j in range(N + 1)]
        idx = [[[[]] for i in range(M + 1)] for j in range(N + 1)]
        self.minimal_waste = np.infty
        self.idx_minimal_waste = []
        for m in range(M, -1, -1):
            n_start = N - M + m
            for n in range(n_start, m-1, -1):
                if max(n, m) > 0:
                    if m == 0:
                        # if no widths remain, compute waste based on  last width set (=n), select optimal path and stop
                        idx_min = np.where(np.array(c[n + 1][m + 1]) == min(c[n + 1][m + 1]))[0][0]
                        idx[n][m] = [idx[n + 1][m + 1].copy()[idx_min] + [n + 1]]
                        c[n][m] = [self.compute_waste(1, idx[n][m][0])]
                        if c[n][m][0] < self.minimal_waste:
                            self.minimal_waste = c[n][m][0]
                            self.idx_minimal_waste = idx[n][m].copy()[0]
                            if verbose:
                                print("minimal waste opt: ", self.minimal_waste)
                    elif m == n:
                        # if remaining widths to be set equal remaining widths, set all, select optimal path and stop
                        c[n][m] = [self.compute_waste(n + 1, k) for k in idx[n + 1][m]]
                        idx_min = np.where(np.array(c[n][m]) == min(c[n][m]))[0][0]
                        c[n][m] = [c[n][m][idx_min]]
                        idx[n][m] = [idx[n+1][m].copy()[idx_min]]
                        if c[n][m][0] < self.minimal_waste:
                            self.minimal_waste = c[n][m][0]
                            self.idx_minimal_waste = idx[n][m].copy()[0]
                            if verbose:
                                print("minimal waste opt: ", self.minimal_waste)
                    else:
                        if n == n_start:
                            c[n][m] = [0]
                            idx[n][m] = [[n_fix for n_fix in range(N + 1, n, -1)]]
                            # if so far all widths have been set (starting at N+1), set index accordingly
                        else:
                            idx[n][m] = idx[n + 1][m].copy()
                            if m == M:
                                # if no widths have been set yet, compute waste according to current width (reduce by 1)
                                c[n][m] = [self.compute_waste(n + 1, idx[n + 1][m][0])]
                            else:
                                # if at least one width has been set and at least one width has not been set AND
                                # not at a stopping point:
                                # track all paths for n+1 if width has not been set (optimum cannot be set here yet)
                                # and also add the (optimum computable) optimal path when previous width has been set
                                c[n][m] = [self.compute_waste(n + 1, k) for k in idx[n + 1][m]]
                                idx_min = np.where(np.array(c[n + 1][m + 1]) == min(c[n + 1][m + 1]))[0][0]
                                c[n][m].append(c[n + 1][m + 1][idx_min])
                                idx[n][m].append(idx[n + 1][m + 1][idx_min].copy() + [n + 1])
        self.w_opt = np.sort(np.array([self.w[idx-1] for idx in self.idx_minimal_waste]))

    # compute waste given the current index and fixed larger indices (= set widths)
    def compute_waste(self, idx_current, idx_fixed):
        waste = 0
        idx_fixed = np.sort(idx_fixed)
        for idx in range(idx_current, 1 + idx_fixed[-1]):
            next_idx = min(idx_fixed[idx <= idx_fixed])
            waste += (self.w[next_idx-1]-self.w[idx-1])*self.demand[idx-1]
        return waste


def waste_calculation(demand, w, chosen_w):
    waste = 0
    opt_w = np.array(chosen_w)
    for idx, w_i in enumerate(w):
        current_w = min(opt_w[opt_w >= w_i])
        cut_w = current_w - w_i
        waste += cut_w * demand[idx]
    return waste


def plot_orders(w, demand, w_order, waste_value, method):
    # solution
    plt.figure()
    for i in range(len(w_order)):
        if i == 0:
            cum_demand_i = sum(demand[w <= w_order[0]])
        else:
            w_left = w[w <= w_order[i]]
            w_between = w_left[w_left > w_order[i-1]]
            cum_demand_i = sum(np.array([demand[np.where(w_between_j == w)] for w_between_j in w_between]))[0]
        x_pair = [w_order[i], w_order[i]]
        y_pair = [0, cum_demand_i]
        plt.plot(x_pair, y_pair, linewidth=5, color="orange")
    # demand
    for i in range(len(w)):
        x_pair = [w[i], w[i]]
        y_pair = [0, demand[i]]
        plt.plot(x_pair, y_pair, linewidth=2.5, color="b")
    plt.xlabel("width")
    plt.ylabel("length (client demand / order)")
    plt.title("Ordered widths (orange) for given demand (blue) using " + method + ". Waste = " + str(waste_value), fontsize=10)




