from min_waste import MinimizeWaste
from min_waste import waste_calculation
from min_waste import plot_orders
import numpy as np

# Given is client demand (length) with a given width of a material. The company can order the material at any width
# and can satisfy the client demand by delivering the length and cutting material to obtain the width demanded by
# the client. The company can only order at a limited number of widths (n_orders). The objective is to minimize the
# produced waste under the given constraints. As an example, the client orders at width 198 with a demand of 610
# and the only width ordered by the company is at 200.
# Then for that demand, the waste is: waste(198) = (200 - 198) x 610 = 1220

# The optimal orders are obtained with a dynamic programming - like approach, where the problem starts with the highest
# width in demand (note that this width must always be ordered by the company) and sequentially reduces to the next
# lower width, where at each width a decision can be made if this width is ordered or not. Those decisions are tracked.
# Iff a width is ordered, the optimal path up to that width (starting with the largest width) can be computed
# and the remaining paths are discarded

# The optimization is compared to a simulation approach, which randomly generates 100000 different scenarios,
# where in each scenario the ordered width are selected randomly (including the largest width). The scenario
# that provides the minimal waste is finally selected. Results are visualized.

w = np.array([100., 102., 104., 106., 108., 110., 112., 114., 116., 118., 120.,
              122., 124., 126., 128., 130., 132., 134., 136., 138., 140., 142.,
              144., 146., 148., 150., 152., 154., 156., 158., 160., 162., 164.,
              166., 168., 170., 172., 174., 176., 178., 180., 182., 184., 186.,
              188., 190., 192., 194., 196., 198., 200.])

demand = np.array([206., 117., 715., 615., 667., 521., 946., 593.,  67., 117., 635.,
                   459., 521., 833., 358., 478., 629., 486., 227., 546., 607., 705.,
                   794., 620., 131., 760., 306., 964., 375., 819., 116., 598., 104.,
                   598., 432., 863., 406., 298., 723., 558., 762., 749., 546.,   7.,
                   973., 756.,  15., 128., 173., 610., 574.])

n_orders = 20
n_simul = 100000  # used for simulation approach only

ClassMinimizeWaste = MinimizeWaste(w, demand)
ClassMinimizeWaste.get_opt_w(n_orders, verbose=True)
min_waste_opt = ClassMinimizeWaste.minimal_waste
min_w_opt = ClassMinimizeWaste.w_opt

min_waste_simul = np.infty
for i in range(n_simul):
    idx_simul = list(np.sort(np.random.choice([i for i in range(len(w)-1)], size=n_orders-1, replace=False)))
    idx_simul.append(len(w)-1)
    w_simul = np.array([w[i] for i in idx_simul])
    waste_simul = waste_calculation(demand, w, w_simul)
    if waste_simul < min_waste_simul:
        min_waste_simul = waste_simul
        min_w_simul = w_simul
        print("minimal waste simul: ", min_waste_simul)

plot_orders(w, demand, min_w_simul, min_waste_simul, method="simulation")
plot_orders(w, demand, min_w_opt, min_waste_opt, method="optimization (DP)")



