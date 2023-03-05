import pandas as pd
from copy import deepcopy as dc
import itertools
from itertools import combinations
from itertools import permutations
import numpy as np
from scipy.optimize import linprog
import math
import pickle
from graphviz import Digraph
import time
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
from mpl_toolkits import mplot3d
import matplotlib.colors as mcolors
from scipy.stats import linregress
from tensorflow.keras.utils import to_categorical

## SIMULATIONS

def get_signature(x):
    return_string = ''
    for ele in list_entite:
        return_string = return_string + str(int(x[ele]))
    return return_string


def simulation(ini_d_s, ini_c_s, nb_ite):
    """
    initial_discrete_state = '100'
    initial_domain = [0.1,0.2,0.3]
    """
    nb_dim = len(ini_c_s)
    return_series = np.zeros((1, nb_dim))
    times = np.zeros((1, 1))
    count = 0
    for dim_i in range(nb_dim):
        return_series[0][dim_i] = int(ini_d_s[dim_i]) + ini_c_s[dim_i]
    times[0][0] = 0
    d_s = ini_d_s
    c_s = ini_c_s.copy()
    for num_ite in range(nb_ite):
        # instant transition
        instant_transition = True
        current_d_s = dc(d_s)
        c_count = 0
        while instant_transition:
            instant_transition = False
            for dim_i in range(nb_dim):
                if c_s[dim_i] == 1 and classify_boundary(d_s, dim_i, 1) == 'output':
                    instant_transition = True
                    c_s[dim_i] = 0
                    d_s = d_s[0:dim_i] + str(int(d_s[dim_i]) + 1) + d_s[dim_i + 1:]
                    break
                elif c_s[dim_i] == 0 and classify_boundary(d_s, dim_i, -1) == 'output':
                    instant_transition = True
                    c_s[dim_i] = 1
                    d_s = d_s[0:dim_i] + str(int(d_s[dim_i]) - 1) + d_s[dim_i + 1:]
                    break
            c_count = c_count + 1
            #             print(d_s)
            if c_count > 1 and current_d_s == d_s:
                return (return_series, times)
        delta_t = math.inf
        first_touch_dim = nb_dim
        first_touch_sign = 0
        one_return_series = np.zeros((1, nb_dim))
        one_times = np.zeros((1, 1))
        set_att = []
        for dim_i in range(nb_dim):
            if c_s[dim_i] == 1 and classify_boundary(d_s, dim_i, 1) == 'attractif':
                set_att = set_att + [dim_i]
            elif c_s[dim_i] == 0 and classify_boundary(d_s, dim_i, -1) == 'attractif':
                set_att = set_att + [dim_i]
        if len(set_att) == nb_dim:
            return (return_series, times)
        for dim_i in range(nb_dim):
            if dim_i not in set_att:
                speed = get_celerity(d_s, dim_i)
                if speed > 0:
                    delta = (1 - c_s[dim_i]) / speed
                    sign = 1
                elif speed < 0:
                    delta = (0 - c_s[dim_i]) / speed
                    sign = 0
                elif speed == 0:
                    delta = math.inf
                if delta < delta_t:
                    delta_t = delta
                    first_touch_dim = dim_i
                    first_touch_sign = sign
        # check if two boundary are reached at the same time
        for dim_i in range(nb_dim):
            if dim_i not in set_att:
                speed = get_celerity(d_s, dim_i)
                if speed > 0:
                    delta = (1 - c_s[dim_i]) / speed
                    sign = 1
                elif speed < 0:
                    delta = (0 - c_s[dim_i]) / speed
                    sign = 0
                elif speed == 0:
                    delta = math.inf
                if delta == delta_t and dim_i != first_touch_dim:
                    print('reach two new boundaries at the same time!')

        c_s[first_touch_dim] = first_touch_sign
        count = count + delta_t
        for dim_i in range(nb_dim):
            if dim_i not in set_att and dim_i != first_touch_dim:
                speed = get_celerity(d_s, dim_i)
                c_s[dim_i] = c_s[dim_i] + delta_t * speed
            one_return_series[0][dim_i] = int(d_s[dim_i]) + c_s[dim_i]
        #         print('discrete state:',d_s)
        #         print('continuous state:',c_s)
        #         print('***********************')
        one_times[0][0] = count
        return_series = np.vstack([return_series, one_return_series])
        times = np.vstack([times, one_times])
    return (return_series, times)


def classify_boundary(state, dim, domain):
    """
    state: '010'
    dim: 2
    domain: 1/0/-1
    """
    current_row = celerities.query('signature == @state')
    speed = current_row['c_' + list_entite[dim]].values[0]
    if domain == 0:
        return 'interior'
    elif domain == 1:
        if speed > 0:
            if int(state[dim]) == max_level[list_entite[dim]]:
                return 'attractif'
            else:
                upper_discrete_state = state
                upper_discrete_state = list(upper_discrete_state)
                upper_discrete_state[dim] = str(int(upper_discrete_state[dim]) + 1)
                upper_discrete_state = "".join(upper_discrete_state)
                upper_speed = celerities.query('signature == @upper_discrete_state')['c_' + list_entite[dim]].values[0]
                if upper_speed <= 0:
                    return 'attractif'
                elif upper_speed > 0:
                    return 'output'
        elif speed == 0:
            return 'neutral'
        elif speed < 0:
            return 'input'
    elif domain == -1:
        if speed < 0:
            if int(state[dim]) == 0:
                return 'attractif'
            else:
                lower_discrete_state = state
                lower_discrete_state = list(lower_discrete_state)
                lower_discrete_state[dim] = str(int(lower_discrete_state[dim]) - 1)
                lower_discrete_state = "".join(lower_discrete_state)
                lower_speed = celerities.query('signature == @lower_discrete_state')['c_' + list_entite[dim]].values[0]
                if lower_speed >= 0:
                    return 'attractif'
                elif lower_speed < 0:
                    return 'output'
        elif speed == 0:
            return 'neutral'
        elif speed > 0:
            return 'input'


def get_celerity(state, dim):
    current_row = celerities.query('signature == @state')
    speed = current_row['c_' + list_entite[dim]].values[0]
    return speed


def get_transition_matrix(state, first_domain, second_domain):
    """
    state: '0101'
    first_domain(second_domain): [0,-1,1,0]
    """
    # get first reach dimension
    first_reach_dim = len(second_domain) + 1
    for num_dim in range(len(second_domain)):
        if second_domain[num_dim] in [-1, 1] and first_domain[num_dim] != second_domain[num_dim]:
            first_reach_dim = num_dim
            break
    # Calculate delta t
    speed = get_celerity(state, first_reach_dim)
    if first_domain[first_reach_dim] in [1, -1]:
        if second_domain[first_reach_dim] == 1:
            delta_cons = 1 / speed
            delta_var = 0
        elif second_domain[first_reach_dim] == -1:
            delta_cons = (-1) / speed
            delta_var = 0
    else:
        if second_domain[first_reach_dim] == 1:
            delta_cons = 1 / speed
            delta_var = (-1) / speed
        elif second_domain[first_reach_dim] == -1:
            delta_cons = 0
            delta_var = (-1) / speed
    # calculate transition matrix
    nb_dim = len(second_domain)
    # the supplementary dimension is 1
    t_m = np.zeros((nb_dim + 1, nb_dim + 1))
    t_m[nb_dim][nb_dim] = 1
    delta1 = time.time()
    for num_dim in range(nb_dim):
        if num_dim == first_reach_dim:
            # for first reached dimension
            if second_domain[first_reach_dim] == 1:
                t_m[first_reach_dim][nb_dim] = 1
            elif second_domain[first_reach_dim] == -1:
                t_m[first_reach_dim][nb_dim] = 0
        else:
            # slide
            if second_domain[num_dim] in [-1, 1] and second_domain[num_dim] == first_domain[num_dim]:
                t_m[num_dim][num_dim] = 1
            else:
                # normal tranform x_new = x + c * delta_t
                current_speed = get_celerity(state, num_dim)
                t_m[num_dim][num_dim] = 1
                t_m[num_dim][first_reach_dim] = current_speed * delta_var
                t_m[num_dim][nb_dim] = current_speed * delta_cons
    return t_m


def get_transition_cross_state(state1, state2):
    """
    state1: '0101'
    state2: '0100'
    """
    nb_dim = len(state1)
    # initialize transiton matrix
    t_m = np.zeros((nb_dim + 1, nb_dim + 1))
    t_m[nb_dim][nb_dim] = 1
    for dim_i in range(nb_dim):
        t_m[dim_i][dim_i] = 1
        if state2[dim_i] > state1[dim_i]:
            t_m[dim_i][nb_dim] = -1
        elif state2[dim_i] < state1[dim_i]:
            t_m[dim_i][nb_dim] = 1
    return t_m


def get_fixed_point(A):
    nb_dim = A.shape[0]
    iden = np.zeros((nb_dim, nb_dim))
    for i_dim in range(nb_dim):
        iden[i_dim][i_dim] = 1
    X = np.subtract(A, iden)
    b = np.zeros(nb_dim)
    fixed_point = np.linalg.solve(X, b)
    return fixed_point


def get_constraint1(discrete_domain):
    dim_free = []
    for one_dim in range(len(discrete_domain)):
        if discrete_domain[one_dim] not in [-1, 1]:
            dim_free = dim_free + [one_dim]
    last_constraint_a = np.zeros((2 * len(dim_free), len(discrete_domain)))
    last_constraint_b = np.zeros((2 * len(dim_free), 1))
    for one_dim in range(len(dim_free)):
        last_constraint_a[one_dim * 2][dim_free[one_dim]] = 1
        last_constraint_a[one_dim * 2 + 1][dim_free[one_dim]] = -1
        last_constraint_b[one_dim * 2][0] = 1
        last_constraint_b[one_dim * 2 + 1][0] = 0
    return last_constraint_a, last_constraint_b


def get_stable_zone(one_trajectory):
    len_state = len(one_trajectory)
    nb_dim = len(list_entite)
    t_m = np.zeros((nb_dim + 1, nb_dim + 1))
    cons_a = np.zeros((1, nb_dim))
    cons_b = np.ones((1, 1))
    for dim_i in range(nb_dim + 1):
        t_m[dim_i][dim_i] = 1
    for i_state in range(len_state):
        current_discrete_state = list(one_trajectory[i_state].keys())[0]
        if i_state < len_state - 1:
            post_discrete_state = list(one_trajectory[i_state + 1].keys())[0]
        current_discrete_trajectory = one_trajectory[i_state][current_discrete_state]
        len_domain = len(current_discrete_trajectory)
        for i_domain in range(len_domain):
            # calculate current constraints
            current_cons_a, current_cons_b = get_constraint1(current_discrete_trajectory[i_domain])
            new_cons_a = np.matmul(current_cons_a, t_m[:-1, :-1])
            temp_b = np.matmul(current_cons_a, t_m[:-1, -1])
            new_cons_b = current_cons_b - temp_b.reshape(temp_b.shape[0], 1)
            cons_a = np.vstack([cons_a, new_cons_a])
            cons_b = np.vstack([cons_b, new_cons_b])
            # calculate next transition matrix
            if i_domain < len_domain - 1:
                current_transition_matrix = get_transition_matrix(current_discrete_state,
                                                                  current_discrete_trajectory[i_domain],
                                                                  current_discrete_trajectory[i_domain + 1])
                t_m = np.matmul(current_transition_matrix, t_m)
            elif i_domain == len_domain - 1 and i_state < len_state - 1:
                current_transition_matrix = get_transition_cross_state(current_discrete_state, post_discrete_state)
                t_m = np.matmul(current_transition_matrix, t_m)
    # regularise the constrains in the first discrete domain
    first_discrete_state = list(one_trajectory[0].keys())[0]
    first_domain = one_trajectory[0][first_discrete_state][0]
    for num_dim in range(len(first_domain)):
        if first_domain[num_dim] == -1:
            cons_a[:, num_dim] = 0
        elif first_domain[num_dim] == 1:
            cons_b = cons_b - cons_a[:, num_dim].reshape(cons_a[:, num_dim].shape[0], 1)
            cons_a[:, num_dim] = 0
    c = np.zeros((cons_a.shape[1],))
    res = linprog(c, cons_a, cons_b)
    if res.success == False:
        return False, [cons_a, cons_b]
    else:
        return True, [cons_a, cons_b]


def dfs(graph, trace, start):
    global cycles
    global reached
    trace = dc(trace)
    if start in trace:
        index = trace.index(start)
        tmp = [i for i in trace[index:]]
        temp_bool = False
        for ele in cycles:
            if same(ele, tmp) == True:
                temp_bool = True
        if temp_bool == False:
            cycles = cycles + [tmp]
        return
    if start not in reached:
        reached = reached + [start]
    trace.append(start)
    for i in graph[start]:
        dfs(graph, trace, i)


def same(list1, list2):
    if len(list1) != len(list2):
        return False
    elif list1[0] not in list2:
        return False
    else:
        length = len(list1)
        first = list2.index(list1[0])
        for i in range(length):
            if list1[i] != list2[(first + i) % length]:
                return False
        return True

def get_next_state(cycle, state):
    len_cycle = len(cycle)
    i_c = cycle.index(state)
    if i_c < len_cycle - 1:
        return cycle[i_c + 1]
    elif i_c == len_cycle - 1:
        return cycle[0]


def dfs_discrete_all(domain, s_domain, state, cycle, t_m, s_z):
    """
    domain: [1,-1,1]
    s_domain: current discrete trajectory without domain [{'01':[[-1,1],[0,-1]]},{}]
    state: discrete state of domain
    cycle: the cycle of discrete state
    t_m: transition matrix without considering domain
    s_z: stable zone without considering domain
    """
    global list_discrete_trajectory
    global list_transition_matrix
    global list_stable_zone
    global list_cycle
    s_domain = dc(s_domain)
    t_m = dc(t_m)
    s_z = dc(s_z)
    cycle = dc(cycle)
    domain = dc(domain)
    state = dc(state)
    #     print('s_domain',s_domain)
    #     print('domain',domain)
    #     print('state',state)
    #     print('cycle',cycle)
    #     print('t_m',t_m)
    #     print('s_z',s_z)

    if len(cycle) > 1 and state == cycle[0]:
        list_discrete_trajectory = list_discrete_trajectory + [s_domain]
        list_transition_matrix = list_transition_matrix + [t_m]
        list_stable_zone = list_stable_zone + [s_z]
        list_cycle = list_cycle + [cycle]
        return
    elif len(cycle) > 1 and state in cycle[:-1] and state != cycle[0]:
        return
    # step1: add domain to s_domain
    if len(s_domain) == 0:
        new_dict = dict()
        new_dict[state] = [dc(domain)]
        s_domain = [new_dict]
        s_z_a, s_z_b = get_constraint1(domain)
        s_z = [s_z_a, s_z_b]
        cycle = cycle + [dc(state)]

        # instant transition
        instant = False
        for i_dim in range(len(domain)):
            if domain[i_dim] == 1 and classify_boundary(state, i_dim, 1) == 'output':
                next_domain = dc(domain)
                next_domain[i_dim] = -1
                next_state = dc(state)
                next_state = list(next_state)
                next_state[i_dim] = str(int(next_state[i_dim]) + 1)
                next_state = "".join(next_state)
                dfs_discrete_all(next_domain, s_domain, next_state, cycle, t_m, s_z)
                instant = True
                break
            elif domain[i_dim] == -1 and classify_boundary(state, i_dim, -1) == 'output':
                next_domain = dc(domain)
                next_domain[i_dim] = 1
                next_state = dc(state)
                next_state = list(next_state)
                next_state[i_dim] = str(int(next_state[i_dim]) - 1)
                next_state = "".join(next_state)
                dfs_discrete_all(next_domain, s_domain, next_state, cycle, t_m, s_z)
                instant = True
                break

        # non instant transition
        if instant == False:
            for i_dim in range(len(domain)):
                if domain[i_dim] == 1 and classify_boundary(state, i_dim, 1) == 'input':
                    domain[i_dim] = 0
                elif domain[i_dim] == -1 and classify_boundary(state, i_dim, -1) == 'input':
                    domain[i_dim] = 0
            for i_dim in range(len(domain)):
                if domain[i_dim] != 1 and classify_boundary(state, i_dim, 1) == 'output':
                    new_domain = dc(domain)
                    new_domain[i_dim] = 1
                    dfs_discrete_all(new_domain, s_domain, state, cycle, t_m, s_z)
                elif domain[i_dim] != -1 and classify_boundary(state, i_dim, -1) == 'output':
                    new_domain = dc(domain)
                    new_domain[i_dim] = -1
                    dfs_discrete_all(new_domain, s_domain, state, cycle, t_m, s_z)
                elif domain[i_dim] != 1 and classify_boundary(state, i_dim, 1) == 'attractif':
                    new_domain = dc(domain)
                    new_domain[i_dim] = 1
                    dfs_discrete_all(new_domain, s_domain, state, cycle, t_m, s_z)
                elif domain[i_dim] != -1 and classify_boundary(state, i_dim, -1) == 'attractif':
                    new_domain = dc(domain)
                    new_domain[i_dim] = -1
                    dfs_discrete_all(new_domain, s_domain, state, cycle, t_m, s_z)
    else:
        # calculate new transition matrix
        current_state = list(s_domain[-1].keys())[0]
        if current_state == state:
            last_domain = s_domain[-1][current_state][-1]
            temp_t_m = get_transition_matrix(state, last_domain, domain)
        elif current_state != state:
            temp_t_m = get_transition_cross_state(current_state, state)
        #         print('temp_t_m',temp_t_m)
        new_t_m = np.matmul(temp_t_m, t_m)
        # calculate new constraint
        c_a, c_b = get_constraint1(domain)
        new_a = np.matmul(c_a, new_t_m[:-1, :-1])
        temp_b = np.matmul(c_a, new_t_m[:-1, -1])
        new_b = c_b - temp_b.reshape(temp_b.shape[0], 1)
        # regulate the constraint considering the first domain
        first_state = cycle[0]
        first_domain = s_domain[0][first_state][0]
        for num_dim in range(len(first_domain)):
            if first_domain[num_dim] == -1:
                new_a[:, num_dim] = 0
            elif first_domain[num_dim] == 1:
                new_b = new_b - new_a[:, num_dim].reshape(new_a[:, num_dim].shape[0], 1)
                new_a[:, num_dim] = 0
        #         print('new_a',new_a)
        #         print('new_b',new_b)
        #         print('************')
        new_s_z_a = np.vstack([s_z[0], new_a])
        new_s_z_b = np.vstack([s_z[1], new_b])
        c = np.zeros((new_s_z_a.shape[1],))
        try:
            res = linprog(c, new_s_z_a, new_s_z_b)
            add_bool = res.success
        except:
            add_bool = False
        if add_bool == False:
            # path end without returning a discrete trajectory
            return
        else:
            # add domain to s_domain
            if current_state == state:
                s_domain[-1][state] = s_domain[-1][state] + [dc(domain)]
            elif current_state != state:
                new_dict = dict()
                new_dict[state] = [dc(domain)]
                s_domain = s_domain + [new_dict]
                cycle = cycle + [dc(state)]
            # choose all possible next domain

            # instant transition
            instant = False
            for i_dim in range(len(domain)):
                if domain[i_dim] == 1 and classify_boundary(state, i_dim, 1) == 'output':
                    next_domain = dc(domain)
                    next_domain[i_dim] = -1
                    next_state = dc(state)
                    next_state = list(next_state)
                    next_state[i_dim] = str(int(next_state[i_dim]) + 1)
                    next_state = "".join(next_state)
                    dfs_discrete_all(next_domain, s_domain, next_state, cycle, new_t_m, [new_s_z_a, new_s_z_b])
                    instant = True
                    break
                elif domain[i_dim] == -1 and classify_boundary(state, i_dim, -1) == 'output':
                    next_domain = dc(domain)
                    next_domain[i_dim] = 1
                    next_state = dc(state)
                    next_state = list(next_state)
                    next_state[i_dim] = str(int(next_state[i_dim]) - 1)
                    next_state = "".join(next_state)
                    dfs_discrete_all(next_domain, s_domain, next_state, cycle, new_t_m, [new_s_z_a, new_s_z_b])
                    instant = True
                    break

            # non instant transition
            if instant == False:
                for i_dim in range(len(domain)):
                    if domain[i_dim] == 1 and classify_boundary(state, i_dim, 1) == 'input':
                        domain[i_dim] = 0
                    elif domain[i_dim] == -1 and classify_boundary(state, i_dim, -1) == 'input':
                        domain[i_dim] = 0
                for i_dim in range(len(domain)):
                    if domain[i_dim] != 1 and classify_boundary(state, i_dim, 1) == 'output':
                        new_domain = dc(domain)
                        new_domain[i_dim] = 1
                        dfs_discrete_all(new_domain, s_domain, state, cycle, new_t_m, [new_s_z_a, new_s_z_b])
                    elif domain[i_dim] != -1 and classify_boundary(state, i_dim, -1) == 'output':
                        new_domain = dc(domain)
                        new_domain[i_dim] = -1
                        dfs_discrete_all(new_domain, s_domain, state, cycle, new_t_m, [new_s_z_a, new_s_z_b])
                    elif domain[i_dim] != 1 and classify_boundary(state, i_dim, 1) == 'attractif':
                        new_domain = dc(domain)
                        new_domain[i_dim] = 1
                        dfs_discrete_all(new_domain, s_domain, state, cycle, new_t_m, [new_s_z_a, new_s_z_b])
                    elif domain[i_dim] != -1 and classify_boundary(state, i_dim, -1) == 'attractif':
                        new_domain = dc(domain)
                        new_domain[i_dim] = -1
                        dfs_discrete_all(new_domain, s_domain, state, cycle, new_t_m, [new_s_z_a, new_s_z_b])
    return

#One simulation

list_entite = ['a','b','c']
max_level = dict()
max_level['a'] = 1
max_level['b'] = 1
max_level['c'] = 1

list_colums = []
list_colums = list_colums + list_entite
for one_ele in list_entite:
    list_colums = list_colums + ['c_'+one_ele]
celerities = pd.DataFrame(columns=list_colums)


sca = np.random.random_sample(1)[0]
scmax = 1
sab = np.random.random_sample(1)[0]
samax = 1
sbc = np.random.random_sample(1)[0]
sbmax = 1

vac0a0=10*np.random.random_sample(1)[0]
vac0a1=10*np.random.random_sample(1)[0]
vac1a0=10*np.random.random_sample(1)[0]
vac1a1=10*np.random.random_sample(1)[0]
vba0b0=10*np.random.random_sample(1)[0]
vba0b1=10*np.random.random_sample(1)[0]
vba1b0=10*np.random.random_sample(1)[0]
vba1b1=10*np.random.random_sample(1)[0]
vcb0c0=10*np.random.random_sample(1)[0]
vcb0c1=10*np.random.random_sample(1)[0]
vcb1c0=10*np.random.random_sample(1)[0]
vcb1c1=10*np.random.random_sample(1)[0]

num = 20
initial_state  = (0.8,0.6,0.1)
nb = 100

noise1 = 0.02
noise2 = 0.02
noise3 = 0.02

cac0a0=vac0a0/(sab - 0)
cac0a1=vac0a1/(samax - sab)
cac1a0=vac1a0/(sab - 0)
cac1a1=vac1a1/(samax - sab)
cba0b0=vba0b0/(sbc - 0)
cba0b1=vba0b1/(sbmax - sbc)
cba1b0=vba1b0/(sbc - 0)
cba1b1=vba1b1/(sbmax - sbc)
ccb0c0=vcb0c0/(sca - 0)
ccb0c1=vcb0c1/(scmax -sca)
ccb1c0=vcb1c0/(sca - 0)
ccb1c1=vcb1c1/(scmax -sca)

df1 = pd.DataFrame([[0,0,0,cac0a0,cba0b0,ccb0c0]],columns=list_colums)
df2 = pd.DataFrame([[0,0,1,-cac1a0,cba0b0,ccb0c1]],columns=list_colums)
df3 = pd.DataFrame([[0,1,0,cac0a0,cba0b1,-ccb1c0]],columns=list_colums)
df4 = pd.DataFrame([[0,1,1,-cac1a0,cba0b1,-ccb1c1]],columns=list_colums)
df5 = pd.DataFrame([[1,0,0,cac0a1,-cba1b0,ccb0c0]],columns=list_colums)
df6 = pd.DataFrame([[1,0,1,-cac1a1,-cba1b0,ccb0c1]],columns=list_colums)
df7 = pd.DataFrame([[1,1,0,cac0a1,-cba1b1,-ccb1c0]],columns=list_colums)
df8 = pd.DataFrame([[1,1,1,-cac1a1,-cba1b1,-ccb1c1]],columns=list_colums)

celerities=pd.concat([df1,df2,df3,df4,df5,df6,df7,df8])
celerities['signature'] = celerities.apply(get_signature,axis=1)

ini_discrete = ''
ini_fractional = []
if initial_state[0]>=sab:
    ini_discrete = ini_discrete+'1'
    ini_fractional = ini_fractional + [(initial_state[0]-sab)/(samax - sab)]
elif initial_state[0]<sab:
    ini_discrete = ini_discrete+'0'
    ini_fractional = ini_fractional + [initial_state[0]/sab]

if initial_state[1]>=sbc:
    ini_discrete = ini_discrete+'1'
    ini_fractional = ini_fractional + [(initial_state[1]-sbc)/(sbmax - sbc)]
elif initial_state[1]<sbc:
    ini_discrete = ini_discrete+'0'
    ini_fractional = ini_fractional + [initial_state[1]/sbc]

if initial_state[2]>=sca:
    ini_discrete = ini_discrete+'1'
    ini_fractional = ini_fractional + [(initial_state[2]-sca)/(scmax - sca)]
elif initial_state[2]<sca:
    ini_discrete = ini_discrete+'0'
    ini_fractional = ini_fractional + [initial_state[2]/sca]

data,t = simulation(ini_discrete,ini_fractional,num)
real_data = dc(data)
for i in range(data.shape[0]):
    if data[i][0] < 1:
        real_data[i][0] = data[i][0]*sab
    elif data[i][0] >= 1:
        real_data[i][0] = (data[i][0] - 1)*(samax - sab) + sab
    if data[i][1] < 1:
        real_data[i][1] = data[i][1]*sbc
    elif data[i][1] >= 1:
        real_data[i][1] = (data[i][1] - 1)*(sbmax - sbc) + sbc
    if data[i][2] < 1:
        real_data[i][2] = data[i][2]*sca
    elif data[i][2] >= 1:
        real_data[i][2] = (data[i][2] - 1)*(scmax - sca) + sca
noise_data = np.zeros((nb+1,3))
delta_t = t[-1][0]/nb
new_t = np.zeros((nb+1,1))
for i in range(nb+1):
    new_t[i][0] = i*delta_t
    for j in range(t.shape[0]-1):
        if t[j][0] <= new_t[i][0] and t[j+1][0] >= new_t[i][0]:
            noise_data[i][0] = random.gauss(0,noise1) + real_data[j][0] + (real_data[j+1][0] - real_data[j][0])*(new_t[i][0] - t[j][0])/(t[j+1][0] - t[j][0])
            noise_data[i][1] = random.gauss(0,noise2) + real_data[j][1] + (real_data[j+1][1] - real_data[j][1])*(new_t[i][0] - t[j][0])/(t[j+1][0] - t[j][0])
            noise_data[i][2] = random.gauss(0,noise3) + real_data[j][2] + (real_data[j+1][2] - real_data[j][2])*(new_t[i][0] - t[j][0])/(t[j+1][0] - t[j][0])
            break

##DATASET

def create_dataset_sc(nb_data,num,nb, noise = 0.0):

    X,Y=[],[]
    initial_state  = (0.5,0.7,0.8)

    noise1 = noise
    noise2 = noise
    noise3 = noise

    for k in range(nb_data):
        list_entite = ['a','b','c']
        max_level = dict()
        max_level['a'] = 1
        max_level['b'] = 1
        max_level['c'] = 1

        list_colums = []
        list_colums = list_colums + list_entite
        for one_ele in list_entite:
            list_colums = list_colums + ['c_'+one_ele]
        celerities = pd.DataFrame(columns=list_colums)

        sca = np.random.random_sample(1)[0]
        scmax = 1
        sab = np.random.random_sample(1)[0]
        samax = 1
        sbc = np.random.random_sample(1)[0]
        sbmax = 1

        vac0a0=10*np.random.random_sample(1)[0]
        vac0a1=10*np.random.random_sample(1)[0]
        vac1a0=10*np.random.random_sample(1)[0]
        vac1a1=10*np.random.random_sample(1)[0]
        vba0b0=10*np.random.random_sample(1)[0]
        vba0b1=10*np.random.random_sample(1)[0]
        vba1b0=10*np.random.random_sample(1)[0]
        vba1b1=10*np.random.random_sample(1)[0]
        vcb0c0=10*np.random.random_sample(1)[0]
        vcb0c1=10*np.random.random_sample(1)[0]
        vcb1c0=10*np.random.random_sample(1)[0]
        vcb1c1=10*np.random.random_sample(1)[0]

        Y.append([sca,sab,sbc])

        cac0a0=vac0a0/(sab - 0)
        cac0a1=vac0a1/(samax - sab)
        cac1a0=vac1a0/(sab - 0)
        cac1a1=vac1a1/(samax - sab)
        cba0b0=vba0b0/(sbc - 0)
        cba0b1=vba0b1/(sbmax - sbc)
        cba1b0=vba1b0/(sbc - 0)
        cba1b1=vba1b1/(sbmax - sbc)
        ccb0c0=vcb0c0/(sca - 0)
        ccb0c1=vcb0c1/(scmax -sca)
        ccb1c0=vcb1c0/(sca - 0)
        ccb1c1=vcb1c1/(scmax -sca)

        df1 = pd.DataFrame([[0,0,0,cac0a0,cba0b0,ccb0c0]],columns=list_colums)
        df2 = pd.DataFrame([[0,0,1,-cac1a0,cba0b0,ccb0c1]],columns=list_colums)
        df3 = pd.DataFrame([[0,1,0,cac0a0,cba0b1,-ccb1c0]],columns=list_colums)
        df4 = pd.DataFrame([[0,1,1,-cac1a0,cba0b1,-ccb1c1]],columns=list_colums)
        df5 = pd.DataFrame([[1,0,0,cac0a1,-cba1b0,ccb0c0]],columns=list_colums)
        df6 = pd.DataFrame([[1,0,1,-cac1a1,-cba1b0,ccb0c1]],columns=list_colums)
        df7 = pd.DataFrame([[1,1,0,cac0a1,-cba1b1,-ccb1c0]],columns=list_colums)
        df8 = pd.DataFrame([[1,1,1,-cac1a1,-cba1b1,-ccb1c1]],columns=list_colums)

        celerities=pd.concat([df1,df2,df3,df4,df5,df6,df7,df8])
        celerities['signature'] = celerities.apply(get_signature,axis=1)

        ini_discrete = ''
        ini_fractional = []
        if initial_state[0]>=sab:
            ini_discrete = ini_discrete+'1'
            ini_fractional = ini_fractional + [(initial_state[0]-sab)/(samax - sab)]
        elif initial_state[0]<sab:
            ini_discrete = ini_discrete+'0'
            ini_fractional = ini_fractional + [initial_state[0]/sab]

        if initial_state[1]>=sbc:
            ini_discrete = ini_discrete+'1'
            ini_fractional = ini_fractional + [(initial_state[1]-sbc)/(sbmax - sbc)]
        elif initial_state[1]<sbc:
            ini_discrete = ini_discrete+'0'
            ini_fractional = ini_fractional + [initial_state[1]/sbc]

        if initial_state[2]>=sca:
            ini_discrete = ini_discrete+'1'
            ini_fractional = ini_fractional + [(initial_state[2]-sca)/(scmax - sca)]
        elif initial_state[2]<sca:
            ini_discrete = ini_discrete+'0'
            ini_fractional = ini_fractional + [initial_state[2]/sca]

        data,t = simulation(ini_discrete,ini_fractional,num)
        real_data = dc(data)
        for i in range(data.shape[0]):
            if data[i][0] < 1:
                real_data[i][0] = data[i][0]*sab
            elif data[i][0] >= 1:
                real_data[i][0] = (data[i][0] - 1)*(samax - sab) + sab
            if data[i][1] < 1:
                real_data[i][1] = data[i][1]*sbc
            elif data[i][1] >= 1:
                real_data[i][1] = (data[i][1] - 1)*(sbmax - sbc) + sbc
            if data[i][2] < 1:
                real_data[i][2] = data[i][2]*sca
            elif data[i][2] >= 1:
                real_data[i][2] = (data[i][2] - 1)*(scmax - sca) + sca
        noise_data = np.zeros((nb+1,3))
        delta_t = t[-1][0]/nb
        new_t = np.zeros((nb+1,1))
        for i in range(nb+1):
            new_t[i][0] = i*delta_t
            for j in range(t.shape[0]-1):
                if t[j][0] <= new_t[i][0] and t[j+1][0] >= new_t[i][0]:
                    noise_data[i][0] = random.gauss(0,noise1) + real_data[j][0] + (real_data[j+1][0] - real_data[j][0])*(new_t[i][0] - t[j][0])/(t[j+1][0] - t[j][0])
                    noise_data[i][1] = random.gauss(0,noise2) + real_data[j][1] + (real_data[j+1][1] - real_data[j][1])*(new_t[i][0] - t[j][0])/(t[j+1][0] - t[j][0])
                    noise_data[i][2] = random.gauss(0,noise3) + real_data[j][2] + (real_data[j+1][2] - real_data[j][2])*(new_t[i][0] - t[j][0])/(t[j+1][0] - t[j][0])
                    break

        X.append(noise_data)
    return np.array(X),np.array(Y)

##METHODE DERIVEE


def derivee_to_classe(X):
    classe_dict= {'000': [], '001': [], '010': [], '011': [], '100': [], '101': [], '110': [], '111': []}
    for i in range(len(X)-1):

        classe_dict[state].append(X[i])

    dist, sab_derivee = 1,0
    for k in range(4):
        for i in range(len(classe_dict['0'+format(k, '#04b')[-2:]])):
            for j in range(len(classe_dict['1'+format(k, '#04b')[-2:]])):

                a0=classe_dict['0'+format(k, '#04b')[-2:]][i][0]
                a1=classe_dict['1'+format(k, '#04b')[-2:]][j][0]

                if np.abs(a0-a1) < dist and (0<a0<1) and (0<a1<1):
                    dist = np.abs(a0-a1)
                    sab_derivee= (np.abs(a0)+np.abs(a1))/2
                    '''print('dist ', dist)
                    print('sab ',sab_derivee)'''
    return sab_derivee

def derivee_dataset(X,Y):
    L=np.array([np.abs(derivee_to_classe(X[i]) - Y[i][1]) for i in range(len(X))])
    return np.mean(L)

def grid2d(min = 0, max = 1, step = 0.1): #crée une grille avec les valeurs de seuil à parcourir
    x=np.arange(min,max+step,step)
    y=np.arange(min,max+step,step)
    return np.array(np.meshgrid(x,y))

def grid3d(min = 0, max = 1, step = 0.1): #crée une grille avec les valeurs de seuil à parcourir
    x=np.arange(min,max+step,step)
    y=np.arange(min,max+step,step)
    z=np.arange(min,max+step,step)
    return np.array(np.meshgrid(x,y,z))

def seuil_grid_search2d(X, seuils, step=0.1, min=0, max=1, nb_points=1): #fonction qui parcoure la grille pour trouver les seuils avec le meilleur score
    xx,yy = grid2d(min,max,step)
    scores = np.zeros(xx.shape)
    sc = seuils[2]
    for i in range(len(xx)):
        for j in range(len(xx[0])):
            seuils= np.array([xx[i,j],yy[i,j], sc])
            X_classes=sc_to_class1(X,seuils) #état de X à chaque pas de temps
            scores[i,j] = correlation_score3d(X, X_classes, nb_points) #état estimés à partir des dérivées comparés avec les états réels
    s=np.unravel_index(np.argmax(scores), np.shape(xx))
    return scores, xx[s], yy[s]

def seuil_grid_search3d_old(X, step=0.1, min=0, max=1, nb_points=1): #fonction qui parcoure la grille pour trouver les seuils avec le meilleur score
    xx,yy,zz=grid3d(min,max,step)
    scores=np.zeros(xx.shape)
    for i in range(len(xx)):
        for j in range(len(xx[0])):
            for k in range(len(xx[0,0])):
                seuils= np.array([xx[i,j,k],yy[i,j,k],zz[i,j,k]])
                X_classes=sc_to_class1(X,seuils) #état de X à chaque pas de temps
                scores[i,j,k] = correlation_score3d(X, X_classes, nb_points) #état estimés à partir des dérivées comparés avec les états réels
    s=np.unravel_index(np.argmax(scores), np.shape(xx))
    return scores, xx[s], yy[s], zz[s]

def seuil_grid_search3d(X, step=0.1, min=0, max=1, nb_points=1): #fonction qui parcoure la grille pour trouver les seuils avec le meilleur score
    xx,yy,zz=grid3d(min,max,step)
    T = table_derivees(X, nb_points)
    scores=np.zeros(xx.shape)
    for i in range(len(xx)):
        for j in range(len(xx[0])):
            for k in range(len(xx[0,0])):
                seuils= np.array([xx[i,j,k],yy[i,j,k],zz[i,j,k]])
                X_classes=sc_to_class1(X,seuils) #état de X à chaque pas de temps
                scores[i,j,k] = correlation_score3d(X=X, X_classes=X_classes, nb_points=nb_points, table_derivees = T) #état estimés à partir des dérivées comparés avec les états réels
    s=np.unravel_index(np.argmax(scores), np.shape(xx))
    return scores, xx[s], yy[s], zz[s]

def seuil_grid_search_opt3d(X, step=0.1, min=0, max=1, nb_points=1): #parcourt les points de la simulation plutôt qu'une grille afin de réduire le temps de calcul
    n=len(X)
    scores=np.zeros(shape = (n,n,n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                print (i,j,k)
                seuils= np.array([X[i,0],X[j,1],X[k,2]])
                X_classes=sc_to_class1(X,seuils) #état de X à chaque pas de temps
                scores[i,j,k] = correlation_score3d(X, X_classes, nb_points) #état estimés à partir des dérivées comparés avec les états réels
    s=np.unravel_index(np.argmax(scores), (n,n,n))
    return scores, X[s[0],0], X[s[1],1], X[s[2],2]

def table_derivees(X, nb_points = 5):
    n=len(X)
    T=np.zeros(shape=(n-2*nb_points,3, nb_points))
    for i in range(len(T)):
        for j in range(3):
            for p in range(1, nb_points+1):
                T[i,j,p-1] = (X[i+p,j]-X[i-p,j])/(2*p)
    return T

def sc_to_class1(X,seuils): #renvoie le vecteur X1 des états discrets dans lesquels se trouvent les points de X
    X1=[]
    state=''
    for i in range(len(X)):
        state=str(int(X[i][0]>seuils[0]))+str(int(X[i][1]>seuils[1]))+str(int(X[i][2]>seuils[2]))
        X1.append(int(state,2)) #on passe du binaire au décimal
    return to_categorical(X1)

def correlation_score3d(X, X_classes, table_derivees, nb_points=1):
    score = 0
    state_derivative = ['111','011','110','010','101','001','100','000']
    for i in range(len(table_derivees)):
        a = np.argmax(X_classes[i]) #état supposé
        for j in range(3):
            for p in range(nb_points):
                s=table_derivees[i,j,p]
                score+=np.abs(s)*int(int(s>0) == int(state_derivative[a][j]))
    return score/(3*(len(X)-2*nb_points))

def correlation_score3d_old(X, X_classes, nb_points=1):
    score = 0
    for i in range(nb_points,len(X_classes)-nb_points):
        a = np.argmax(X_classes[i]) #état supposé
        for k in range(3):
            score += point_score_value1(X,a,i,k, nb_points) #ajout d'un score traduisant la corrélation état discret/valeur de la dérivée
    return score/(3*(len(X)-2*nb_points))

def point_score(X,a,i,k, nb_points=1): #score traduisant la corrélation état discret/valeur de la dérivée
    state_derivative = ['111','011','110','010','101','001','100','000'] #signe de la dérivée attendu pour chaque état
    regression_range = nb_points
    Y=X[i-regression_range:i+regression_range+1, k]
    linregress_model = linregress(x=np.arange(0,len(Y)/10,0.1), y=Y) #ajustement d'une droite sur les points considérés
    slope, rvalue = linregress_model.slope, linregress_model.rvalue
    while(regression_range>1 and np.abs(rvalue)<0.90):
        regression_range-=1
        Y=X[i-regression_range:i+regression_range+1, k]
        linregress_model = linregress(x=np.arange(0,len(Y)/10,0.1), y=Y) #ajustement d'une droite sur les points considérés
        slope, rvalue = linregress_model.slope, linregress_model.rvalue
    #if regression_range>1:
        #print('yes!')
    #print(Y,'slope',slope, 'rvalue', rvalue, regression_range)
    return np.abs(slope)*regression_range*(int(slope>0)==int(state_derivative[a][k]))

def point_score_sign(X,a,i,k): #score suivant le signe de la dérivée
    state_derivative = ['111','011','110','010','101','001','100','000'] #signe de la dérivée attendu pour chaque état
    return int(int((X[i+1][k]>X[i][k])) == int(state_derivative[a][k])) #comparaison avec l'état suivant

def point_score_sign1(X,a,i,k, nb_points=1):
    state_derivative = ['111','011','110','010','101','001','100','000'] #signe de la dérivée attendu pour chaque état
    x=X[i,k]
    s=np.sum(x-X[i-nb_points:i,k])+np.sum(X[i+1:i+1+nb_points,k]-x)
    return int(int(s>0) == int(state_derivative[a][k]))

def point_score_value(X,a,i,k, nb_points=1):
    state_derivative = ['111','011','110','010','101','001','100','000'] #signe de la dérivée attendu pour chaque état
    s=0
    for j in range(1,nb_points+1): #nb de points à considérer de part et d'autre
        s+= (X[i+j,k]-X[i-j,k])/(2*j)
    return np.abs(s)*int(int(s>0) == int(state_derivative[a][k]))
    # =valeur de la pente si le signe correspond, 0 sinon

def point_score_value1(X,a,i,k, nb_points=1):
    state_derivative = ['111','011','110','010','101','001','100','000'] #signe de la dérivée attendu pour chaque état
    s=0
    for j in range(1,nb_points+1): #nb de points à considérer de part et d'autre
        s_temp = (X[i+j,k]-X[i-j,k])/(2*j)
        s+= np.abs(s_temp)*int(int(s_temp>0) == int(state_derivative[a][k]))
    return s
    # =valeur de la pente si le signe correspond, 0 sinon

def evaluate_grid_search(Ns, step= 0.1, nb_points=1): #Utilise la méthode un grand nombre de fois et évalue son efficacité
    score_tot = 0
    for i in range(Ns):
        print(i)
        G=create_dataset_sc(1,10,50)
        X, seuils= np.array(G[0][0]), np.array(G[1][0])
        scores, sa, sb, sc = seuil_grid_search3d(X, seuils, step=step)
        score_tot = score_tot + np.abs(sa-seuils[0]) + np.abs(sb-seuils[1]) + np.abs(sb-seuils[2])
    return score_tot/(3*Ns)

def score_map(min = 0, max = 1, step = 0.1, nb_points=1):
    G = create_dataset_sc(1,10,50)
    X, seuils = np.array(G[0][0]), np.array(G[1][0])
    grid = grid2d(min, max, step)
    xx, yy = grid[0], grid[1]
    scores, sa, sb = seuil_grid_search2d(X=X, seuils=seuils, step=step, nb_points=nb_points)
    plt.pcolor(xx, yy, scores)
    #plt.imshow(scores)
    plt.colorbar()
    plt.xlabel('sab estimé : ' + str(round(sa,2))+' sab réel ' + str(round(seuils[0],2)))
    plt.ylabel(('sbc estimé : ' + str(round(sb,2))+' sbc réel ' + str(round(seuils[1],2))))
    print(scores)
    print('sa estimé : ', sa, 'sa réel', seuils[0])
    print('sb estimé : ', sb, 'sb réel', seuils[1])
    print('sc réel', seuils[2])
    plt.show()
    #affichage2d1d(X,seuils)

def score_normalisation(scores): #normalisation du tableau des scores entre 0 et 1
    indmax = np.unravel_index(np.argmax(scores), np.shape(scores))
    indmin = np.unravel_index(np.argmin(scores), np.shape(scores))
    return (1/(scores[indmax] - scores[indmin]))*(scores - scores[indmin])

def indice_vrais_seuils(seuils, step=0.1):
    xx, yy, zz = grid3d(step)
    sab_round, sbc_round, sca_round = np.round(seuils[0],1), np.round(seuils[1],1), np.round(seuils[2],1) #A CHANGER SI LE STEP N'EST PAS 0.1
    x, y, z = int(sbc_round//step), int(sab_round//step), int(sca_round//step)
    return x,y,z

def score_quantile(N, step=0.1, nb_points=1, noise = 0.0):
    score_max_distrib = []
    score_min_distrib = []
    score_vrais_seuils_distrib = []
    score_moy_distrib = []
    xx, yy, zz = grid3d(step=step)

    for i in range(N):
        G = create_dataset_sc(1,10,50, noise = noise)
        X, seuils = np.array(G[0][0]), np.array(G[1][0])
        scores, sab, sbc, sca = seuil_grid_search3d(X=X, seuils=seuils, step=step, nb_points=nb_points)

        indmax = np.unravel_index(np.argmax(scores), np.shape(scores)) #coordonnées du score maximum
        score_max_distrib.append(scores[indmax]) #On ajoute la valeur du score maximum

        indmin = np.unravel_index(np.argmin(scores), np.shape(scores)) #coordonnées du score minimum
        score_min_distrib.append(scores[indmin]) #On ajoute la valeur du score minimum

        x,y,z = indice_vrais_seuils(seuils, step)
        score_vrais_seuils_distrib.append(scores[x,y,z]) #On ajoute la valeur du score calculé pour les vrais valeurs des seuils

        score_moy_distrib.append(np.mean(scores)) #score moyen
    return score_max_distrib, score_min_distrib, score_vrais_seuils_distrib, score_moy_distrib

def seuil_grid_search1d(X, step=0.1, min=0, max=1, nb_points=1):
    xx = np.arange(min, max+step, step)
    score_vect=[]
    for i in range(len(xx)):
        score = 0
        sab = xx[i]
        for j in range(nb_points, len(X)-nb_points):
            s=0
            for k in range(1,nb_points+1): #nb de points à considérer de part et d'autre
                s+= (X[j+k,1]-X[j-k,1])/(2*k) #Concentration B pour le seuil A
            #s+= score_saturation(X, sab, j) #on augmente le score si la concentration est à un maximum ou un minimum
            score += np.abs(s)*int(int(s>0) == int(X[i,0]<sab)) #vaut abs(s) si le signe correspond, 0 sinon
        score_vect.append(score)
    return np.array(score_vect)

#def saturation(X, sab, j):

#à changer : augmenter le score si la concentration est proche du min ou du max et que la dérivée est nulle
# et ne pas prendre en compte les points après : n'influent pas sur ce qu'il se passe à l'instant t

G=create_dataset_sc(1,10,50)
X, seuils= np.array(G[0][0]), np.array(G[1][0])
classes=sc_to_class1(X,seuils)

#scipy.stats.linregress