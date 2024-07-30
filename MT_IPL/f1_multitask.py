import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import bernoulli
import itertools
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import ast

def generate_exp_sample(num_features, bernoulli_params_UZ, bernoulli_params_UX, bernoulli_params_UY, coefficients_MX, coefficients_MY, C):
    UZ_values = np.random.binomial(1, bernoulli_params_UZ, size=num_features)
    UX_value = np.random.binomial(1, bernoulli_params_UX)
    UY_value = np.random.binomial(1, bernoulli_params_UY)
    MX_value = np.dot(UZ_values, coefficients_MX)
    MY_value = np.dot(UZ_values, coefficients_MY)
    X_value = np.random.binomial(1, 0.5)
    Y_value = 1 if (C + MX_value + UY_value > 0 and C + MX_value + UY_value < 1) or \
                   (C + MX_value + UY_value > 1 and C + MX_value + UY_value < 2) else 0
    sample = {'UZ': UZ_values, 'UX': UX_value, 'UY': UY_value, 'X': X_value, 'Y': Y_value}
    return sample

def generate_exp_samples(num_samples, num_features, bernoulli_params_UZ, bernoulli_params_UX, bernoulli_params_UY, coefficients_MX, coefficients_MY, C):
    experimental_samples = []
    for _ in tqdm(range(num_samples), desc="Generating Experimental Samples"):
        sample = generate_exp_sample(num_features, bernoulli_params_UZ, bernoulli_params_UX, bernoulli_params_UY, coefficients_MX, coefficients_MY, C)
        experimental_samples.append(sample)
    return experimental_samples

def generate_obs_samples(num_samples, num_features, bernoulli_params_UZ, bernoulli_params_UX, bernoulli_params_UY, coefficients_MX, coefficients_MY, C):
    observational_samples = []
    for _ in tqdm(range(num_samples), desc="Generating Observational Samples"):
        UZ_values = np.random.binomial(1, bernoulli_params_UZ, size=num_features)
        UX_value = np.random.binomial(1, bernoulli_params_UX)
        UY_value = np.random.binomial(1, bernoulli_params_UY)
        UX_value = np.array([UX_value])
        MX_value = np.dot(UZ_values, coefficients_MX)
        MY_value = np.dot(UZ_values, coefficients_MY)
        X_value = 1 if MX_value + UX_value[0] > 0.5 else 0
        Y_value = 1 if (C + MX_value + UY_value > 0 and C + MX_value + UY_value < 1) or \
                       (C + MX_value + UY_value > 1 and C + MX_value + UY_value < 2) else 0
        sample = {'UZ': UZ_values, 'UX': UX_value, 'UY': UY_value, 'X': X_value, 'Y': Y_value}
        observational_samples.append(sample)
    return observational_samples

def select_over_1300(experimental_samples):
    UZ_samples = [tuple(sample['UZ']) for sample in experimental_samples]
    uz_counter = Counter(UZ_samples)
    selected_samples = []
    for sample in tqdm(experimental_samples, desc="Selecting Samples over 1300"):
        if uz_counter[tuple(sample['UZ'])] >= 1300:
            selected_samples.append(sample)
    return selected_samples

def calculate_Y(X, MY, UY):
    num_observed_features = 15
    C = -0.77953605542
    bernoulli_params_UZ = ([0.524110233482, 0.689566064108, 0.180145428970, 0.317153536644, 0.046268153873,
                            0.340145244411, 0.100912238566, 0.772038172066, 0.913108434869, 0.364272299067,
                            0.063667554704, 0.454839320009, 0.586687215140, 0.018824647595, 0.871017316787])
    bernoulli_params_UX = 0.601680857267
    bernoulli_params_UY = 0.497668975278
    coefficients_MX = np.array([0.843870221861, 0.178759296447, -0.372349746729, -0.950904544846, -0.439457721339,
                                -0.725970103834, -0.791203963585, -0.843183562918, -0.68422616618, -0.782051030131,
                                -0.434420454146, -0.445019418094, 0.751698021555, -0.185984172192, 0.191948271392])
    coefficients_MY = np.array([-0.453251661832, 0.424563325534, 0.0924810605305, 0.312680246141, 0.7676961338,
                                0.124337421843, -0.435341306455, 0.248957751703, -0.161303883519, -0.537653062121,
                                -0.222087991408, 0.190167775134, -0.788147770713, -0.593030174012, -0.308066297974])
    MY_array = np.array(MY, dtype=int)
    MY_value = np.dot(MY_array, coefficients_MY)
    CX_MY_UY = C * X + MY_value + UY
    if (0 < CX_MY_UY < 1) or (1 < CX_MY_UY < 2):
        return 1
    else:
        return 0

def calculate_X(MX, UX):
    num_observed_features = 15
    C = -0.77953605542
    bernoulli_params_UZ = ([0.524110233482, 0.689566064108, 0.180145428970, 0.317153536644, 0.046268153873,
                            0.340145244411, 0.100912238566, 0.772038172066, 0.913108434869, 0.364272299067,
                            0.063667554704, 0.454839320009, 0.586687215140, 0.018824647595, 0.871017316787])
    bernoulli_params_UX = 0.601680857267
    bernoulli_params_UY = 0.497668975278
    coefficients_MX = np.array([0.843870221861, 0.178759296447, -0.372349746729, -0.950904544846, -0.439457721339,
                                -0.725970103834, -0.791203963585, -0.843183562918, -0.68422616618, -0.782051030131,
                                -0.434420454146, -0.445019418094, 0.751698021555, -0.185984172192, 0.191948271392])
    coefficients_MY = np.array([-0.453251661832, 0.424563325534, 0.0924810605305, 0.312680246141, 0.7676961338,
                                0.124337421843, -0.435341306455, 0.248957751703, -0.161303883519, -0.537653062121,
                                -0.222087991408, 0.190167775134, -0.788147770713, -0.593030174012, -0.308066297974])
    MX_array = np.array(MX, dtype=float)
    MX_value = np.dot(MX_array, coefficients_MX)
    if MX_value + UX > 0.5:
        return 1
    else:
        return 0

def calculate_probabilities(subpopulation):
    num_observed_features = 15
    C = -0.77953605542
    bernoulli_params_UZ = ([0.524110233482, 0.689566064108, 0.180145428970, 0.317153536644, 0.046268153873,
                            0.340145244411, 0.100912238566, 0.772038172066, 0.913108434869, 0.364272299067,
                            0.063667554704, 0.454839320009, 0.586687215140, 0.018824647595, 0.871017316787])
    bernoulli_params_UX = 0.601680857267
    bernoulli_params_UY = 0.497668975278
    coefficients_MX = np.array([0.843870221861, 0.178759296447, -0.372349746729, -0.950904544846, -0.439457721339,
                                -0.725970103834, -0.791203963585, -0.843183562918, -0.68422616618, -0.782051030131,
                                -0.434420454146, -0.445019418094, 0.751698021555, -0.185984172192, 0.191948271392])
    coefficients_MY = np.array([-0.453251661832, 0.424563325534, 0.0924810605305, 0.312680246141, 0.7676961338,
                                0.124337421843, -0.435341306455, 0.248957751703, -0.161303883519, -0.537653062121,
                                -0.222087991408, 0.190167775134, -0.788147770713, -0.593030174012, -0.308066297974])
    observed_features = list(subpopulation[:num_observed_features])
    MX_observed = coefficients_MX[:num_observed_features][np.array(list(observed_features), dtype=int)]
    MY_observed = coefficients_MY[:num_observed_features][np.array(list(observed_features), dtype=int)]
    Y = np.zeros(4)
    Y[0] = calculate_Y(0, observed_features, 0)
    Y[1] = calculate_Y(1, observed_features, 0)
    Y[2] = calculate_Y(0, observed_features, 1)
    Y[3] = calculate_Y(1, observed_features, 1)
    P_UY_0 = bernoulli.pmf(0, bernoulli_params_UY)
    P_UY_1 = bernoulli.pmf(1, bernoulli_params_UY)
    P_UX_0 = bernoulli.pmf(0, bernoulli_params_UX)
    P_UX_1 = bernoulli.pmf(1, bernoulli_params_UX)
    P_yx = (P_UY_0 * calculate_Y(1, observed_features, 0) + P_UY_1 * calculate_Y(1, observed_features, 1))
    P_y_x_prime = (P_UY_0 * calculate_Y(0, observed_features, 0) + P_UY_1 * calculate_Y(0, observed_features, 1))
    P_y_prime_x_prime = (P_UY_0 * calculate_Y(1, observed_features, 0) + P_UY_1 * calculate_Y(1, observed_features, 1))
    P_y = P_UY_0 * (1 - calculate_Y(0, observed_features, 0)) + P_UY_1 * (1 - calculate_Y(0, observed_features, 1))
    P_y_and_x = (P_UX_0 * P_UY_0 * calculate_Y(calculate_X(observed_features, 0), observed_features, 0) +
                 P_UX_0 * P_UY_1 * calculate_Y(calculate_X(observed_features, 0), observed_features, 1) +
                 P_UX_1 * P_UY_0 * calculate_Y(calculate_X(observed_features, 1), observed_features, 0) + 
                 P_UX_1 * P_UY_1 * calculate_Y(calculate_X(observed_features, 1), observed_features, 1))
    P_y_and_x_prime = P_UY_0 * (P_UX_0 * (1 - calculate_Y(calculate_X(observed_features, 0), observed_features, 0)) +
                                P_UX_1 * (1 - calculate_Y(calculate_X(observed_features, 1), observed_features, 0)))
    P_y_prime_and_x = P_UY_0 * (P_UX_0 * calculate_Y(calculate_X(observed_features, 0), observed_features, 1) +
                                P_UX_1 * calculate_Y(calculate_X(observed_features, 1), observed_features, 1))
    P_y_prime_and_x_prime = P_UY_0 * (P_UX_0 * (1 - calculate_Y(calculate_X(observed_features, 0), observed_features, 0)) +
                                      P_UX_0 * (1 - calculate_Y(calculate_X(observed_features, 0), observed_features, 1)) +
                                      P_UX_1 * (1 - calculate_Y(calculate_X(observed_features, 1), observed_features, 0)) +
                                      P_UX_1 * (1 - calculate_Y(calculate_X(observed_features, 1), observed_features, 1)))
    return P_yx, P_y_x_prime, P_y_prime_x_prime, P_y, P_y_and_x, P_y_prime_and_x, P_y_and_x_prime, P_y_prime_and_x_prime

def calculate_pns_bounds(P_yx, P_y_x_prime, P_y_prime_x_prime, P_y, P_y_and_x, P_y_prime_and_x, P_y_and_x_prime, P_y_prime_and_x_prime):
    sigma = 1
    W = ((P_yx) - 2 * (P_y_x_prime) - (P_y_prime_and_x_prime))
    L = max(0, P_yx - P_y_x_prime, P_y - P_y_x_prime, P_yx - P_y)
    U = min(P_yx, P_y_prime_x_prime, P_y_and_x + P_y_prime_and_x_prime, P_yx - P_y_x_prime + P_y_and_x_prime + P_y_prime_and_x)
    lower_bound = (W + sigma * U)
    upper_bound = (W + sigma * L)
    return lower_bound, upper_bound

def calc_pns(subpopulation):
    probs = calculate_probabilities(subpopulation)
    lower_bound, upper_bound = calculate_pns_bounds(*probs)
    return (lower_bound + upper_bound) / 2

def calculate_observed_pns_bound(subpopulation):
    bernoulli_params_UZ = [0.524110233482, 0.689566064108, 0.180145428970, 0.317153536644, 0.046268153873,
                           0.340145244411, 0.100912238566, 0.772038172066, 0.913108434869, 0.364272299067,
                           0.063667554704, 0.454839320009, 0.586687215140, 0.018824647595, 0.871017316787,
                           0.164966968157, 0.578925020078, 0.983082980658, 0.018033993991, 0.074629121266]
    P_z16_0 = bernoulli.pmf(0, bernoulli_params_UZ[15])
    P_z17_0 = bernoulli.pmf(0, bernoulli_params_UZ[16])
    P_z18_0 = bernoulli.pmf(0, bernoulli_params_UZ[17])
    P_z19_0 = bernoulli.pmf(0, bernoulli_params_UZ[18])
    P_z20_0 = bernoulli.pmf(0, bernoulli_params_UZ[19])
    P_z16_1 = bernoulli.pmf(1, bernoulli_params_UZ[15])
    P_z17_1 = bernoulli.pmf(1, bernoulli_params_UZ[16])
    P_z18_1 = bernoulli.pmf(1, bernoulli_params_UZ[17])
    P_z19_1 = bernoulli.pmf(1, bernoulli_params_UZ[18])
    P_z20_1 = bernoulli.pmf(1, bernoulli_params_UZ[19])
    pns = (P_z16_0 * P_z17_0 * P_z18_0 * P_z19_0 * P_z20_0 * calc_pns(subpopulation + [0, 0, 0, 0, 0]) +
           P_z16_0 * P_z17_0 * P_z18_0 * P_z19_0 + P_z20_1 * calc_pns(subpopulation + [0, 0, 0, 0, 1]) +
           P_z16_0 * P_z17_0 * P_z18_0 * P_z19_1 + P_z20_0 * calc_pns(subpopulation + [0, 0, 0, 1, 0]) +
           P_z16_0 * P_z17_0 * P_z18_1 * P_z19_0 + P_z20_0 * calc_pns(subpopulation + [0, 0, 1, 0, 0]) +
           P_z16_0 * P_z17_1 * P_z18_0 * P_z19_0 + P_z20_0 * calc_pns(subpopulation + [0, 1, 0, 0, 0]) +
           P_z16_1 * P_z17_0 * P_z18_0 * P_z19_0 + P_z20_0 * calc_pns(subpopulation + [1, 0, 0, 0, 0]) +
           P_z16_1 * P_z17_1 * P_z18_0 * P_z19_0 + P_z20_0 * calc_pns(subpopulation + [1, 1, 0, 0, 0]) +
           P_z16_1 * P_z17_0 * P_z18_1 * P_z19_0 + P_z20_0 * calc_pns(subpopulation + [1, 0, 1, 0, 0]) +
           P_z16_1 * P_z17_0 * P_z18_0 * P_z19_1 + P_z20_0 * calc_pns(subpopulation + [1, 0, 0, 1, 0]) +
           P_z16_1 * P_z17_0 * P_z18_0 * P_z19_0 + P_z20_1 * calc_pns(subpopulation + [1, 0, 0, 0, 1]) +
           P_z16_0 * P_z17_1 * P_z18_1 * P_z19_0 + P_z20_0 * calc_pns(subpopulation + [0, 1, 1, 0, 0]) +
           P_z16_0 * P_z17_1 * P_z18_0 * P_z19_1 + P_z20_0 * calc_pns(subpopulation + [0, 1, 0, 1, 0]) +
           P_z16_0 * P_z17_1 * P_z18_0 * P_z19_0 + P_z20_1 * calc_pns(subpopulation + [0, 1, 0, 0, 1]) +
           P_z16_0 * P_z17_0 * P_z18_1 * P_z19_1 + P_z20_0 * calc_pns(subpopulation + [0, 0, 1, 1, 0]) +
           P_z16_0 * P_z17_0 * P_z18_1 * P_z19_0 + P_z20_1 * calc_pns(subpopulation + [0, 0, 1, 0, 1]) +
           P_z16_0 * P_z17_0 * P_z18_0 * P_z19_1 + P_z20_1 * calc_pns(subpopulation + [0, 0, 0, 1, 1]) +
           P_z16_1 * P_z17_1 * P_z18_1 * P_z19_0 + P_z20_0 * calc_pns(subpopulation + [1, 1, 1, 0, 0]) +
           P_z16_1 * P_z17_1 * P_z18_0 * P_z19_1 + P_z20_0 * calc_pns(subpopulation + [1, 1, 0, 1, 0]) +
           P_z16_1 * P_z17_1 * P_z18_0 * P_z19_0 + P_z20_1 * calc_pns(subpopulation + [1, 1, 0, 0, 1]) +
           P_z16_1 * P_z17_0 * P_z18_1 * P_z19_1 + P_z20_0 * calc_pns(subpopulation + [1, 0, 1, 1, 0]) +
           P_z16_1 * P_z17_0 * P_z18_1 * P_z19_0 + P_z20_1 * calc_pns(subpopulation + [1, 0, 1, 0, 1]) +
           P_z16_1 * P_z17_0 * P_z18_0 * P_z19_1 + P_z20_1 * calc_pns(subpopulation + [1, 0, 0, 1, 1]) +
           P_z16_0 * P_z17_1 * P_z18_1 * P_z19_1 + P_z20_0 * calc_pns(subpopulation + [0, 1, 1, 1, 0]) +
           P_z16_0 * P_z17_1 * P_z18_1 * P_z19_0 + P_z20_1 * calc_pns(subpopulation + [0, 1, 1, 0, 1]) +
           P_z16_0 * P_z17_1 * P_z18_0 * P_z19_1 + P_z20_1 * calc_pns(subpopulation + [0, 1, 0, 1, 1]) +
           P_z16_0 * P_z17_0 * P_z18_1 * P_z19_1 + P_z20_1 * calc_pns(subpopulation + [0, 0, 1, 1, 1]) +
           P_z16_1 * P_z17_1 * P_z18_1 * P_z19_1 + P_z20_0 * calc_pns(subpopulation + [1, 1, 1, 1, 0]) +
           P_z16_1 * P_z17_1 * P_z18_1 * P_z19_0 + P_z20_1 * calc_pns(subpopulation + [1, 1, 1, 0, 1]) +
           P_z16_1 * P_z17_1 * P_z18_0 * P_z19_1 + P_z20_1 * calc_pns(subpopulation + [1, 1, 0, 1, 1]) +
           P_z16_1 * P_z17_0 * P_z18_1 * P_z19_1 + P_z20_1 * calc_pns(subpopulation + [1, 0, 1, 1, 1]) +
           P_z16_0 * P_z17_1 * P_z18_1 * P_z19_1 + P_z20_1 * calc_pns(subpopulation + [0, 1, 1, 1, 1]) +
           P_z16_1 * P_z17_1 * P_z18_1 * P_z19_1 * P_z20_1 * calc_pns(subpopulation + [1, 1, 1, 1, 1]))
    return pns / 32

def main_data_generation():
    num_features = 20
    C = -0.77953605542
    bernoulli_params_UZ = [0.524110233482, 0.689566064108, 0.180145428970, 0.317153536644, 0.046268153873,
                           0.340145244411, 0.100912238566, 0.772038172066, 0.913108434869, 0.364272299067,
                           0.063667554704, 0.454839320009, 0.586687215140, 0.018824647595, 0.871017316787,
                           0.164966968157, 0.578925020078, 0.983082980658, 0.018033993991, 0.074629121266]
    bernoulli_params_UX = 0.29908139311
    bernoulli_params_UY = 0.9226108109253
    coefficients_MX = np.array([0.843870221861, 0.178759296447, -0.372349746729, -0.950904544846, -0.439457721339,
                                -0.725970103834, -0.791203963585, -0.843183562918, -0.68422616618, -0.782051030131,
                                -0.434420454146, -0.445019418094, 0.751698021555, -0.185984172192, 0.191948271392,
                                0.401334543567, 0.331387702568, 0.522595634402, -0.928734581669, 0.203436441511])
    coefficients_MY = np.array([-0.453251661832, 0.424563325534, 0.0924810605305, 0.312680246141, 0.7676961338,
                                0.124337421843, -0.435341306455, 0.248957751703, -0.161303883519, -0.537653062121,
                                -0.222087991408, 0.190167775134, -0.788147770713, -0.593030174012, -0.308066297974,
                                0.218776507777, -0.751253645088, -0.11151455376, 0.785227235182, -0.568046522383])
    num_samples = 5000000
    experimental_samples = generate_exp_samples(num_samples, num_features, bernoulli_params_UZ, bernoulli_params_UX, bernoulli_params_UY, coefficients_MX, coefficients_MY, C)
    observational_samples = generate_obs_samples(num_samples, num_features, bernoulli_params_UZ, bernoulli_params_UX, bernoulli_params_UY, coefficients_MX, coefficients_MY, C)
    selected_samples_ex = select_over_1300(experimental_samples)
    selected_UZ_samples = [sample['UZ'] for sample in selected_samples_ex]
    unique_selected_UZ_sets = set(map(tuple, selected_UZ_samples))
    unique_selected_UZ_lists = [list(uz_set)[:15] for uz_set in unique_selected_UZ_sets]
    informer_data = []
    benefits = []
    for idx, subpopulation in enumerate(unique_selected_UZ_lists):
        p1, p2, p3, p4, p5, p6, p7, p8 = calculate_probabilities(subpopulation)
        subpopulation_values = [int(char) for char in subpopulation]
        benefit_values = calc_pns(subpopulation)
        benefits.append(benefit_values)
        subpopulation_label = f"subpopulation_{idx}"
        informer_data.append({
            'subpopulation': subpopulation_label,
            'subpopulation_values': subpopulation_values,
            'p1': p1,
            'p2': p2,
            'p3': p3,
            'p4': p4,
            'p5': p5,
            'p6': p6,
            'p7': p7,
            'p8': p8,
            'benefit' : benefit_values
        })
    benefit_mean = np.mean(benefits)
    for data in informer_data:
        data['group_label'] = 1 if data['benefit'] >= benefit_mean else 0
    df = pd.DataFrame(informer_data)
    df.to_csv('MT_1_8.csv', index=False)

def main_test_data_generation():
    num_features = 20
    C = -0.77953605542
    bernoulli_params_UZ = [0.524110233482, 0.689566064108, 0.180145428970, 0.317153536644, 0.046268153873,
                           0.340145244411, 0.100912238566, 0.772038172066, 0.913108434869, 0.364272299067,
                           0.063667554704, 0.454839320009, 0.586687215140, 0.018824647595, 0.871017316787,
                           0.164966968157, 0.578925020078, 0.983082980658, 0.018033993991, 0.074629121266]
    bernoulli_params_UX = 0.29908139311
    bernoulli_params_UY = 0.9226108109253
    coefficients_MX = np.array([0.843870221861, 0.178759296447, -0.372349746729, -0.950904544846, -0.439457721339,
                                -0.725970103834, -0.791203963585, -0.843183562918, -0.68422616618, -0.782051030131,
                                -0.434420454146, -0.445019418094, 0.751698021555, -0.185984172192, 0.191948271392,
                                0.401334543567, 0.331387702568, 0.522595634402, -0.928734581669, 0.203436441511])
    coefficients_MY = np.array([-0.453251661832, 0.424563325534, 0.0924810605305, 0.312680246141, 0.7676961338,
                                0.124337421843, -0.435341306455, 0.248957751703, -0.161303883519, -0.537653062121,
                                -0.222087991408, 0.190167775134, -0.788147770713, -0.593030174012, -0.308066297974,
                                0.218776507777, -0.751253645088, -0.11151455376, 0.785227235182, -0.568046522383])
    num_samples = 5000000
    all_subpopulations = list(itertools.product([0, 1], repeat=15))
    informer_data = []
    benefits = []
    for idx, subpopulation in enumerate(tqdm(all_subpopulations, desc="Calculating PNS Bounds")):
        p1, p2, p3, p4, p5, p6, p7, p8 = calculate_probabilities(subpopulation)
        subpopulation_values = list(subpopulation)
        benefit_values = calc_pns(subpopulation)
        benefits.append(benefit_values)
        subpopulation_label = f"subpopulation_{idx}"
        informer_data.append({
            'subpopulation': subpopulation_label,
            'subpopulation_values': subpopulation_values,
            'p1': p1,
            'p2': p2,
            'p3': p3,
            'p4': p4,
            'p5': p5,
            'p6': p6,
            'p7': p7,
            'p8': p8,
            'benefit': benefit_values
        })
    benefit_mean = np.mean(benefits)
    for data in informer_data:
        data['group_label'] = 1 if data['benefit'] >= benefit_mean else 0
    df = pd.DataFrame(informer_data)
    df_sampled = df.sample(n=200, random_state=42)
    df_sampled.to_csv('test_data_mt.csv', index=False)


class MultiTaskMLP(nn.Module):
    def __init__(self):
        super(MultiTaskMLP, self).__init__()
        self.shared_fc1 = nn.Linear(15, 512)
        self.dropout = nn.Dropout(0.5)
        self.task_fc2 = nn.ModuleList([nn.Linear(512, 256) for _ in range(8)])
        self.task_fc3 = nn.ModuleList([nn.Linear(256, 128) for _ in range(8)])
        self.task_fc4 = nn.ModuleList([nn.Linear(128, 1) for _ in range(8)])
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.leaky_relu(self.shared_fc1(x))
        x = self.dropout(x)
        task_outputs = []
        for i in range(8):
            task_x = F.leaky_relu(self.task_fc2[i](x))
            task_x = F.leaky_relu(self.task_fc3[i](task_x))
            task_output = torch.sigmoid(self.task_fc4[i](task_x))
            task_outputs.append(task_output)
        return torch.cat(task_outputs, dim=-1)

def train_model(model, criterion, optimizer, scheduler, num_epochs, train_loader):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')


def calculate_predicted_benefit(preds):
    lower_bound, upper_bound = calculate_pns_bounds(*preds)
    return (lower_bound + upper_bound) / 2

def evaluate_model(data_loader, model):
    model.eval()
    mse_loss = torch.nn.MSELoss()
    total_loss = 0.0
    count = 0
    correct = 0
    total = 0

    df = pd.read_csv('MT_1_8.csv')
    benefit_mean = np.mean(df['benefit'])

    with torch.no_grad():
        for inputs, labels in data_loader:
            preds = model(inputs)
            loss = mse_loss(preds, labels)
            total_loss += loss.item() * inputs.size(0)
            count += inputs.size(0)

            # Calculate group labels based on predicted p1 to p8 using calculate_predicted_benefit
            predicted_benefits = [calculate_predicted_benefit(pred.numpy()) for pred in preds]
            predicted_group_labels = torch.tensor([1 if benefit >= benefit_mean else 0 for benefit in predicted_benefits])

            # Compare predicted group labels with actual group labels
            actual_benefits = [calculate_predicted_benefit(label.numpy()) for label in labels]
            actual_group_labels = torch.tensor([1 if benefit >= benefit_mean else 0 for benefit in actual_benefits])
            correct += (predicted_group_labels == actual_group_labels).sum().item()
            total += actual_group_labels.size(0)

    accuracy = correct / total
    f1 = f1_score(actual_group_labels, predicted_group_labels, average='weighted')
    return total_loss / count, accuracy, f1

def main_model_training():
    df = pd.read_csv('MT_1_8.csv')
    features = [ast.literal_eval(observed)[:15] for observed in df['subpopulation_values']]
    labels = df[['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']].values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    dataset = TensorDataset(features, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    model = MultiTaskMLP()
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    train_model(model, criterion, optimizer, scheduler, num_epochs=600, train_loader=train_loader)
    val_loss, val_accuracy, val_f1 = evaluate_model(val_loader, model)
    print(f'Validation MSE Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}')
    test_df = pd.read_csv('test_data_mt.csv')
    test_df['subpopulation_values'] = test_df['subpopulation_values'].apply(ast.literal_eval)
    test_features = scaler.transform(test_df['subpopulation_values'].tolist())
    test_features = torch.tensor(test_features, dtype=torch.float32)
    test_labels = torch.tensor(test_df[['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']].values, dtype=torch.float32)
    test_dataset = TensorDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=64)
    test_loss, test_accuracy, test_f1 = evaluate_model(test_loader, model)
    print(f'Test MSE Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, F1 Score: {test_f1:.4f}')

if __name__ == "__main__":
    main_data_generation()
    main_test_data_generation()
    main_model_training()
