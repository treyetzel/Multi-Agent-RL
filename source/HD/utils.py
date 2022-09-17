import math

# basically taken from https://github.com/google-research/google-research/blob/master/hyperbolic_discount/agent_utils.py
def compute_eval_gamma_intervals(gamma_max, hyperbolic_exponent, number_of_gammas):
    if number_of_gammas > 1:
        b = math.exp(math.log(1. - math.pow(gamma_max, 1. / hyperbolic_exponent)) / number_of_gammas)
        eval_gammas = [1. - math.pow(b, i) for i in range(number_of_gammas)]
    else:
        eval_gammas = [gamma_max]
    
    return eval_gammas

# basically taken from https://github.com/google-research/google-research/blob/master/hyperbolic_discount/agent_utils.py
def integrate_q_values(q_values, integral_estimate, eval_gammas, number_of_gammas, gammas):
    integral = 0
    if integral_estimate == 'lower':
        gammas_plus_one = eval_gammas + [1.]
        weights = [gammas_plus_one[i+1] - gammas_plus_one[i] for i in range(number_of_gammas)]
        for weight, q_value in zip(weights, q_values):
            integral += weight * q_value
    elif integral_estimate == 'upper':
        gamma = eval_gammas
        weights = [gamma[i+1] - gamma[i] for i in range (number_of_gammas-1)]
        weights = [0.] + weights
        for weight, q_value in zip(weights, q_values):
            integral += weight * q_value
    else:
        raise NotImplementedError
    
    return integral