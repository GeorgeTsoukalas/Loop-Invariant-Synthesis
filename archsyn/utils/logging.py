import logging
import os
import dsl
import z3

# STUFF I ADDED
from fractions import Fraction
from functools import reduce
import itertools
from itertools import product
import operator
import numpy as np
import math

verbose = False

# Least common multiple function
def lcm(denominators):
    return reduce(lambda a,b: a*b // math.gcd(a,b), denominators)

# Returns a string version of a given predicate (Example: Affine(2x + 3y + 1) --> "2*x + 3*y + 1 >= 0")
def numericalInvariant_to_str(parameters, env, type_of_function): #parameters should come in np array format, bias can be in tensor format
    weights = [str(parameters[i]) + "*" + env[i] + " + " for i in range(len(parameters) - 1)]
    weights_str = reduce(operator.concat, weights, "")
    if type_of_function == "affine":
        return weights_str + str(parameters[-1:][0]) + " >= 0"
    elif type_of_function == "equality":
        return weights_str + str(parameters[-1:][0]) + " = 0"
    else:
        raise NotImplementedError("Passed in a function which was not affine or equality")

# Given an array of floating point values, returns a list of all possible floors/ceils of these values (Example: [1.2, 2.2] --> [[1,2], [1,3], [2,2], [2,3]])
def floor_ceil_combinations(arr):
    floor_ceil = [math.floor, math.ceil]
    return [list(map(lambda x: x[0](x[1]), zip(combination, arr))) for combination in itertools.product(floor_ceil, repeat=len(arr))]

# What follows is a series of different "smoothing" functions, some more aggressive than others (refer to writeup for example)
# Some of the functionality here can be reduced in size by using the Fraction package

def smoothed_numerical_invariant_new_nuclear(params): # created 2/13
    weights = (params["weights"][0].detach()).numpy()

    # next we will preprocess the weights to ensure ones that are too close to zero are removed
    for weight_index in range(len(weights)):
        copied_weights = [abs(weight/weights[weight_index]) for weight in weights]
        if max(copied_weights) >= 15: # 15 is a hyperparameter
            weights[weight_index] = 0

    # Find the smallest nonzero element in weights (that is not the bias) (smallest means in magnitude, it could be negative)
    smallest_nonzero_weight = 1000
    for weight in weights:
        if abs(weight) < smallest_nonzero_weight and weight != 0.0:
            smallest_nonzero_weight = weight
    
    # the bias range should now be between bias, bias + smallest_nonzero_weight
    bias = float(params["bias"][0].detach())
    bias_new = bias + abs(smallest_nonzero_weight) # I believe, after further analysis, it should always be positive exclusive (but one extra point probably won't hurt at this depth)

    # now do the smoothing operation
    biggest_weight = np.max(np.absolute(weights)) # this does not contain the bias term (it shouldn't)
    assert biggest_weight != 0 # just in case

    new_weights = [weight/biggest_weight for weight in weights]

    approximations = []
    N = 5 # this is the maximum denominator allowed, it is a hyperparameter (N = 2) is also good
    # This can also be vastly simplified by using the Fraction package
    for new_weight in new_weights:
        second_closest_approx_values = (-50, -50)
        second_closest_approx = 999
        closest_approx_values = (0, -100)
        closest_approx = 1000
        if new_weight == 0:
            second_closest_approx_values = (0,1)
            closest_approx_values = (0, 1)
            approximations.append([closest_approx_values, second_closest_approx_values])
            continue
        for i in range(-1*N - 1, N + 1):
            for j in range(1, N+1):
                if (abs(new_weight - i/j) < closest_approx and (np.sign(new_weight) == np.sign(i) or i == 0)):
                    second_closest_approx_values = closest_approx_values
                    second_closest_approx = closest_approx
                    closest_approx = abs(new_weight - i/j)
                    closest_approx_values = (i,j)
        approximations.append([closest_approx_values, second_closest_approx_values])

    # Now we have a choice of two fractional representations for each coefficient.
    smoothed_params = []
    for approximation in itertools.product(*approximations): # this wont ignore if the second closest approx is -1000,-100
        # simplify the fractional approximation for each coefficient to be in reduced form
        new_approximation = [(int(approx[0]/math.gcd(approx[0], approx[1])), int(approx[1]/math.gcd(approx[0], approx[1]))) for approx in approximation]
        least_common_multiple = lcm([frac[1] for frac in new_approximation])

        # adjust the bias accordingly 
        # we allow for a range [bias_floor, bias_ceil] to compensate for the fact that training does not always produce accurate bias terms (TODO: this is a point to focus on improving)
        bias_floor = (bias - abs(smallest_nonzero_weight)) * least_common_multiple/biggest_weight
        bias_ceil = (bias + abs(smallest_nonzero_weight)) * least_common_multiple/biggest_weight

        # multiply the new_weights by the lcm of the denominators to get integer coefficients
        weights_smoothed = [least_common_multiple * new_approximation[i][0]/new_approximation[i][1] for i in range(len(weights))]
        smoothed_bias_min = int(min(bias_floor, bias_ceil))
        smoothed_bias_max = int(max(bias_floor, bias_ceil))
        for bias_term in range(smoothed_bias_min, smoothed_bias_max + 1): # inclusive
            smoothed_params.append(weights_smoothed + [bias_term])
    # smoothed_params is of type list (list int) of different potential approximations of the generated predicate
    return smoothed_params

# mostly the same as the smoothed_numerical_invariant_new_nuclear, refer to it for comments
def smoothed_numerical_invariant_fourth(params): 
    weights = list((params["weights"][0].detach()).numpy())

    # next we will preprocess the weights to ensure ones that are too close to zero are removed
    for weight_index in range(len(weights)):
        copied_weights = [abs(weight/weights[weight_index]) for weight in weights]
        if max(copied_weights) >= 10: # 10 is a hyperparameter (though probably insignificant)
            weights[weight_index] = 0

    # Find the smallest nonzero element in weights
    smallest_nonzero_weight = 1000
    for weight in weights:
        if abs(weight) < smallest_nonzero_weight and weight != 0.0:
            smallest_nonzero_weight = weight

    bias = float(params["bias"][0].detach())
    weights.append(bias)
    assert smallest_nonzero_weight != 0
    assert smallest_nonzero_weight != 1000
    for i in range(len(weights)):
        weights[i]/=abs(smallest_nonzero_weight)
    
    # here is the main innovation - we also attempt the extra 2^|num vars| possibilities where we floor and ceil the weights additionally using floor_ceil_combinations
    smoothed_params = floor_ceil_combinations(weights) 
    
    # this one is not fully fleshed out
    return smoothed_params

# this one is intended to mimic the cln2inv paper implementatoin
def smoothed_numerical_invariant_third(params):
    weights = list((params["weights"][0].detach()).numpy())
    bias = float(params["bias"][0].detach())
    weights.append(bias)
    # next we will preprocess the weights to ensure ones that are too close to zero are removed

    for weight_index in range(len(weights)):
        copied_weights = [abs(weight/weights[weight_index]) for weight in weights]
        if max(copied_weights) >= 10: # 10 is a hyperparameter
            weights[weight_index] = 0

    # Find the smallest nonzero element in weights
    smallest_nonzero_weight = 1000
    for weight in weights:
        if abs(weight) < smallest_nonzero_weight and weight != 0.0:
            smallest_nonzero_weight = weight
    
    assert smallest_nonzero_weight != 0
    assert smallest_nonzero_weight != 1000
    for i in range(len(weights)):
        weights[i]/=abs(smallest_nonzero_weight) 
    print(weights)
    weights_np = np.asarray(weights)
    scaled_weights = np.round(weights_np)
    smoothed_params = [list(scaled_weights)]

    return smoothed_params

def smoothed_numerical_invariant_cln2inv(params):
    weights = (params["weights"][0].detach()).numpy()
    # next we will preprocess the weights to ensure ones that are too close to zero are removed, 
    for weight_index in range(len(weights)):
        copied_weights = [abs(weight/weights[weight_index]) for weight in weights]
        if max(copied_weights) >= 10: # 15 is a tunable parameter,
            weights[weight_index] = 0

    # Find the smallest nonzero element in weights
    smallest_nonzero_weight = 1000
    for weight in weights:
        if abs(weight) < smallest_nonzero_weight and weight != 0.0: 
            smallest_nonzero_weight = weight
    
    assert smallest_nonzero_weight != 0
    assert smallest_nonzero_weight != 1000
    bias = float(params["bias"][0].detach())
    bias_floor = (bias/abs(smallest_nonzero_weight) - 1)
    bias_ceil = (bias/abs(smallest_nonzero_weight) + 1)
    weights /= abs(smallest_nonzero_weight) 
    max_denominator = 5 # this is a hyperparameter (called N in above smoothing functions)
    frac_approximations = []
    denominator = 1

    # here I used the Fraction package, notice the shortening of the code
    for coeff in weights:
        frac = Fraction.from_float(float(coeff)).limit_denominator(max_denominator)
        frac_approximations.append(frac)
        denominator = denominator * frac.denominator // math.gcd(denominator, frac.denominator) # this is essentially a fold/reduce, in the style of cln2inv code
    new_weights = [math.floor(a * denominator) for a in frac_approximations] 

    smoothed_params = []
    smoothed_bias_min = int(min(bias_floor, bias_ceil))
    smoothed_bias_max = int(max(bias_floor, bias_ceil))
    for bias_term in range(smoothed_bias_min, smoothed_bias_max + 1):
        smoothed_params.append(new_weights + [bias_term])

    # as presented, this has very little bias mobility as this essentially gives at most 3 bias choices per predicate
    return smoothed_params



# This is the one in use currently
def smoothed_numerical_invariant_new(params): # created 2/2
    weights = (params["weights"][0].detach().cpu()).numpy() # .cpu() included for different device support

    # next we will preprocess the weights to ensure ones that are too close to zero are removed
    for weight_index in range(len(weights)):
        copied_weights = [abs(weight/weights[weight_index]) for weight in weights]
        if max(copied_weights) >= 15: # 15 is a hyperparameter
            weights[weight_index] = 0

    # Find the smallest nonzero element in weights
    smallest_nonzero_weight = 1000
    for weight in weights:
        if abs(weight) < smallest_nonzero_weight and weight != 0.0: 
            smallest_nonzero_weight = weight

    bias = float(params["bias"][0].detach())
    bias_new = bias + abs(smallest_nonzero_weight) 

    # now do the smoothing operation
    biggest_weight = np.max(np.absolute(weights)) 
    assert biggest_weight != 0 # just in case

    new_weights = [weight/biggest_weight for weight in weights] 

    approximations = []
    N = 2 # maximum denominator allowed, this is a hyperparameter (lower can be better!)
    for new_weight in new_weights:
        closest_approx_values = (-100, -100)
        closest_approx = 1000
        if new_weight == 0:
            closest_approx_values = (0, 1)
            approximations.append(closest_approx_values)
            continue
        for i in range(-1*N - 1, N + 1):
            for j in range(1, N+1):
                if (abs(new_weight - i/j) < closest_approx and (np.sign(new_weight) == np.sign(i) or i == 0)):
                    closest_approx = abs(new_weight - i/j)
                    closest_approx_values = (i,j)
        approximations.append(closest_approx_values)

    # we now have fraction representations, next step is to simplify these down
    new_approximations = [(int(approx[0]/math.gcd(approx[0], approx[1])), int(approx[1]/math.gcd(approx[0], approx[1]))) for approx in approximations]

    # now we want to multiply each part by the LCM of all the denominators to get smallest integers
    least_common_multiple = lcm([frac[1] for frac in new_approximations]) # this should always be positive as denominator > 0 always
    smoothed_params = []

    bias_floor = (bias - abs(smallest_nonzero_weight)) * least_common_multiple/biggest_weight
    bias_ceil = bias_new * least_common_multiple/biggest_weight

    weights_smoothed = [least_common_multiple * new_approximations[i][0]/new_approximations[i][1] for i in range(len(weights))]

    smoothed_bias_min = int(min(bias_floor, bias_ceil))
    smoothed_bias_max = int(max(bias_floor, bias_ceil))

    for bias_term in range(smoothed_bias_min-1, smoothed_bias_max + 2): # inclusive (a little more expanded in this one, just in case)
        smoothed_params.append(weights_smoothed + [bias_term])

    return smoothed_params

# this is the original implementation
def smoothed_numerical_invariant(params):
    weights = (params["weights"][0].detach()).numpy()
    biggest_weight =  abs(np.max(weights)) 
    assert biggest_weight != 0 
    bias = float(params["bias"][0].detach())/biggest_weight
    new_weights = [weight/biggest_weight for weight in weights]
    approximations = []
    N = 5 # this is the maximum denominator allowed
    for new_weight in new_weights:
        closest_approx_values = (-100, -100)
        closest_approx = 1000
        for i in range(-5, N+1):
            for j in range(1, N+1):
                if (abs(new_weight - i/j) < closest_approx and (np.sign(new_weight) == np.sign(i) or i == 0)):
                    closest_approx = abs(new_weight - i/j)
                    closest_approx_values = (i,j)
        approximations.append(closest_approx_values)

    # Problem which the following code addresses: (5,5) = (1,1) as an approximation but the latter one will be chosen. We want the approximation denominators to be small as possible.
    new_approximations = [(int(approx[0]/math.gcd(approx[0], approx[1])), int(approx[1]/math.gcd(approx[0], approx[1]))) for approx in approximations]
    least_common_multiple = lcm([frac[1] for frac in new_approximations])

    # I do not believe this one will work with the current implementation as new_main is expecting a value of type list (list int)
    return [least_common_multiple * approximations[i][0]/approximations[i][1] for i in range(len(weights))] + [ math.ceil(least_common_multiple * bias)] #early investigations looked into how to find a better bias term

def print_program2(program, env, smoothed = False):
    if program.name == "affine" or program.name == "equality":
        if smoothed:
            weights = smoothed_numerical_invariant(program.parameters)
            print("( " + program.name + " " + numericalInvariant_to_str(weights, env, program.name))
        else:
            weights = list((program.parameters["weights"][0].detach().cpu()).numpy())
            bias = float(program.parameters["bias"][0].detach())
            weights.append(bias) #converting to proper form
            print("( " + program.name + " " + numericalInvariant_to_str(weights, env, program.name))
    else:
        print("(" + program.name)
        for submodule, function in program.submodules.items():
            print_program2(function, env, smoothed)
        print(" )")

# The .pop(0) I recall was included because I was unsure if copies of the array were being made and wanted to be sure I still pointed to the correct data
# I am not sure if this was ever a problem or if this is the best way to solve it
def invariant_from_program_new(program, params, env):
    if program.name == "affine":
        weights = params.pop(0) # again, the question is does this modify params?
        z3_ineq = 0
        z3_ineq = sum(weight * z3.Real(var) for weight, var in zip(weights, env)) + weights[-1] >= 0
        return z3_ineq
    elif program.name == "equality":
        weights = params.pop(0) # again, the question is does this modify params?
        z3_eq = 0
        z3_eq = sum(weight * z3.Real(var) for weight, var in zip(weights, env)) + weights[-1] == 0
        return z3_eq
    elif program.name == "and":
        funcs = []
        for submodule, function in program.submodules.items():
            funcs.append(function) 
        return z3.And(invariant_from_program_new(funcs[0], params, env), invariant_from_program_new(funcs[1], params, env)) # this assumes only 2 elements in funcs, if you ever expand this you will need to modify
    elif program.name == "or":
        funcs = []
        for submodule, function in program.submodules.items():
            funcs.append(function) 
        return z3.Or(invariant_from_program_new(funcs[0], params, env), invariant_from_program_new(funcs[1], params, env))
        
# Deprecated by the above function
def invariant_from_program(program, env):
    if program.name == "affine":
        weights = smoothed_numerical_invariant(program.parameters)
        z3_ineq = 0
        z3_ineq = sum(weight * z3.Real(var) for weight, var in zip(weights, env)) + weights[-1] >= 0.0
        return z3_ineq
    elif program.name == "equality":
        weights = smoothed_numerical_invariant(program.parameters)
        z3_eq = 0
        z3_eq = sum(weight*z3.Real(var) for weight, var in zip(weights, env)) + weights[-1] == 0.0 # z3 equality is defined with ==
        return z3_eq
    elif program.name == "and":
        funcs = []
        for submodule, function in program.submodules.items():
            funcs.append(function)
        return z3.And(invariant_from_program(funcs[0], env), invariant_from_program(funcs[1], env))
    elif program.name == "or":
        funcs = []
        for submodule, function in program.submodules.items():
            funcs.append(function)
        return z3.Or(invariant_from_program(funcs[0], env), invariant_from_program(funcs[1], env))
    
def init_logging(save_path):
    logfile = os.path.join(save_path, 'log.txt')

    # clear log file
    with open(logfile, 'w'):
        pass
    # remove previous handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=logfile, level=logging.INFO, format='%(message)s')

def log_and_print(line):
    if verbose: # this can be tuned by a variable at the top of this file
        print(line)
    logging.info(line)

# this is deprecated I believe (by print_program2)
def print_program(program, ignore_constants=True):
    if not isinstance(program, dsl.LibraryFunction):
        return program.name
    else:
        collected_names = []
        for submodule, functionclass in program.submodules.items():
            collected_names.append(print_program(functionclass, ignore_constants=ignore_constants))
        if program.has_params:
            parameters = "params: {}".format(program.parameters.values())
            if not ignore_constants:
                collected_names.append(parameters)
        joined_names = ', '.join(collected_names)
        return program.name + "(" + joined_names + ")"

def print_program_dict(prog_dict):
    log_and_print(print_program(prog_dict["program"], ignore_constants=True))
    log_and_print("struct_cost {:.4f} | score {:.4f} | path_cost {:.4f} | time {:.4f}".format(
        prog_dict["struct_cost"], prog_dict["score"], prog_dict["path_cost"], prog_dict["time"]))
