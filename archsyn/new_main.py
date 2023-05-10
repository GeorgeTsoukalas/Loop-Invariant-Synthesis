from __future__ import print_function
import math
from functools import reduce
import os
import sys
import numpy as np
import torch
import pandas as pd
import random
import torch.optim as optim
import itertools
from itertools import chain
from tqdm import tqdm
import z3
import ast
import argparse
import pickle
import time
import heapq

# import program_learning
from dataset import Dataset
from algorithms import NAS
from program_graph import ProgramGraph
from utils.data_loader import CustomLoader, IOExampleLoader
from utils.evaluation import label_correctness, value_correctness

from utils.logging import init_logging, log_and_print, print_program, print_program2, lcm, numericalInvariant_to_str, smoothed_numerical_invariant, smoothed_numerical_invariant_new, invariant_from_program_new
from utils.logging import smoothed_numerical_invariant_cln2inv, smoothed_numerical_invariant_third, smoothed_numerical_invariant_fourth, smoothed_numerical_invariant_new_nuclear
from utils.logging import invariant_from_program
from utils.loss import SoftF1LossWithLogits
from dsl_inv import DSL_DICT, CUSTOM_EDGE_COSTS


from dsl.library_functions import LibraryFunction

from metal.common.utils import CEHolder
from metal.common.constants import CE_KEYS
from metal.parser.sygus_parser import SyExp

from utils.variable_dict import variable_dictionary
# import cln2inv stuff

from cln2inv_stuff.invariant_checking import InvariantChecker

import pdb



def parse_args():
    parser = argparse.ArgumentParser()
    # cmd_args for experiment setup
    # parser.add_argument('-t', '--trial', type=int, required=True,
    #                     help="trial ID")
    # parser.add_argument('--exp_name', type=str, required=True,
    #                     help="experiment_name")
    parser.add_argument('--save_dir', type=str, required=False, default="results/",
                        help="directory to save experimental results")

    # cmd_args for data
    # parser.add_argument('--train_data', type=str, required=True,
    #                     help="path to train data")
    # parser.add_argument('--test_data', type=str, required=True,
    #                     help="path to test data")
    # parser.add_argument('--valid_data', type=str, required=False, default=None,
    #                     help="path to val data. if this is not provided, we sample val from train.")
    # parser.add_argument('--train_labels', type=str, required=True,
    #                     help="path to train labels")
    # parser.add_argument('--test_labels', type=str, required=True,
    #                     help="path to test labels")
    # parser.add_argument('--valid_labels', type=str, required=False, default=None,
    #                     help="path to val labels. if this is not provided, we sample val from train.")
    # parser.add_argument('--input_type', type=str, required=True, choices=["atom", "list"],
    #                     help="input type of data")
    # parser.add_argument('--output_type', type=str, required=True, choices=["atom", "list"],
    #                     help="output type of data")
    # parser.add_argument('--input_size', type=int, required=True,
    #                     help="dimenion of features of each frame")
    # parser.add_argument('--output_size', type=int, required=True,
    #                     help="dimension of output of each frame (usually equal to num_labels")
    # parser.add_argument('--num_labels', type=int, required=True,
    #                     help="number of class labels")

    # cmd_args for program graph
    parser.add_argument('--max_num_units', type=int, required=False, default=16,
                        help="max number of hidden units for neural programs")
    parser.add_argument('--min_num_units', type=int, required=False, default=4,
                        help="max number of hidden units for neural programs")
    parser.add_argument('--max_num_children', type=int, required=False, default=10,
                        help="max number of children for a node")
    parser.add_argument('--max_depth', type=int, required=False, default=3,
                        help="max depth of programs")
    parser.add_argument('--penalty', type=float, required=False, default=0.0,
                        help="structural penalty scaling for structural cost of edges")
    parser.add_argument('--ite_beta', type=float, required=False, default=1.0,
                        help="beta tuning parameter for if-then-else")
    parser.add_argument('--sem', type=str, required=False, choices=["arith","minmax","luka"], default="minmax",
                        help="discrete semantics approximation")

    # cmd_args for training
    parser.add_argument('--train_valid_split', type=float, required=False, default=1.0, #TODO: changed this to 1 no need for validation here
                        help="split training set for validation."+\
                        " This is ignored if validation set is provided using valid_data and valid_labels.")
    parser.add_argument('--normalize', action='store_true', required=False, default=False,
                        help='whether or not to normalize the data')
    parser.add_argument('--batch_size', type=int, required=False, default=50,
                        help="batch size for training set")
    parser.add_argument('-lr', '--learning_rate', type=float, required=False, default=0.045,
                        help="learning rate")
    parser.add_argument('-search_lr', '--search_learning_rate', type=float, required=False, default=0.02,
                        help="learning rate")
    parser.add_argument('--neural_epochs', type=int, required=False, default=4,
                        help="training epochs for neural programs")
    parser.add_argument('--symbolic_epochs', type=int, required=False, default=50,
                        help="training epochs for symbolic programs")
    parser.add_argument('--pre_cooked_data', type=bool, required=False, default=False,
                        help="whether to use cln2inv premade data")
    parser.add_argument('--top_k_programs', type=int, required=False, default=4,
                        help="training epochs for symbolic programs")      
    parser.add_argument('--max_train_data', type=int, required=False, default=20,
                        help="training epochs for symbolic programs")                                            
    # parser.add_argument('--lossfxn', type=str, required=True, choices=["crossentropy", "bcelogits", "softf1"],
    #                     help="loss function for training")
    parser.add_argument('--f1double', action='store_true', required=False, default=False,
                        help='whether use double for soft f1 loss')
    parser.add_argument('--class_weights', type=str, required=False, default = None,
                        help="weights for each class in the loss function, comma separated floats")
    parser.add_argument('--topN_select', type=int, required=False, default=2,
                        help="number of candidates remain in each search")
    parser.add_argument('--resume_graph', type=str, required=False, default=None,
                        help="resume graph from certain path if necessary")
    parser.add_argument('--sec_order', action='store_true', required=False, default=False,
                        help='whether use second order for architecture search')
    parser.add_argument('--spec_design', action='store_true', required=False, default=False,
                        help='if specific, train process is defined manually')
    parser.add_argument('--random_seed', type=int, required=False, default=0,
                        help="manual seed")
    parser.add_argument('--finetune_epoch', type=int, required=False, default=12, #CHANGED --finetune_epoch to --finetune_epochs
                        help='Epoch for finetuning the result graph.')
    parser.add_argument('--finetune_lr', type=float, required=False, default=0.01,
                        help='Epoch for finetuning the result graph.')

    # cmd_args for algorithms
    # parser.add_argument('--algorithm', type=str, required=True,
    #                     choices=["mc-sampling", "mcts", "enumeration", "genetic", "astar-near", "iddfs-near", "rnn", 'nas'],
    #                     help="the program learning algorithm to run")
    parser.add_argument('--frontier_capacity', type=int, required=False, default=float('inf'),
                        help="capacity of frontier for A*-NEAR and IDDFS-NEAR")
    parser.add_argument('--initial_depth', type=int, required=False, default=1,
                        help="initial depth for IDDFS-NEAR")
    parser.add_argument('--performance_multiplier', type=float, required=False, default=1.0,
                        help="performance multiplier for IDDFS-NEAR (<1.0 prunes aggressively)")
    parser.add_argument('--depth_bias', type=float, required=False, default=1.0,
                        help="depth bias for  IDDFS-NEAR (<1.0 prunes aggressively)")
    parser.add_argument('--exponent_bias', type=bool, required=False, default=False,
                        help="whether the depth_bias is an exponent for IDDFS-NEAR"+
                        " (>1.0 prunes aggressively in this case)")
    parser.add_argument('--num_mc_samples', type=int, required=False, default=10,
                        help="number of MC samples before choosing a child")
    parser.add_argument('--max_num_programs', type=int, required=False, default=100,
                        help="max number of programs to train got enumeration")
    parser.add_argument('--population_size', type=int, required=False, default=10,
                        help="population size for genetic algorithm")
    parser.add_argument('--selection_size', type=int, required=False, default=5,
                        help="selection size for genetic algorithm")
    parser.add_argument('--num_gens', type=int, required=False, default=10,
                        help="number of genetions for genetic algorithm")
    parser.add_argument('--total_eval', type=int, required=False, default=100,
                        help="total number of programs to evaluate for genetic algorithm")
    parser.add_argument('--mutation_prob', type=float, required=False, default=0.1,
                        help="probability of mutation for genetic algorithm")
    parser.add_argument('--max_enum_depth', type=int, required=False, default=7,
                        help="max enumeration depth for genetic algorithm")
    parser.add_argument('--cell_depth', type=int, required=False, default=3,
                        help="max depth for each cell for nas algorithm")


    parser.add_argument('-data_root', default=None, help='root of dataset')
    parser.add_argument('-file_list', default=None, help='list of programs')
    parser.add_argument('-single_sample', default=None, type=str, help='tune single program')
    parser.add_argument('-use_interpolation', default=0, type=int, help='whether use interpolation')
    parser.add_argument('-top_left', type=bool, default=False, help="set to true to use top-left partition")
    parser.add_argument('-GM', type=bool, default=False, help="set to true to use Gradient Matching")

    parser.add_argument('--problem_num', type = int, default = 0, help = "The problem number from the cln2inv benchmarks")
    return parser.parse_args()

#Print program in Sygus format. (Not used anywhere currently, don't need SyExp format to evaluate the program anymore)
def convert_to_sygus(program):
    if not isinstance(program, LibraryFunction):
        return SyExp(program.name, [])
    else:
        if program.has_params:
            Q = SyExp(program.name, [program.parameters])
            return SyExp(program.name, [program.parameters])
        else:
            collected_names = []
            for submodule, functionclass in program.submodules.items():
                collected_names.append(convert_to_sygus(functionclass))
            return SyExp(program.name, collected_names)


def reward_w_interpolation(sample_index, holder, lambda_holder_eval, lambda_new_ce):
    # check if it passes
    status, key, ce = lambda_new_ce()
    if status > 0:
        return 1.0

    # interpolate ce and add neary ones into the buffer
    holder.interpolate_ce(ce)

    #harmonic mean
    scores = []
    for key in CE_KEYS:
        score = lambda_holder_eval(key)
        scores.append(score)
    t = sum(scores) # t \in [0, 2.0]
    if t > 0:
        hm_t = 4.0 * scores[0] * scores[1] / t
    else:
        hm_t = 0.0

    return -2.0 + hm_t


def reward_1(sample_index, holder, lambda_holder_eval, lambda_new_ce):
    # print("\n\nsample_index:", sample_index)
    # holder.show_stats()
    # ct = 0
    # s = 0
    scores = []
    for key in CE_KEYS:
        score = lambda_holder_eval(key)
        # print("key:", key,  "score: ", score, "ce_per_key:", holder.ce_per_key)
        # if key in holder.ce_per_key:
        #     ct += len(holder.ce_per_key[key].ce_list)
        #     s += 0.99
        scores.append(score)
    t = sum(scores) # t \in [0, 2.0]
    if t > 0:
        hm_t = 4.0 * scores[0] * scores[1] / t
    else:
        hm_t = 0.0
    # print("ct=",ct, "t=", t, "s=",s)

    return -2.0 + hm_t

def function_accuracy(func, data, labels):
    Missed = []
    for datum in zip(data,labels): # this is for checking that the output function actually works before smoothing
        if datum[1][0] == 2.0: # false
            if func(*(datum[0][0])):
                Missed.append(list(datum[0]))
        elif datum[1][0] == 1.0: # true
            if not func(*(datum[0][1])):
                Missed.append(list(datum[0]))
        elif datum[1][0] == 3.0: # implication example
            if not ((not func(*(datum[0][0]))) or (func(*(datum[0][1])))):
                Missed.append(list(datum[0])) 
    return Missed

cln2inv_invariant_dictionary = {
    15: lambda x,m,n: (m == 0) or (m-n < 0),
    18: lambda x,m,n: (x >= 1) and (m >= 1),
    59: lambda c,n: (c == 0) and(n > 0),
    64: lambda x,y: (x - 10 <= 0) or (y - 10 < 0),
    83: lambda x,y: (x < 0) or (y > 0),
    95: lambda i,j,x,y: (y == 1) and (i - j == 0),
    99: lambda n,x,y: (n - x - y == 0),
    124: lambda i,j,x,y: (i - j - x + y == 0),
    103: lambda x: (x - 100 <= 0),
    6: lambda x,t,y,z: (x == 0) or (z - y >= 0)
}

def get_all_params(program, Smoothed = False):
    params = []
    if program.name == "affine":
        if Smoothed:
            if i == 1:
                vals = smoothed_numerical_invariant_new(program.parameters)
            elif i == 2:
                vals = smoothed_numerical_invariant_cln2inv(program.parameters)
            elif i == 3:
                vals = smoothed_numerical_invariant_third(program.parameters)
            elif i == 4:
                vals = smoothed_numerical_invariant_fourth(program.parameters)
            elif i == 5:
                vals = smoothed_numerical_invariant_new_nuclear(program.parameters)
            #print(vals)
            params.append(vals)
        else:
            vals = [float(x.detach()) for x in program.parameters["weights"][0]] + [float(program.parameters["bias"][0].detach())]
            params.append([vals])
    elif program.name == "equality":
        if Smoothed:
            if i == 1:
                vals = smoothed_numerical_invariant_new(program.parameters)
            elif i == 2:
                vals = smoothed_numerical_invariant_cln2inv(program.parameters)
            elif i == 3:
                vals = smoothed_numerical_invariant_third(program.parameters)
            elif i == 4:
                vals = smoothed_numerical_invariant_fourth(program.parameters)
            elif i == 5:
                vals = smoothed_numerical_invariant_new_nuclear(program.parameters)
            params.append(vals)
        else:
            vals = [float(x.detach()) for x in program.parameters["weights"][0]] + [float(program.parameters["bias"][0].detach())]
            params.append([vals])
    elif program.name == "and":
        params += get_all_params(list(program.submodules.items())[0][1], Smoothed)
        params += get_all_params(list(program.submodules.items())[1][1], Smoothed)
    elif program.name == "or":
        params += get_all_params(list(program.submodules.items())[0][1], Smoothed)
        params += get_all_params(list(program.submodules.items())[1][1], Smoothed)
    return params

def lambda_program_generator_new(program, params):
    if program.name == "affine":
        copy_of_params = [x for x in params[0]]
        func = lambda *args: sum(val * arg for val, arg in zip(copy_of_params, args)) + copy_of_params[-1] >= 0 
        params.pop(0)
        return func
    elif program.name == "equality":
        copy_of_params = [x for x in params[0]] # reference issue I think
        func = lambda *args: sum(val * arg for val, arg in zip(copy_of_params, args)) + copy_of_params[-1] == 0
        params.pop(0)
        return func
    elif program.name == "and":
        func1 = (lambda_program_generator_new(list(program.submodules.items())[0][1], params)) 
        func2 = (lambda_program_generator_new(list(program.submodules.items())[1][1], params)) 
        return lambda *args  : func1(*args) and func2(*args)
    elif program.name == "or":
        func1 = (lambda_program_generator_new(list(program.submodules.items())[0][1], params)) 
        func2 = (lambda_program_generator_new(list(program.submodules.items())[1][1], params)) 
        return lambda *args  : func1(*args) or func2(*args)

def evaluate(algorithm, graph, train_loader, train_config, device):
    validset = train_loader.validset
    with torch.no_grad():
        metric = algorithm.eval_graph(graph, validset, train_config['evalfxn'], train_config['num_labels'], device)
    return metric

def run_on_problem(problem_num, cmd_args, neural_epochs, symbolic_epochs, max_depth, batch_size, lr, i, pre_cooked_data, top_k, random_seed, max_train_data):
    
    # added stuff from cln2inv implementation
    fname = str(problem_num) + '.c'
    csvname = str(problem_num) + '.csv'
    src_path = 'benchmarks-cln2inv/code2inv/c/'
    check_path = 'benchmarks-cln2inv/code2inv/smt2'
    trace_path = 'benchmarks-cln2inv/code2inv/csv/'


    env = variable_dictionary[problem_num]

    invariantChecker = InvariantChecker(fname, check_path)
    # manual seed all random for debug
    log_and_print('random seed {}'.format(cmd_args.random_seed))
    torch.random.manual_seed(cmd_args.random_seed)
    #torch.manual_seed(cmd_args.seed)
    np.random.seed(cmd_args.random_seed)
    random.seed(cmd_args.random_seed)

    full_exp_name = 'Test'
    save_path = os.path.join(cmd_args.save_dir, full_exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # init log
    init_logging(save_path)
    log_and_print("Starting experiment {}\n".format(full_exp_name))

    #///////////////////////////////
    #///////////////////////////////
    #///////////////////////////////

    # TODO allow user to choose device
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    Testing = True 

    # If you tweak this definition of the loss function, make the analagous change to the loss function in evaluation.py
    def lossfxn_t(out, labels):
        #LOG VERSION OF LOSS
        eps = 1e-6 # I believe this is the same as the japan paper (see writeup for a reference)
        augmented_out = [out[l][1] if labels[l] == 1 else (torch.ones(1).to(device) - out[l][0] if labels[l] == 2 else torch.ones(1).to(device) - (torch.ones(1).to(device) - out[l][1])*out[l][0]) for l in range(len(out))]
        return torch.tensor(-1).to(device) * torch.mean(torch.log(torch.hstack(augmented_out) + eps)) 
        
        # MSE VERSION OF LOSS
        #augmented_out = [(torch.ones(1).to(device) - out[l][1]) if labels[l] == 1 else (out[l][0] if labels[l] == 2 else (torch.ones(1).to(device) - (torch.ones(1).to(device) - out[l][0])*out[l][1])) for l in range(len(out))] 
        #augmented_out = [(torch.ones(1).to(device) - out[l][1]) if labels[l] == 1 else (out[l][0] if labels[l] == 2 else (torch.ones(1).to(device) - out[l][1])*out[l][0]) for l in range(len(out))]
        #return torch.mean(torch.hstack(augmented_out))
    
    # this wrapper is here for converting the output to the correct device (this is the solution I found, not sure if there is a better one)
    def lossfxn_wrapper(out, labels):
        return lossfxn_t(out, labels).to(device)

    lossfxn = lossfxn_wrapper if device == "cuda:0" else lossfxn_t

    input_size = len(env) 
    output_size = 1
    num_labels = 1
    input_type = output_type = "atom"

    train_config = {
        'arch_lr' : lr,#cmd_args.search_learning_rate,
        'model_lr' : lr,#cmd_args.search_learning_rate,
        'train_lr' : lr,#cmd_args.learning_rate,
        'search_epoches' : cmd_args.neural_epochs,
        'finetune_epoches' : cmd_args.symbolic_epochs,
        'arch_optim' : optim.Adam,
        'model_optim' : optim.Adam,
        'lossfxn' : lossfxn,
        'evalfxn' : value_correctness,
        'num_labels' : num_labels,
        'save_path' : save_path,
        'topN' : cmd_args.topN_select,
        'arch_weight_decay' : 0,
        'model_weight_decay' : 0,
        'penalty' : cmd_args.penalty,
        'secorder' : cmd_args.sec_order,
        'specific' : [#[None, 2, 0.01, 5], [4, 2, 0.01, 5], [3, 2, 0.01, 5], [2, 2, 0.01, 5], \
                [None, max_depth, lr, neural_epochs]]#, ["astar", max_depth, 0.1, 5]]#, [4, 4, 0.01, 500], [3, 4, 0.01, 500], [2, 4, 0.01, 500]]#, ["astar", 4, 0.01, cmd_args.neural_epochs]]
    }

    # Initialize program graph
    if cmd_args.resume_graph is None:
        program_graph = ProgramGraph(DSL_DICT, input_type, output_type, input_size, output_size,
                                    cmd_args.max_num_units, cmd_args.min_num_units, max_depth,
                                    device, ite_beta = cmd_args.ite_beta, cfg = None, var_ids = None, root_symbol = None)
        start_depth = 0
    else:
        assert os.path.isfile(cmd_args.resume_graph)
        program_graph = pickle.load(open(cmd_args.resume_graph, "rb"))
        program_graph.max_depth = max_depth
        start_depth = program_graph.get_current_depth()

    # Initialize algorithm
    algorithm = NAS(frontier_capacity=cmd_args.frontier_capacity)
    verification_iter = 0
    if pre_cooked_data:
        train_data, train_labels = load_spreadsheet_data_test(problem_num, env)
    else:
        train_data = []
        train_labels = []

    # To get the first datapoint, we check using this potential loop invariant
    # The cln2inv paper does not do this, they have some premade progrma trace data available in spreadsheets
    non_loop_invariant = 1.0 * z3.Real(env[0]) >= 0.0 
    # TODO: here we could change this with some warm-starting of the program

    result = invariantChecker.check_cln([non_loop_invariant], env)

    if result[0]:
        return "Solved!", result[1], verification_iter
    else:
        if result[2] == "loop":
            train_data.append(result[3])
            train_labels.append([3.])
        else:
            if result[2] == "pre": # then the result is a true value that the invariant fails on
                train_data.append([[-1000. for i in env], [float(result_element) for result_element in result[3]]])
                train_labels.append([1.])
            elif result[2] == "post": # then this is actually a false value 
                train_data.append([[float(result_element) for result_element in result[3]], [-1000. for i in env]])
                train_labels.append([2.])
    
    # next we begin the loop
    while verification_iter < max_train_data: # a hyperparameter - lower values will allow for quicker experiment time
        # Initialize program graph
        if cmd_args.resume_graph is None:
            program_graph = ProgramGraph(DSL_DICT, input_type, output_type, input_size, output_size,
                                        cmd_args.max_num_units, cmd_args.min_num_units, max_depth,
                                        device, ite_beta = cmd_args.ite_beta, cfg = None, var_ids = None, root_symbol = None)
            start_depth = 0
        else:
            assert os.path.isfile(cmd_args.resume_graph)
            program_graph = pickle.load(open(cmd_args.resume_graph, "rb"))
            program_graph.max_depth = max_depth
            start_depth = program_graph.get_current_depth()

        # Initialize algorithm
        algorithm = NAS(frontier_capacity=cmd_args.frontier_capacity)
        partition_num = 0
        best_program_holder = 0
        num_data_missed = 1000 # This is a tracker of how many datapoints are missed (minimum value) by a function seen so far
        # TODO: Idea: num_data_missed could be calculated in a more clever way, by assigning weights to different types of datapoints (for example, implication examples weighted less)
        
        all_graphs = [[0, program_graph]]
        print("Length of training data is ", len(train_data), " and the current training data is ", train_data)
        working_params = []
        while(True):
            _, program_graph = heapq.heappop(all_graphs)
            search_loader = IOExampleLoader(train_data, train_labels, batch_size=batch_size, shuffle=False)
            batched_trainset = search_loader.get_batch_trainset()
            batched_validset = search_loader.get_batch_validset()

            # for program train
            train_loader = IOExampleLoader(train_data, train_labels, batch_size=batch_size, shuffle=False)
            batched_prog_trainset = train_loader.get_batch_trainset()
            prog_validset = train_loader.get_batch_validset()
            testset = train_loader.testset

            log_and_print('data for architecture search')
            log_and_print('batch num of train: {}'.format(len(batched_prog_trainset)))
            log_and_print('batch num of valid: {}'.format(len(prog_validset)))
            log_and_print('total num of test: {}'.format(len(testset)))

            # Run program learning algorithm
            best_graph, program_graph = algorithm.run_specific(program_graph,\
                                        search_loader, train_loader,
                                        train_config, device, start_depth=start_depth, warmup=False, cegis=(partition_num>0), sem=cmd_args.sem)


            best_program = best_graph.extract_program()
            program_graph.show_graph()
            # print program
            log_and_print("Best Program Found:")
            program_str = print_program(best_program)
            log_and_print(program_str)

            # Save best program
            pickle.dump(best_graph, open(os.path.join(save_path, "graph.p"), "wb"))

            # Finetune
            # TODO: Warning - the finetuning here does not work, as the label_correctness function has not been changed to work properly with the data format
            # If you change False to True, you will see that there are several bugs which I have not resolved
            if False and cmd_args.finetune_epoch is not None:
                train_config = {
                    'train_lr' : cmd_args.finetune_lr,
                    'search_epoches' : cmd_args.neural_epochs,
                    'finetune_epoches' : cmd_args.finetune_epoch, # changed from cmd_args.finetune_epochs as this could not be found
                    'model_optim' : optim.Adam,
                    'lossfxn' : lossfxn,
                    'evalfxn' : label_correctness,
                    'num_labels' : num_labels,
                    'save_path' : save_path,
                    'topN' : cmd_args.topN_select,
                    'arch_weight_decay' : 0,
                    'model_weight_decay' : 0,
                    'secorder' : cmd_args.sec_order
                }
                log_and_print('Finetune')
                # start time
                start = time.time()
                best_graph = algorithm.train_graph_model(best_graph, train_loader, train_config, device, lr_decay=1.0)
                # calculate time
                total_spend = time.time() - start
                log_and_print('finetune time spend: {} \n'.format(total_spend))
                # store
                pickle.dump(best_graph, open(os.path.join(save_path, "finetune_graph.p"), "wb"))

                # debug
                testset = train_loader.testset
                best_program = best_graph.extract_program()

            best_program = best_program.submodules["program"]
            print_program2(best_program, env)

            # We now get the possible smoothed parameters
            
            smoothed_params = get_all_params(best_program, Smoothed = True)
        
            # Take the cartesian product
            all_smoothed_param_choices = itertools.product(*smoothed_params) 

            # And we get the nonsmoothed parameters in the correct format
            nonsmoothed_params = get_all_params(best_program)
            all_nonsmoothed_param_choices = itertools.product(*nonsmoothed_params)

            # Want to check the nonsmoothed one first
            is_non_smoothed_correct = False

            for param_choice in all_nonsmoothed_param_choices: # there is just one, but to keep with format
                print("*** NONSMOOTHED CASE ***")
                copied_list_param_choice = [x for x in list(param_choice)] 
                print(copied_list_param_choice)
                func = lambda_program_generator_new(best_program, list(copied_list_param_choice))
                Missed = function_accuracy(func, train_data, train_labels)
                print(len(Missed), " out of ", len(train_data), "examples missed")
                print("The missed are ", Missed, " and the training data is ", train_data, " with labels ", train_labels)
                if len(Missed) == 0:
                    is_non_smoothed_correct = True
            found_solution = False

            print("*** SMOOTHED CASE ATTEMPTS ***")            
            for param_choice in all_smoothed_param_choices:
                # If I build the program in the exact same way I got the parameter order, the structure should be preserved.
                print("## The parameters in this choice are ", list(param_choice)) 
                list_param_choice = list(param_choice) 
                copied_list_param_choice = [x for x in list_param_choice] 
                func = lambda_program_generator_new(best_program, list_param_choice)
                Missed_Smooth = function_accuracy(func, train_data, train_labels)
                print(len(Missed_Smooth), " out of ", len(train_data), "examples missed and they are ", Missed_Smooth)
                if len(Missed_Smooth) < num_data_missed:
                    num_data_missed = len(Missed_Smooth)
                    best_program_holder = best_program 
                    working_params = [x for x in copied_list_param_choice]
                if len(Missed_Smooth) == 0:
                    found_solution = True
                    working_params = copied_list_param_choice
                    break
            if found_solution:
                print("A solution has been found, breaking out of search loop to test it")
                break
            # We also evaluate a version of the nonsmoothed parameters and see if it has a good performance
            elif is_non_smoothed_correct:
                prod = itertools.product(*nonsmoothed_params)
                for param_choice in prod: # there should be one
                    working_params = [[math.floor(el*1000.)/1000. for el in x] for x in list(param_choice)] 
                break
            elif partition_num < top_k: 
                print("!!!! No solution was found yet, and we are under the top k partition count, moving on !!!!")
                # first, check that the correct invariant works on all examples
                # This is more of a sanity check, to ensure we are classifying data points corrected
                if problem_num in cln2inv_invariant_dictionary:
                    assert len(function_accuracy(cln2inv_invariant_dictionary[problem_num], train_data, train_labels)) == 0, "Something wrong with datapoints"
                
                if is_non_smoothed_correct: # This should never run, but just in case
                    assert False, "Improper Smoothing"
                train_loader = IOExampleLoader(train_data, train_labels, batch_size=batch_size, shuffle=False)
                for pair in all_graphs:
                    pair[0] = evaluate(algorithm, pair[1], train_loader, train_config, device)
                splited_subgraph = program_graph.partition(cmd_args.top_left, cmd_args.GM)

                partition_num += 1
                if splited_subgraph is not None:
                    for subgraph in splited_subgraph:
                        metric = evaluate(algorithm, subgraph, train_loader, train_config, device)
                        if metric < 90: # to get around a weird error, from program_graph line ~360
                            # What I think was happening there was that right before the attempt fails, upon expanding all possible 
                            # invariant structures, it tries to run just the invariant structure "StartFunction", which has no meaningful
                            # Returned result, learning to an empty intermediate result return, which canno bt  concatenated
                            all_graphs.append([metric, subgraph])
                heapq.heapify(all_graphs)
            else:  # Now we use the "best" program found thus far, as no solution to all the data has been found yet
                print("Have tried the top_k_programs structures and now using best smoothed invariant found")
                # TODO: Idea: Keep a track of how many times this branch executes, a good heuristic may be to stop searching if this executes > r = 3 times ( to cut down on computational cost )
                # Continuing the search even though we haven't found a working invariant also serves as a way to run the algorithm many times on the same problem - eliminated need for multiple attempts explicitly 
                # All we have to do is break out of the loop and the check commences
                break
            print("number of partitions: ", partition_num)
        
        # This is where we check the invariant            
        working_params_copy = [x for x in working_params]
        func_smoothed = lambda_program_generator_new(best_program_holder, working_params_copy)
        inv_smt = invariant_from_program_new(best_program_holder, working_params, env)
        print("Invariant smt is ", inv_smt)
        result = invariantChecker.check_cln([inv_smt], env)
        print("The result was", result) 
        if result[0]:
            print(result[1]) # this is the invariant string
            return "Solved!", result[1], verification_iter
        elif result[2] == "loop": # we have an implication example, just add it into the training data
            train_data.append(result[3])
            train_labels.append([3.])
        else:
            # I have a sat example from pre or post, but I don't know which category it falls into, so I check its output from the generated function and add it to the opposite label
            output = func_smoothed(*result[3])
            if output: # then we need this datapoint to be false
                train_data.append([[float(result_element) for result_element in result[3]], [-1000. for i in env]])
                train_labels.append([2.])
            else: # then the datapoint needs to be true
                train_data.append([[-1000. for i in env],[float(result_element) for result_element in result[3]]])
                train_labels.append([1.])
        
        verification_iter+=1
    assert False, "Did not solve after {} verification iterations".format(max_train_data)
    return "Did not solve", "", verification_iter


def load_trace(csv_name): # unclear if we want to drop init, final columns are this is important for me to distinguish things
    df = pd.read_csv(csv_name)
    #df_data = df.drop(columns=['init', 'final'])
    df['1'] = 1
    return df

# This is for loading the spreadsheet data from the cln2inv implementation, as an attempt to warm-start the search
# However, I do not believe their spreadsheet data contains any negative examples, so I did not find great results with this
def load_spreadsheet_data_test(problem_num, env):
    csv_name = str(problem_num) + '.csv'
    trace_path = 'benchmarks-cln2inv/code2inv/csv/'
    data = load_trace(trace_path + csv_name)

    training_data = []
    train_labels = []
    for index, row in data.iterrows():
        if row['init'] == 0 and row['final'] == 0: # this is an implication example
            next_row = data.iloc[index+1] 
            training_data.append([[row[var] for var in env], [next_row[var] for var in env]])
            train_labels.append([3.])
        if row['init'] == 0 and row['final'] == 1: # post example
            training_data.append([[-1000. for var in env], [row[var] for var in env]])
            train_labels.append([1.])
        if row['init'] == 1 and row['init'] == 0: # pre example
            training_data.append([[-1000. for var in env], [row[var] for var in env]])
            train_labels.append([1.])
    return training_data, train_labels

def return_sem():
    cmd_args = parse_args()
    return cmd_args.sem


if __name__ == '__main__':
    cmd_args = parse_args()
    neural_epochs = cmd_args.neural_epochs # training epochs for neural programs
    symbolic_epochs = cmd_args.symbolic_epochs # training epochs for symbolic programs
    max_depth = cmd_args.max_depth
    batch_size = cmd_args.batch_size # default 50
    lr = cmd_args.learning_rate # default 0.045
    top_k_programs = cmd_args.top_k_programs # This is the argument controlling how many programs structures to try before checking best found invariant with z3
    random_seed = cmd_args.random_seed # False for no random seed 
    pre_cooked_data = cmd_args.pre_cooked_data # whether or not to use the spreadsheet data from cln2inv implementation
    problem_num = cmd_args.problem_num
    max_train_data = cmd_args.max_train_data
    smoothing_functions_to_try = [1] # choices from ints in [1,2,3,4,5]
    sem = cmd_args.sem

    # Hacky fix to write the semantics so that LibraryFunctions can see what it is, this could be fixxed through nas.py but there are many more interconnected spots to fix there, this was easier

    relative_path = "archsyn/dsl/current_semantics.txt"
    absolute_path = os.path.abspath(relative_path)
    with open(absolute_path, 'w') as f:
        f.write(sem)

    invariant_dict = {}
    # this is for testing the pregenerated spreadsheet data from cln2inv, where additionally we can add new CEs in the process, all we do is change the starting data from none to this.
    if pre_cooked_data: 
        time1 = time.time()
        solved_probs = []
        unsolved_probs = {} # the error messages are entered here
        for p_num in range(1,134):
            unsolved_probs[p_num] = []
        for p_num in [1, 100, 101, 106, 108, 11, 110, 111, 112, 113, 118, 119, 12, 120, 121, 122, 123, 124, 125, 126, 127, 128, 13, 14, 16, 18, 2, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 35, 36, 4, 44, 48, 51, 61, 62, 63, 7, 70, 72, 75, 77, 79, 8, 88, 9, 93, 94]: # these 58 are the ones which there is actually spreadsheet data for
            try: # currently, this just tries the first smoothing function
                solved, inv_string, num_iter = run_on_problem(p_num, cmd_args, neural_epochs, symbolic_epochs, max_depth, batch_size, lr, 1, pre_cooked_data, top_k_programs, random_seed, max_train_data) # set to 1, the leastnuclear option here
                print(solved)
                print(inv_string)
                print(num_iter)
                if solved == "Solved!":
                    num_solved+=1
                    solved_probs.append(p_num)
                    invariant_dict[p_num] = inv_string
                    break
            except Exception as e:
                print("An error occurred: ", str(e)) # an empty message means smoothing error
                unsolved_probs[p_num] += [str(e)]
                print(solved_probs)
        print(num_solved)
        print(invariant_dict)
        print("unsolved probs are ", unsolved_probs)
        time2 = time.time()
        print("Running on all programs took ", time2 - time1, " seconds") 
    # here we run on a specific problem, without pre cooked data
    elif problem_num != 0: # 0 is the default problem number, meant to indicate to run on all problems
        time1 = time.time()
        for i in smoothing_functions_to_try:
            solved, inv_string, num_iter = run_on_problem(problem_num, cmd_args, neural_epochs, symbolic_epochs, max_depth, batch_size, lr, i, False, top_k_programs, random_seed, max_train_data)
            print(solved)
            print(inv_string)
            print(num_iter)
            if solved:
                break
        time2 = time.time()
        print("The program for this particular problem took ", time2 - time1, " seconds")
    # here problem_num must then be 0, which indicates to run the algorithm on all problems
    else:
        time1 = time.time()
        num_solved = 0
        solved_probs = []
        unsolved_probs = {} # here I insert the error messages
        probs_times = {}
        for p_num in range(1,134):
            unsolved_probs[p_num] = []
            probs_times[p_num] = [0]
        for p_num in range(1,134): # running on all programs in Code2Inv benchmark
            if p_num in [16, 26, 27, 31, 32, 61, 62, 72, 75, 106]:
                continue # unsolvable so we skip
            print("Have solved ", num_solved, " / ", p_num - 1)
            print("Now running on problem number ", p_num)

            for i in smoothing_functions_to_try:
                for attempt in [1]: # with the random seed, this line does not make sense, but if you remove the random seed multiple attempts could be useful (or you can find another way to do this by writing results to different files)
                    for depth in [2,3]: # this part does not actually use max_depth, here we wanted to see if the algorithm fails on a search with max depth 2, we try it with max depth 3.
                        time3 = time.time()
                        try:
                            solved, inv_string, num_iter = run_on_problem(p_num, cmd_args, neural_epochs, symbolic_epochs, depth, batch_size, lr, i, False, top_k_programs, random_seed, max_train_data)
                            print(solved)
                            print(inv_string)
                            print(num_iter)
                            if solved == "Solved!":
                                num_solved+=1
                                solved_probs.append(p_num)
                                invariant_dict[p_num] = inv_string
                                probs_times[p_num][0] += time4 - time3
                                break
                        except Exception as e:
                            print("An error occurred: ", str(e)) # an empty message means smoothing error
                            unsolved_probs[p_num] += [str(e)]
                        time4 = time.time()
                        probs_times[p_num][0] += time4 - time3

        print(solved_probs)
        print(num_solved)
        print(invariant_dict)
        print("Problem times are ", probs_times)
        print("unsolved probs are ", unsolved_probs)
        time2 = time.time()
        print("Running on all programs took ", time2 - time1, " seconds") 
