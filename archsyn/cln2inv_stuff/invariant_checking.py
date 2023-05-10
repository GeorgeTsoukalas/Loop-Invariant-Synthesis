import z3
import random

# This checker is mostly from the cln2inv implementation, but with some modifications I made

class InvariantChecker():
    
    def __init__(self, c_file, smt_check_path):
        self.c_file = c_file
        
        # read top pre rec post checks in
        with open(smt_check_path+'/'+c_file+'.smt.1') as f:
            self.top = f.read()
            
        with open(smt_check_path+'/'+c_file+'.smt.2') as f:
            self.pre = f.read()
            
        with open(smt_check_path+'/'+c_file+'.smt.3') as f:
            self.loop = f.read()
            
        with open(smt_check_path+'/'+c_file+'.smt.4') as f:
            self.post = f.read()
            
        # self.solver = z3.Solver()
        # self.solver.set("timeout", 2000)
        
    def check(self, inv_str, env):
        for ind in range(3):
            check = [self.pre, self.post, self.loop][ind] # The order matters (a changed order could introduce more implication examples)
            full_check = self.top + inv_str + check
            solver = z3.Solver()
            solver.set("timeout", 2000)
            solver.from_string(full_check)
            res = solver.check()
            current_part = ""
            if ind == 0:
                current_part = "pre"
            elif ind == 1:
                current_part = "post"
            else:
                current_part = "loop"
            print("The result is ", res, " during the ", current_part)

               
            if res != z3.unsat and current_part == "loop": 
                print("Not a valid invariant: here is a counterexample")
                model = solver.model()
                print(model) # printing model for extra information, also the model is often quite large so it makes a noticeable chunk of the output in the cmd prompt, for visibility
                z3_env = list(map(lambda x: z3.Int(x), env)) # instantiating variables for pre-loop
                z3_env_new = list(map(lambda x: z3.Int(x), map(lambda x: x + '!', env))) # instantiating variables for after-loop
                returned_assignment = []
                returned_assignment_new = []
                for var in z3_env:
                    if model[var] != None:
                        returned_assignment.append(model[var].as_long())
                    else:
                        # If the model does not need some variable, we assign it a random value in the returned datapoint
                        # It might be best to choose a value in the distribution of the previous data points, but I just choose a random one 0 - 10 here
                        random_choice = random.randint(0, 10)
                        returned_assignment.append(random_choice)     
                for var in z3_env_new: # these are for the values are the loop body executes
                    if model[var] != None:
                        returned_assignment_new.append(model[var].as_long())
                    else:
                        random_choice = random.randint(0, 10)
                        returned_assignment_new.append(random_choice)             
                return False, current_part, [returned_assignment, returned_assignment_new] 
            elif res != z3.unsat: # pre or post, only return the single point
                print("Not a valid invariant: here is a counterexample")
                model = solver.model()
                z3_env = list(map(lambda x: z3.Int(x), env))
                print(model)
                returned_assignment = []
                for var in z3_env:
                    if model[var] != None:
                        returned_assignment.append(model[var].as_long())
                    else:
                        random_choice = random.randint(0, 10)
                        returned_assignment.append(random_choice)
                return False, current_part, returned_assignment
        return True, "" # has worked (get z3.unsat)

    def check_cln(self, inv_smts, env):
        correct = False
        vals = []
        for inv_smt in inv_smts:
            inv_str = inv_smt.sexpr()
            inv_str = inv_str.replace('|', '')
            correct = self.check(inv_str, env)
            if correct[0]:
                return True, inv_str, []
            else:
                vals = correct[2]
        return False, '', correct[1], vals # correct[1] is the part of the verification that failed

            
