import dsl.library_functions as dsl

# Here is where the DSL for the architectures is defined, there are no rules for list -> atom,
# but there are several choices I have provided for atom -> atom, depending on which functions you would like to use

DSL_DICT = {('list', 'list') : [],
                        ('list', 'atom') : [],
                        #('atom', 'atom') : [dsl.AffineFunc, dsl.LogicAndFunction]}
                        #('atom', 'atom') : [dsl.EqualityFunc, dsl.LogicAndFunction]}
                        #('atom', 'atom') : [dsl.AffineFunc, dsl.EqualityFunc, dsl.LogicAndFunction]}
                        #('atom', 'atom') : [dsl.AffineFunc, dsl.EqualityFunc, dsl.LogicOrFunction]}
                        #('atom', 'atom') : [dsl.AffineFunc, dsl.LogicAndFunction, dsl.LogicOrFunction]}
                        ('atom', 'atom') : [dsl.AffineFunc, dsl.EqualityFunc, dsl.LogicAndFunction, dsl.LogicOrFunction]} 

CUSTOM_EDGE_COSTS = {
    ('list', 'list') : {},
    ('list', 'atom') : {},
    ('atom', 'atom') : {}
}

