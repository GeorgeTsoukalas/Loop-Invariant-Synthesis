# A differentiable program synthesizer for SyGuS

## Introduction

Here lies code adapted from DifferentiableSygus for loop invariant synthesis.

## Requirements
- Python 3.6+
- PyTorch 1.4.0+
- scikit-learn 0.22.1+
- Numpy
- tqdm
- Z3

python3 archsyn/new_main.py --batch_size 50 --random_seed 0 -top_left True
-GM True --top_k_programs 4 --max_depth 3 --neural_epochs 50
--symbolic_epochs 50 --max_train_data 20 --sem luka --problem_num 70
```
