run_my_solution = False
assignmentNumber = '5'

import os
import copy
import signal
import os
import numpy as np

if run_my_solution:
    from A5mysolution import *
    print('##############################################')
    print("RUNNING INSTRUCTOR's SOLUTION!!!!!!!!!!!!")
    print('##############################################')

else:
    
    print('\n======================= Code Execution =======================\n')


    import subprocess, glob, pathlib
    n = assignmentNumber
    nb_names = glob.glob(f'*-A{n}-[0-9].ipynb') + glob.glob(f'*-A{n}.ipynb')
    nb_names = np.unique(nb_names)
    nb_names = sorted(nb_names, key=os.path.getmtime)
    if len(nb_names) > 1:
        print(f'More than one ipynb file found: {nb_names}. Using first one.')
    filename = nb_names[0]
    print('Extracting python code from notebook named \'{}\' and storing in notebookcode.py'.format(filename))
    if not filename:
        raise Exception('Please rename your notebook file to <Your Last Name>-A{}.ipynb'.format(assignmentNumber))
    with open('notebookcode.py', 'w') as outputFile:
        subprocess.call(['jupyter', 'nbconvert', '--to', 'script',
                         filename, '--stdout'], stdout=outputFile)
    # from https://stackoverflow.com/questions/30133278/import-only-functions-from-a-python-file
    import sys
    import ast
    import types
    with open('notebookcode.py') as fp:
        tree = ast.parse(fp.read(), 'eval')
    print('Removing all statements that are not function or class defs or import statements.')
    for node in tree.body[:]:
        if (not isinstance(node, ast.FunctionDef) and
            not isinstance(node, ast.Import) and
            not isinstance(node, ast.ImportFrom) and
            not isinstance(node, ast.ClassDef)):
            tree.body.remove(node)
    # Now write remaining code to py file and import it
    module = types.ModuleType('notebookcodeStripped')
    code = compile(tree, 'notebookcodeStripped.py', 'exec')
    sys.modules['notebookcodeStripped'] = module
    exec(code, module.__dict__)
    # import notebookcodeStripped as useThisCode
    from notebookcodeStripped import *

required_funcs = ['NeuralNetworkTorch', 'NeuralNetworkClassifierTorch', 'NeuralNetworkClassifierConvolutionalTorch',
                      'rmse', 'percent_correct', 'partition', 'confusion_matrix',
                      'multiple_runs_regression', 'multiple_runs_classifiation',
                      'multiple_runs_convolutional', 'train_for_best_validation']

for func in required_funcs:
    if func not in dir() or not callable(globals()[func]):
        print('CRITICAL ERROR: Function named \'{}\' is not defined'.format(func))
        print('  Check the spelling and capitalization of the function name.')

exec_grade = 0


#### constructor ##################################################################

print('''
## Testing constructor ####################################################################

    nnet = NeuralNetworkClassifierConvolutionalTorch([1, 10, 10], 
                                                     [(2, (3, 3), (1, 1)), (3, (5, 5), (2, 2))],
                                                     [30, 20], 2)

    # Is isinstance(nnet, NeuralNetworkClassifierTorch)   True?
''')
      
try:
    pts = 10
    nnet = NeuralNetworkClassifierConvolutionalTorch([1, 10, 10], 
                                                     [(2, (3, 3), (1, 1)), (3, (5, 5), (2, 2))],
                                                     [30, 20], 2)
    if isinstance(nnet, NeuralNetworkClassifierTorch):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. NeuralNetworkClassifierConvolutionalTorch is correctly of type NeuralNetworkClassifierTorch')
    else:
        print(f'\n---  0/{pts} points. NeuralNetworkClassifierConvolutionalTorch should be of type NeuralNetworkClassifierTorch but it is not.')
        print(f'                  Correct values are {correct}')

    if len(nnet.layers) == 5:
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. nnet correctly has 5 pytorch layers.')
    else:
        print(f'\n---  0/{pts} points. nnet has {len(nnet.layers)} pytorch layers, but it should have 5.')

except Exception as ex:
    print(f'\n--- 0/20 points. NeuralNetworkClassifierConvolutionalTorch raised the exception\n')
    print(ex)



#### train  ##################################################################
print('''
## Testing train ####################################################################

    np.random.seed(42)
    X = np.random.uniform(0, 1, size=(100, 2))
    T = (np.abs(X[:, 0:1] - X[:, 1:2]) < 0.5).astype(int)
    results = multiple_runs_classification(5, X, T, (0.4, 0.3, 0.3), [10], 100, 0.01)
''')

try:
    pts = 10
    np.random.seed(42)
    X = np.random.uniform(0, 1, size=(100, 2))
    T = (np.abs(X[:, 0:1] - X[:, 1:2]) < 0.5).astype(int)
    results = multiple_runs_classification(5, X, T, (0.4, 0.3, 0.3), [10], 100, 0.01)
    
    if len(results) == 15:
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. results correctly has 15 elements')
    else:
        print(f'\n---  0/{pts} points. results is length {len(results)} but it should be 15.')

    if len(results[0]) == 3:
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. results[0] correctly has 3 elements')
    else:
        print(f'\n---  0/{pts} points. results[0] is length {len(results[0])} but it should be 3.')

    mx = max([r[2] for r in results])
    if mx == 100:
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. max of results accuracy is correctly 100%.')
    else:
        print(f'\n---  0/{pts} points. max of results is {mx} but should be 100%.')
        
        
except Exception as ex:
    print(f'\n--- 0/30 points. multiple_runs_classification raised the exception\n')
    print(ex)
    




#### train  ##################################################################
print('''
## Testing train ####################################################################

    np.random.seed(42)
    X = np.random.uniform(0, 1, size=(100, 1, 10, 10))
    T = (X[:, 0, :, 0].mean(-1) < X[:, 0, :, -1].mean(-1)).astype(int).reshape(-1, 1)
    results = multiple_runs_convolutional(10, X, T, (0.4, 0.3, 0.3), [[10, (4, 4), (1, 1)]], [10], 1000, 0.01)
    mean_train = np.mean([r[2] for r in results if r[1] == 'train'])
''')

try:
    pts = 20
    np.random.seed(42)
    X = np.random.uniform(0, 1, size=(100, 1, 10, 10))
    T = (X[:, 0, :, 0].mean(-1) < X[:, 0, :, -1].mean(-1)).astype(int).reshape(-1, 1)
    results = multiple_runs_convolutional(10, X, T, (0.4, 0.3, 0.3), [[10, (4, 4), (1, 1)]], [10], 1000, 0.01)
    mean_train = np.mean([r[2] for r in results if r[1] == 'train'])
    
    if mean_train > 80:
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Mean of train accuracy is correct greater than 80%.')
    else:
        print(f'\n---  0/{pts} points. Mean of train accuracy is not greater than 80% but it should be.')
        
except Exception as ex:
    print(f'\n--- 0/20 points. multiple_runs_convolutional raised the exception\n')
    print(ex)
    


name = os.getcwd().split('/')[-1]

print()
print('='*70)
print(f'{name} Execution Grade is {exec_grade} / 70')
print('='*70)

print('''
__ / 5 points. Violin plots for each of the three datasets.

__ / 10 points. Confusion matrices for the two classification problems.
                to MNIST data.

__ / 15 points. Total of at least 30 sentences discussion results of all three datasets.
''')

print()
print('='*70)
print(f'{name} Results and Discussion Grade is ___ / 30')
print('='*70)


print()
print('='*70)
print(f'{name} FINAL GRADE is  _  / 100')
print('='*70)


print('''Extra Credit: Earn one point of extra credit for adding code to one of your NeuralNetwork...Torch classes and training and using it on a GPU for one data set.      ''')


print(f'\n{name} EXTRA CREDIT is 0 / 1')


if run_my_solution:
    print('##############################################')
    print("RUNNING INSTRUCTOR's SOLUTION!!!!!!!!!!!!")
    print('##############################################')

    
