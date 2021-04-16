# Chess AI
## Install
```pip install chess keras tensorflow numpy cython typing```

Make sure C++ libraries are up to date as well.

Then run
```python setup.py build_ext --inplace```

`abustibia.py` contains the code for our class
## Features
* Minimax Algorithm
* Alpha-Beta Pruning
* CNN for evaluation function (works well but pretty slow, we noticed our algorithm performs really well at depth > 3, but exceeds 60 second time limit)
* Move Orderin based on Capture Value
* Cython implementation
* Opening Book Usage

## File Functions
### Folders
* `code1` folder includes the code for our chess AI
* `data` folder includes data for CNN evaluation function
* `storage` folder is where we store our CNN
### Files
* `code1/abustibia.pyx` contains the main code for our chess player class
* `code1/abustibia2.pyx` contains the main code for the chess player class that uses Sree's evaluation function
* `code1/heuristics.pyx` contains the code for move ordering based on capture value
* `code1/inputs.zip` contains the numpy array for the input to our CNN
* `code1/outputs.zip` contains the numpy array for the output to our CNN
* `code1/setup.py` sets up the Cython implementation
* `code1/train_eval.ipynb` contains the code for our CNN model (training, architecture, data processing etc.) for our evaluation function

