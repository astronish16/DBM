# P-DBM : Probabilistic Drag Based Model

This repository consist of python class to predict Coronal Mass Ejection (CME) arrival time in the heliosphere using Drag Based Model (DBM) and Probabilistic Drag based Model (P-DBM).

## Quick Start:

For better project management, we have used [UV](https://docs.astral.sh/uv/). We recommend to have UV installed on your system.

### Installation
1. Clone the repository:
```
git clone https://github.com/astronish16/DBM.git
cd DBM
```
2. Install dependencies:
Required dependencies are listed in [pyproject.toml](pyproject.toml). Following command will create virtual enviroment for this project inside the directory and install all the dependecies.
```
uv sync
```
Alternatively, you can run any `.py` file with `uv run <file_name.py>` command.\
After this, you are ready to perform (P-)DBM calculations.

### Basic Usage
We have created two main python classes `DBM1D` and `DBM2D` to perform all the necessary tasks such as input validation, simulation, plotting and few more. A succeful execution of these classes provides results of (P-)DBM simulation as dictionary.\
The detailed demonstartion of these classes is shown in [example.ipynb](example.ipynb).\
For ondeamnd quick calculations, use [main.py](main.py) file: just change the input with your desiered values and you are good to go.



## Project structure

```
DBM/
├── calculation_1D.py    # DBM1D class
├── calculation_2D.py    # DBM2D class
├── dbm_functions.py     # Core functions for (P-)DBM
├── example.ipynb        # Jupyter notebook showing usage of the python DBM class
├── LICENSE              # Licence file for project
├── main.py              # .py file to perform the (P-)DBM simulation on demand
├── Old_codes/           # Codes from very inital version of the repository
├── pyproject.toml       # Project dependecies
├── README.md

```

## Contact
We are still updating the code base in this repository (mostly optimization and better functionalities). In case you find a bug or if you have any questions, please conatct me [ronish@roma2.infn.it]. I would also appreciate if you tell me the use of the codes in your work.

