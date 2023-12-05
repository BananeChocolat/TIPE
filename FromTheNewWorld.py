import numpy as np
default_n = 2
from pprint import pprint



class Cell:
    """Classe qui définit un cube de n*n de voxels"""
    def __init__(self, obj, model_length:int, n:int=default_n):
        self.obj = np.array(obj)
        self.dim = n
        self.possible = np.array(dtype=string, shape=(model_length))
    
    def __str__(self):
        return self.possible


def identifier(obj:np.array) -> str:
    """Identifie un cube de n*n de voxels"""
    return ''.join(obj.flatten().astype(str))

def create_model(dense:np.array, n:int=default_n) -> dict:
    model = {}
    ##TODO: make it work

def generate_struct(model:Model, size:int=16, n:int=default_n):
    """Génère une structure à partir d'un modèle"""
    default_cell = Cell(np.zeros(shape=(n,n,n), dtype=int), len(model), n)
    default_cell.possible = np.array(list(model.keys()))
    cell_tensor = np.array(default_cell, dtype=Cell, shape=(size, size, size))
    pprint(cell_tensor)


# create model for a random int tensor of shape 10*10*10
pprint(create_model(np.random.randint(0, 2, size=(10,10,10))))