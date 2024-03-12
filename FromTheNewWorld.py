import numpy as np
from pprint import pprint

green = "\u001b[32m"
white = "\u001b[37m"

# === Variables globales ===
default_n = 2
out_size = 15

# === Script ===

class Cell:
    """Classe qui définit un cube de n*n de voxels"""
    def __init__(self, obj, model_length:int, n:int=default_n):
        self.obj = np.zeros(shape=(n,n,n), dtype=int)
        self.dim = n
        self.possible = np.full(fill_value="", dtype=str, shape=(model_length))
    
    def __str__(self):
        return self.possible

class Unit:
    """Classe pour un cube de NxN de voxels"""
    def __init__(self, obj, n=2):
        self.obj = np.array(obj)
        self.size = n
        self.xp, self.xm, self.yp, self.ym, self.zp, self.zm = [], [], [], [], [], []

    def __repr__(self):
        return str(self.xp) + str(self.xm) + str(self.yp) + str(self.ym) + str(self.zp) + str(self.zm)


def identifier(obj:np.array) -> str:
    """Identifie un cube de n*n de voxels"""
    return ''.join(obj.flatten().astype(str))

def create_model(voxels:np.array, n:int=default_n) -> dict :
    """ Cree un modele a partir d'un dense"""
    print(f"[{green}INFO{white}] creating model...")
    model = {}
    print(f"{voxels.shape = }")
    if (voxels.shape[0] % n != 0) :
        voxels.resize((voxels.shape[0] + n - voxels.shape[0] % n, voxels.shape[1] + n - voxels.shape[1] % n, voxels.shape[2] + n - voxels.shape[2] % n))
    print(f"{voxels.shape = }")
    for x in range(0, len(voxels), n):
        for y in range(0, len(voxels[0]), n):
            for z in range(0, len(voxels[0][0]), n):
                unit = Unit(voxels[x:x+n,y:y+n,z:z+n])
                id = identifier(unit.obj)
                if id not in model:
                    model[id] = unit
                if (id_adj := identifier(voxels[x+n:x+2*n,y:y+n,z:z+n])) not in model[id].xp and id_adj is not None : model[id].xp.append(id_adj)
                if (id_adj := identifier(voxels[max(0,x-n):x,y:y+n,z:z+n])) not in model[id].xm and id_adj is not None : model[id].xm.append(id_adj)
                if (id_adj := identifier(voxels[x:x+n,y+n:y+2*n,z:z+n])) not in model[id].yp and id_adj is not None : model[id].yp.append(id_adj)
                if (id_adj := identifier(voxels[x:x+n,max(0,y-n):y,z:z+n])) not in model[id].ym and id_adj is not None : model[id].ym.append(id_adj)
                if (id_adj := identifier(voxels[x:x+n,y:y+n,z+n:z+2*n])) not in model[id].zp and id_adj is not None : model[id].zp.append(id_adj)
                if (id_adj := identifier(voxels[x:x+n,y:y+n,max(0,z-n):z])) not in model[id].zm and id_adj is not None : model[id].zm.append(id_adj)
    print(f"{len(model) = }")
    return model

def generate_struct(model:dict, size:int=16, n:int=default_n):
    """Génère une structure à partir d'un modèle"""
    default_cell = Cell(np.zeros(shape=(n,n,n), dtype=int), len(model), n)
    default_cell.possible = np.array(list(model.keys()))
    cell_tensor = np.full(fill_value=default_cell, dtype=Cell, shape=(size, size, size))
    pprint(cell_tensor)



# create model for a random int tensor of shape 10*10*10
# pprint(create_model(np.random.randint(0, 2, size=(10,10,10))))


generate_struct(create_model(np.random.randint(0, 2, size=(10,10,10))))