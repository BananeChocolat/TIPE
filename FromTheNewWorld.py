import numpy as np
import random as r
from pprint import pprint

green = "\u001b[32m"
white = "\u001b[37m"

# === Variables globales ===
default_n = 2
out_size = 15

# === Script ===

def saver(chunk, filename="test_file") -> None :
    """ Sauvegarde un chunk dans un fichier .vox"""
    save = Entity().from_dense(chunk)
    save.set_palette_from_file('palette.png')
    save.save(f'./vox/{filename}.vox')
    print(f"[{green}INFO{white}] Chunk saved as {filename}.vox")


class Cell:
    """Classe qui définit un cube de n*n de voxels"""
    def __init__(self, obj, model_length:int, n:int=default_n):
        self.obj = np.zeros(shape=(n,n,n), dtype=int)
        self.dim = n
        self.possible = np.full(fill_value="", dtype=str, shape=(model_length))
        self.collapsed = False

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

    # TODO
    def collapse():
        """ WORK IN PROGRESS - Réduit la cellule de plus faible entropie à un état"""
        tensor_size = cell_tensor.shape
        entropy_map = np.zeros(shape=tensor_size, dtype=int)
        for x in range(tensor_size[0]):
            for y in range(tensor_size[1]):
                for z in range(tensor_size[2]):
                    entropy_map[x][y][z] = len(cell_tensor[x][y][z].possible)
        entropy_map[15][14][13] = 0
        min_entropy_index = np.unravel_index(np.argmin(entropy_map), tensor_size)
        print(f'{min_entropy_index = }')


        # test
        # print(f'{r.choice(list(filter(lambda x: x!="",cell_tensor[min_entropy_index].possible))) = }')


    collapse()

    # pprint(cell_tensor)





# create model for a random int tensor of shape 10*10*10
# pprint(create_model(np.random.randint(0, 2, size=(10,10,10))))


generate_struct(create_model(np.random.randint(0, 2, size=(10,10,10))))


test = np.zeros((2,2,2), int)
test[0][0] = 1

test1 = list(filter(lambda x: x!=1, test))


print(test)