import numpy as np
import random as r
from pprint import pprint
from voxypy.models import Entity

green = "\u001b[32m"
white = "\u001b[37m"

# === Variables globales ===
default_n = 2
out_size = 15
upper_bound = 8192

# === Script ===

def saver(chunk, filename="test_file") -> None :
    """ Sauvegarde un chunk dans un fichier .vox"""
    save = Entity().from_dense(chunk)
    save.set_palette_from_file('palette.png')
    save.save(f'./vox/{filename}.vox')
    print(f"[{green}INFO{white}] Chunk saved as {filename}.vox")

def from_id(id:str, n:int) :
    """ Renvoie un array 3d correspondant à l'id avec 0 pour vide et 1 pour plein """
    dense = np.zeros(shape=(n,n,n),dtype=int)
    for (i,c) in enumerate(id) :
        if (c =='1'):
            dense[(i//n**2) %n][(i//n) %n][i%n] = 1
    return dense

class Cell:
    """Classe qui définit un cube de n*n de voxels"""
    def __init__(self, obj, model_length:int, n:int=default_n):
        self.obj = np.zeros(shape=(n,n,n), dtype=int)
        self.dim = n
        self.possible = np.full(fill_value="", dtype=str, shape=(model_length))
        self.collapsed = False

    def __str__(self):
        return str(self.possible)

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

def generate_struct(model:dict, size:int=16, n:int=default_n) -> np.ndarray:
    """Génère une structure à partir d'un modèle"""
    default_cell = Cell(np.zeros(shape=(n,n,n), dtype=int), len(model), n)
    default_cell.possible = np.array(list(model.keys()))
    cell_tensor = np.full(fill_value=default_cell, dtype=Cell, shape=(size, size, size))
    for x in range(size):
        for y in range(size):
            for z in range(size):
                cell_tensor[x,y,z] = Cell(np.zeros(shape=(n,n,n), dtype=int), len(model), n)
                cell_tensor[x,y,z].possible = np.array(list(model.keys()))

    def collapse():
        """ WORK IN PROGRESS - Réduit la cellule de plus faible entropie à un état"""
        tensor_size = cell_tensor.shape
        entropy_map = np.zeros(shape=tensor_size, dtype=int)
        for x in range(tensor_size[0]):
            for y in range(tensor_size[1]):
                for z in range(tensor_size[2]):
                    active_cell = cell_tensor[x,y,z]
                    entropy_map[x,y,z] = len(active_cell.possible) if (not active_cell.collapsed) else upper_bound

        # min_entropy_index = np.unravel_index(np.argmin(entropy_map), tensor_size)
        min_entropy_index = np.unravel_index(np.random.choice(np.flatnonzero(entropy_map == entropy_map.min())) , tensor_size)

        try:
            collapsed_obj_str = r.choice(list(filter(lambda x: x!="",cell_tensor[min_entropy_index].possible)))
        except:
            collapsed_obj_str = '0'*n*n*n
        cell_tensor[min_entropy_index].possible = np.array([collapsed_obj_str])
        cell_tensor[min_entropy_index].obj =from_id(collapsed_obj_str, n)
        cell_tensor[min_entropy_index].collapsed = True

        # print(f'Just collapsed cell {min_entropy_index}')

        propagate(min_entropy_index)

        # print(f'Just propagated around cell {min_entropy_index}')

    def propagate(cell_index:(int,int,int)):
        (x,y,z) = cell_index
        cell_tensor[min(x+1,size-1),y,z].possible = np.intersect1d(cell_tensor[min(x+1,size-1),y,z].possible, model[identifier(cell_tensor[x,y,z].obj)].xp)
        cell_tensor[max(x-1,0),y,z].possible = np.intersect1d(cell_tensor[max(x-1,0),y,z].possible, model[identifier(cell_tensor[x,y,z].obj)].xm)
        cell_tensor[x,min(y+1,size-1),z].possible = np.intersect1d(cell_tensor[x,min(y+1,size-1),z].possible, model[identifier(cell_tensor[x,y,z].obj)].yp)
        cell_tensor[x,max(y-1,0),z].possible = np.intersect1d(cell_tensor[x,max(y-1,0),z].possible, model[identifier(cell_tensor[x,y,z].obj)].ym)
        cell_tensor[x,y,min(z+1,size-1)].possible = np.intersect1d(cell_tensor[x,y,min(z+1,size-1)].possible, model[identifier(cell_tensor[x,y,z].obj)].zp)
        cell_tensor[x,y,max(z-1,0)].possible = np.intersect1d(cell_tensor[x,y,max(z-1,0)].possible, model[identifier(cell_tensor[x,y,z].obj)].zm)

    def build_dense():
        print(f'{cell_tensor.shape = }')
        print(f'{cell_tensor.shape*n = }')
        dense = np.zeros(shape=tuple(i * n for i in cell_tensor.shape), dtype=int)

        for x in range(size):
            for y in range(size):
                for z in range(size):
                    if int(identifier(cell_tensor[x,y,z].obj)) == 0: pass
                    else:
                        dense[x*n:x*n+n,y*n:y*n+n,z*n:z*n+n] = cell_tensor[x,y,z].obj
        
        return dense



    for i in range(size*size*size):
        collapse()

    return build_dense()

    # pprint(cell_tensor)




# create model for a random int tensor of shape 10*10*10
# pprint(create_model(np.random.randint(0, 2, size=(10,10,10))))


dense = generate_struct(create_model(np.array(Entity().from_file('./vox/castle.vox').get_dense())))

def saver(chunk, filename="test_file") -> None :
    """ Sauvegarde un chunk dans un fichier .vox"""
    save = Entity().from_dense(chunk)
    save.set_palette_from_file('palette.png')
    save.save(f'./vox/{filename}.vox')
    print(f"[{green}INFO{white}] Chunk saved as {filename}.vox")

saver(dense, "TEST")

