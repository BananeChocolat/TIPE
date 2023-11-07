from voxypy.models import Entity, Voxel
from pprint import pprint
import numpy as np
import random as r
from progress.bar import Bar

green = "\u001b[32m"
white = "\u001b[37m"

print("\n\n\n => Script exec :")

# créé un chunk de 10x10x10 rempli de 0
dense = np.zeros((10, 10, 10), dtype=int)

# transforme le chunk en Entity de la librairie voxypy
entity = Entity(data=dense)

# modifie le voxel aux coordonnées (1, 2, 3) avec la valeur 42
entity.set(x=1, y=2, z=3, color=42)

# récupère le voxel aux coordonnées (1, 2, 3)
voxel = entity.get(1, 2, 3)


# voxel.add(1) # Voxel object with value 43
# new_voxel = Voxel(255)
# new_voxel.add(1) # Returns Voxel object with value 1
# entity.set(5, 5, 5, new_voxel)
# entity.set(5, 5, 5, 69)

entity = Entity().from_file('./vox/test_file.vox')
# print(entity.get_dense())
# [[[0 0]
#   [1 0]]
#  [[2 0]
#   [4 3]]]

# def get_structure(entity) :
#     """detects the structures inside a chunk and returns a list of those structures
#         split each struture based on color"""
#     chunk = entity.get_dense()
#     structures = {}
#     for x in range(len(chunk)):
#         for y in range(len(chunk[x])):
#             for z in range(8, len(chunk[x][y])):
#                 color = chunk[x][y][z]
#                 if (color != 0):
#                     if color not in structures:
#                         structures[color] = []
#                     structures[color].append((x, y, z))
#     return structures

# struct = get_structure(entity)
# print(struct)


def identifier(chunk:list[list[list[int]]]) -> str :
    """ Identifie un chunk en lui attribuant un id unique"""
    id = ""
    for x in range(len(chunk)):
        for y in range(len(chunk[x])):
            for z in range(len(chunk[x][y])):
                id += str("1" if chunk[x][y][z] else "0")
    return id if id else None

def from_id(id:str, n:int) :
    """ Renvoie un array 3d correspondant à l'id avec 0 pour vide et 1 pour plein """
    dense = np.zeros(shape=(n,n,n),dtype=int)
    for (i,c) in enumerate(id) :
        if (c =='1'):
            dense[(i//n**2) %n][(i//n) %n][i%n] = 1
    return dense



def get_structures(chunk:Entity) -> dict :
    """ Detecte les differentes structures presentes
        dans un chunk et les retourne dans un dictionnaire.
        Chaque structure est un Entity"""


def create_model(chunk:Entity, n:int) -> dict :
    """ Cree un modele a partir d'un chunk"""
    print(f"[{green}INFO{white}] creating model...")
    model = {}
    voxels = np.array(chunk.get_dense())
    print(f"{len(voxels) = }")
    print(voxels.shape)
    if (voxels.shape[0] % n != 0) :
        voxels.resize((voxels.shape[0] + n - voxels.shape[0] % n, voxels.shape[1], voxels.shape[2]))
    print(f"{len(voxels) = }")
    creation_bar = Bar('Model creation', max=len(voxels)*len(voxels[0])*len(voxels[0][0])/n**3)
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
                creation_bar.next()
    creation_bar.finish()
    print(f"{len(model) = }")
    return model



def generate_struct(model:dict, n) -> np.ndarray :
    """ Genere une structure a partir d'un modele
        en utilisant une implementation de la WFC"""
    print(f"[{green}INFO{white}] generating structure...")
    out_size = 32
    dense = np.zeros(shape=(out_size+n,out_size+n,out_size+n),dtype=int)
    block = from_id(list(model.keys())[0], n)

    def sub_id(x,y,z) :
        return identifier(dense[x*n:x*n+n,y*n:y*n+n,z*n:z*n+n])
        # imagine oublier les "*n", nan je rigole... mais imagine quand meme

    def place_sub(x,y,z,id) :
        if int(id) == 0 : return
        else :
            # print(x,y,z,id)
            a = from_id(id,n)
            a.resize(n,n,n)
            # print(identifier(a))
            dense[x*n:x*n+n,y*n:y*n+n,z*n:z*n+n] = a

    place_sub(0,0,0,identifier(block))

    for z in range(0,out_size//n) :
        for y in range(0,out_size//n) :
            for x in range(0,out_size//n) :
                if ((id := sub_id(x,y,z)) in model) :
                    if model[id].xp : place_sub(x+1,y,z,r.choice(model[id].xp))
                    if model[id].yp : place_sub(x,y+1,z,r.choice(model[id].yp))
                    if model[id].zp : place_sub(x,y,z+1,r.choice(model[id].zp))

    return dense

    # print(dense[:5,:5,:5])




def place_struct(chunk:Entity, struct:Entity, x:int, y:int, z:int) -> Entity :
    """ Place une structure dans un chunk a une position donnee"""

def saver(chunk, filename="test_file") -> None :
    """ Sauvegarde un chunk dans un fichier .vox"""
    save = Entity().from_dense(chunk)
    save.set_palette_from_file('palette.png')
    save.save(f'./vox/{filename}.vox')
    print(f"[{green}INFO{white}] Chunk saved as {filename}.vox")


class Unit:
    """Classe pour un cube de NxN de voxels"""
    def __init__(self, obj, n=2):
        self.obj = np.array(obj)
        self.size = n
        self.xp = []
        self.xm = []
        self.yp = []    
        self.ym = []
        self.zp = []
        self.zm = []

    def __repr__(self):
        return str(self.xp) + str(self.xm) + str(self.yp) + str(self.ym) + str(self.zp) + str(self.zm)




a = Unit([[[4,3],[2,0]],[[1,0],[0,0]]])
b = Unit([[[(i+j+k)%2 for k in range(4)] for j in range(4)] for i in range(4)])
c = Unit([[[(i+j+k)%3 for k in range(4)] for j in range(4)] for i in range(4)])


div = 4
test_entity = Entity().from_file('./vox/castle.vox')
test_model = create_model(test_entity, div)
# pprint(test_model)
test_gen = generate_struct(test_model, div)
saver(test_gen, "castle_gen")

# print(create_model(Entity().from_file('./vox/menger.vox'),3))

# large = Entity().from_file('./TIPE.vox')
# build = np.array(large.get_dense())[6:14,6:14,8:27]
# print(create_model(Entity().from_dense(build),2))


# d = np.concatenate((b.obj,c.obj),axis=3)

# saver(a.obj, "test_file1")

# print(create_model(Entity().from_dense(a.obj), 1))
# print(create_model(Entity().from_dense(b.obj), 2))
# print(b.obj)
# print(identifier(a.obj))
# print(b.obj[0:2,0:2,0:2])
