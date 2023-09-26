from voxypy.models import Entity, Voxel
import numpy as np

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


# print(entity.get(1, 1, 1))

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

def from_id(id:str) :
    """ Renvoie un array 3d correspondant à l'id avec 0 pour vide et 1 pour plein """
    n = int(len(id)**(1/3))
    dense = np.zeros(shape=(n,n,n),dtype=int)
    for (i,c) in enumerate(id) :
        if (c =='1'):
            dense[i%n**2,i%n,i] = 1
    print(dense)





def get_structures(chunk:Entity) -> dict :
    """ Detecte les differentes structures presentes
        dans un chunk et les retourne dans un dictionnaire.
        Chaque structure est un Entity"""


def create_model(chunk:Entity, n:int=2) -> dict :
    """ Cree un modele a partir d'un chunk"""
    model = {}
    voxels = chunk.get_dense()
    for x in range(0, len(voxels), n):
        for y in range(0, len(voxels[x]), n):
            for z in range(0, len(voxels[x][y]), n):
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

    return model
    


def generate_struct(model:dict) -> Entity :
    """ Genere une structure a partir d'un modele
        en utilisant une implementation de la WFC"""
    dense = np.array()

def place_struct(chunk:Entity, struct:Entity, x:int, y:int, z:int) -> Entity :
    """ Place une structure dans un chunk a une position donnee"""

def saver(chunk, filename="test_file") -> None :
    """ Sauvegarde un chunk dans un fichier .vox"""
    save = Entity().from_dense(chunk)
    save.set_palette_from_file('palette.png')
    save.save('./vox/test_file1.vox')
    print(f"[{green}INFO{white}] Chunk saved as {filename}.vox")


class Unit:
    """Classe pour un cube de NxN de voxels"""
    def __init__(self, obj):
        self.obj = np.array(obj)
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


print(from_id('11101000'))
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
