from tqdm import tqdm
import sys, os

CC_path = os.path.join(os.path.dirname(sys.argv[0]),"CloudCompare")
if CC_path not in sys.path:
    sys.path.append(CC_path)

import CloudCompare.cloudComPy as cc

cc.initCC()

def merge_models(model_list, offset_list, output_path):
    mesh_list = []
    for i in tqdm(range(len(model_list))):
        model = model_list[i]
        model_offset = offset_list[i]
        mesh = cc.loadMesh(model,mode=cc.CC_SHIFT_MODE.XYZ, x = model_offset[0], y = model_offset[1], z = model_offset[2])
        mesh_list.append(mesh)
    combined = cc.MergeEntities(mesh_list, True)
    ret = cc.SaveMesh(combined, output_path)




