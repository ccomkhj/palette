import os
import glob
from PIL import Image
import numpy as np
from loguru import logger
import argparse

""" 
Mask to Palette  
"""

def load(path_ann):
    
    anns = sorted(glob.glob(os.path.join(path_ann, "*" )), key=os.path.basename)

    return anns

def mask2palette(ann, conv):
    mask = np.array(Image.open(ann)) # get rid of transparency

    drawing = np.zeros(mask.shape[:2]+(3,),dtype=np.uint8) # make it rgb
    for key, value in conv.items():
        drawing[mask==key] = value
    return drawing

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser(
        description="overlap")
    parser.add_argument("--anns",
                        default='outputs',
                        help="Location of mask files.")

    parser.add_argument("--save",
                        default="palettes",
                        help="Location of RGB masks to be saved.")


if __name__ == "__main__":
    path_ann = "outputs"
    path_save = "palettes"

    anns= load(path_ann)
    conv ={
        0: [26,27,26],
        1: [120,210,10],
        2: [209,233,11]
    }

    for ann in anns:
        
        mask = mask2palette(ann, conv)
        name = os.path.basename(ann) # It should be modifed for your use-case.

        logger.info(f"Annotation of {name} is processed.")

        Image.fromarray(mask).convert('RGB').save(os.path.join(path_save,name))

    logger.info(f"Conversion is successfully done.")