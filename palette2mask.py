import os
import glob
from PIL import Image
import numpy as np
from loguru import logger
import argparse

""" 
Palette to Mask (MMsegmentation Customdataset friendly)
"""

def load(path_ann):
    
    anns = sorted(glob.glob(os.path.join(path_ann, "*" )), key=os.path.basename)

    return anns

def palette2mask(ann, conv):
    palette = np.array(Image.open(ann))[...,:3] # get rid of transparency

    drawing = np.zeros(palette.shape[:2])
    for key, value in conv.items():
        region = np.all(palette==value,axis=-1)
        drawing[region] = key    
    return drawing

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser(
        description="overlap")
    parser.add_argument("--anns",
                        default='anns',
                        help="Location of annotation files.")

    parser.add_argument("--save",
                        default="outputs",
                        help="Location of image files.")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    path_ann = args.anns
    path_save = args.save
    anns= load(path_ann)

    # Key&Value should be adjusted based on your palette.
    conv ={
        0: [26,27,0],
        1: [120,210,10],
        2: [209,233,11]
    }

    for ann in anns:
        
        mask = palette2mask(ann, conv)
        name = os.path.basename(ann).split('__')[0] # It should be modifed for your use-case.

        logger.info(f"Annotation of {name} is processed.")

        name = name.split('.')[0] # get rid of jpg format
        Image.fromarray(mask).convert('P').save(os.path.join(path_save,name+'.png'))

    logger.info(f"Conversion is successfully done.")