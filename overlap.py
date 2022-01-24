import matplotlib.pyplot as plt
import os
import glob
import cv2
import argparse

def load(path_ann, path_img):
    
    anns = sorted(glob.glob(os.path.join(path_ann, "*" )), key=os.path.basename)
    imgs = sorted(glob.glob(os.path.join(path_img, "*" )), key=os.path.basename)
    return anns, imgs


def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser(
        description="overlap")
    parser.add_argument("--anns",
                        default='anns',
                        help="Location of annotation files.")

    parser.add_argument("--images",
                        default='images',
                        help="Location of image files.")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    path_ann = args.anns
    path_img = args.images
    anns, imgs = load(path_ann, path_img)

    for ann, img in zip(anns, imgs):
        palette = cv2.cvtColor(cv2.imread(ann), cv2.COLOR_BGR2RGB)
        im = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        
        alpha = 0.5
        overlap = cv2.addWeighted(palette, alpha, im, 1-alpha, 0.0)

        plt.imshow(overlap)
        plt.title('overlap image')
        plt.show()
