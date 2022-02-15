import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.morphology import label
from skimage.measure import regionprops
import numpy as np
import utils
import sys 
np.set_printoptions(threshold=sys.maxsize)


matrix_1= np.zeros((50,50))
matrix_1[2:10,2:10]=1
matrix_1[25:30,25:30]=1

matrix_2= np.zeros((50,50))
matrix_2[2:10,2:10]=1
matrix_2[25:30,25:30]=1

matrix_3= np.zeros((50,50))
matrix_3[2:10,2:10]=1
matrix_3[25:30,25:30]=1
def Mask2Detection(mask, img_id):

    label_image = label(mask, connectivity=1)
    print(label_image)
    

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(label_image)

    predicciones = []
    print(len(regionprops(label_image)))
    for region in regionprops(label_image):

        # take regions with large enough areas
        
            # draw rectangle around segmented coins
        min_row, min_col, max_row, max_col = region.bbox
        print("print:",min_row, min_col, max_row, max_col)
        rect = mpatches.Rectangle((min_col-1, min_row-1), max_col - min_col+1, max_row - min_row+1,
                                fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
        #mask= cv2.rectangle(mask, (int(min_row), int(min_col)), (int(max_row), int(max_col)), 3,thickness= 7)
        
        #score = utils.pred_score(mask)
        #print(score)
        
        dic = {"image_id": img_id, 
            "category_id": 1,
            "bbox": [
                min_row,
                min_col,
                max_row-min_row,
                max_col -min_col
            ],
            "score": 1}
        img_id += 1
        predicciones.append(dic)
        
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    print(predicciones)
    return predicciones

    
Mask2Detection(matrix_2, 1)
"""
split={}
predicciones = []
img_id= 1

for img in split:
    mask = split[img]
    predicciones.append(Mask2Detection(mask, img_id))
    img_id +=1
"""