from typing import Tuple

import cv2
import openslide
import numpy as np
from matplotlib import pyplot as plt

from .register import ImageRegister

class Pairwise_Extractor:
    THRESHOLD = 50
    def __init__(self,
                 src_slide: openslide.OpenSlide,
                 dest_slide: openslide.OpenSlide,
                 plot:bool=False) -> None:
        
        self.src_slide = src_slide
        self.dest_slide = dest_slide
        self.plot = plot
        
        #Get projection matrices
        self.M, self.M_inv = self.wsi_registration()


    @classmethod
    def from_path(cls,
                  src_path:str,
                  dest_path:str,
                  plot:bool=False,
                ):
        """
        Builds openslide objects given their paths for registration
        """
        src_slide = openslide.OpenSlide(src_path)
        dest_slide = openslide.OpenSlide(dest_path)

        return cls(src_slide,dest_slide,plot)

    def get_homography(self,
                       src_img:np.array,
                       dest_img:np.array,
                       ) ->Tuple[np.array,ImageRegister]:
        """
        Given two images , gets projection matrix such that dest_img = H.src_img
        Returns:
            H: Projection matrix
            proj_img: Projected image
        """
        src_object = ImageRegister(image=src_img)
        dest_object = ImageRegister(image=dest_img)
        
        # prepare two images for registration
        src_object.prepare_img_registration()
        dest_object.prepare_img_registration()

        # perform registration
        good, M = src_object.perform_registration(dest_object, draw_match=self.plot)

        #Obtain projected image
        proj_object = ImageRegister(image=src_img)
        proj_object.warp_img(M, (dest_img.shape[1], dest_img.shape[0]))

        return M, proj_object.warped

    def wsi_registration(self) -> Tuple[np.array,np.array]:
        """
        Registers two whole slides and outputs the project matrix that can be used for the original slide
        Populates two variables:
            M_adjusted: projection matrix from source slide to destination slide
            M_inv_adjusted: projection matrix from destination slide to source slide
        """
        src_dimension = self.src_slide.dimensions
        dest_dimension = self.dest_slide.dimensions

        #Extract thumbnail from given slide
        src_thumb = np.asarray(self.src_slide.get_thumbnail((3000,3000)).convert("RGB"))
        dest_thumb = np.asarray(self.dest_slide.get_thumbnail((3000,3000)).convert("RGB"))

        #Get the scaling for getting M
        src_scaling = np.array([
                                [src_dimension[0]/src_thumb.shape[1], 0, 0],
                                [0,src_dimension[1]/src_thumb.shape[0],0],
                                [0,0,1]]
                                )
        dest_scaling = np.array([
                                [dest_dimension[0]/dest_thumb.shape[1], 0, 0],
                                [0,dest_dimension[1]/dest_thumb.shape[0],0],
                                [0,0,1]]
                                )
        
        #Get projection matrix for thumbnails
        M_thumb, _ = self.get_homography(src_thumb,dest_thumb)
        M_thumb_inv, _ = self.get_homography(dest_thumb,src_thumb)
        M_adjusted = dest_scaling @ M_thumb @ np.linalg.inv(src_scaling)
        M_inv_adjusted = src_scaling @ M_thumb_inv @ np.linalg.inv(dest_scaling)
        
        return M_adjusted,M_inv_adjusted
        
    def extract(self,
                dest_x:int,
                dest_y:int,
                size:Tuple[int,int]=(256,256),
                )->Tuple[np.array,np.array,np.array]:
        """
        Given parameters extracts patchs from source and destication slide, co-registered
        """        
        #Step 1: Inverse projection, get coordinates in no ink coordinate system
        four_box = [(dest_x, dest_y),
                    (dest_x + size[0], dest_y),
                    (dest_x, dest_y + size[1]),
                    (dest_x + size[0],dest_y + size[1])
                    ]
        
        transformed_box = np.array([self.transform_coords(*pts,self.M_inv) for pts in four_box])

        #Step 2: Get the coordinates of the bounding box surrounding the transformed box and the patch
        x_corner,y_corner = np.min(transformed_box,axis=0)
        h,w = np.max(transformed_box,axis=0) - np.min(transformed_box,axis=0)
        
        x_mod,y_mod = x_corner-self.THRESHOLD, y_corner-self.THRESHOLD
        new_src_size = (h+2*self.THRESHOLD,w+2*self.THRESHOLD)

        src_dimension = self.src_slide.dimensions
        #Check corner case
        if (x_mod>=src_dimension[0]) or ( (x_mod+new_src_size[1])>=src_dimension[0]) or (y_mod>=src_dimension[1]) or (y_mod+new_src_size[0]>=src_dimension[1]):
            return None, None, None  
        
        dest_patch = np.asarray(self.dest_slide.read_region((dest_x,dest_y),
                                                            0,
                                                            size
                                                            ).convert("RGB"))
        
        src_patch = np.asarray(self.src_slide.read_region((x_mod,y_mod),
                                                          0,
                                                          new_src_size,
                                                         ).convert("RGB"))

        # M_new,paired_img = self.get_homography(src_patch,dest_patch)
        M_new = self.find_new_homography(src_corner=(x_mod,y_mod),dest_corner=(dest_x,dest_y),M=self.M)
        paired_img = cv2.warpPerspective(src_patch, M_new, size)

        if self.plot:
           #Compare
            fig,axes= plt.subplots(1,3,num="compare")
            axes[0].set_title("destination patch")
            axes[0].imshow(dest_patch)
            axes[1].set_title("source patch, coordinates transform")
            axes[1].imshow(src_patch)
            axes[2].set_title("source patch, registered")
            axes[2].imshow(paired_img)

        return dest_patch, src_patch, paired_img    
    
    def find_new_homography(self, src_corner:np.array, dest_corner:np.array, M:np.array):
        """
        Finds homography for each patch
        """
        s1 = src_corner[0]
        s2 = src_corner[1]
        t1 = dest_corner[0]
        t2 = dest_corner[1]
        Mat = np.array([[M[0,0]-M[2,0]*t1, M[0,1]-M[2,1]*t1, M[0,2]-M[2,2]*t1 + s1*(M[0,0]-M[2,0]*t1) + s2*(M[0,1]-M[2,1]*t1)],
                        [M[1,0]-M[2,0]*t2, M[1,1]-M[2,1]*t2, M[1,2]-M[2,2]*t2 + s1*(M[1,0]-M[2,0]*t2) + s2*(M[1,1]-M[2,1]*t2)],
                        [M[2,0], M[2,1], M[2,2] + M[2,0]*s1 + M[2,1]*s2]])

        return Mat

    @staticmethod
    def transform_coords(x:int,y:int,M:np.array)->Tuple[int,int]:
        p = np.array((x,y,1)).reshape((3,1))
        temp_p = M.dot(p)
        sum = np.sum(temp_p ,1)
        px = int(round(sum[0]/sum[2]))
        py = int(round(sum[1]/sum[2]))
        return px,py
    
    @property
    def proj_matrix(self):
        return self.M
    
    @property
    def inv_proj_matrix(self):
        return self.M_inv
    
    
if __name__=="__main__":
    size_img = (256,256)

    #Read slides
    img_ink_path = "img1.svs"
    img_noink_path = "img2.svs"

    # #Read region from ink
    x_point,y_point = (14349,12345)

    patch_extractor = Pairwise_Extractor.from_path(src_path=img_noink_path, dest_path=img_ink_path, plot=True)

    start = time.time()
    ink_img, src_patch, reg_noink = patch_extractor.extract(x_point,y_point,size_img)
    end = time.time()
    print("Time taken:{}".format(end-start))

    plt.show()
    
    
    
    
