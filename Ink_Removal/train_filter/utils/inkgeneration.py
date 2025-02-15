#
# --------------------------------------------------------------------------------------------------------------------------
# Created on Tue Jul 05 2022 at University of Toronto
#
# Author: Vishwesh Ramanathan
# Email: vishwesh.ramanathan@mail.utoronto.ca
# Description: This script is about ink generation. This code is based on ink generation code written during rotations, find
# --------------------------------------------------------------------------------------------------------------------------

import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy.stats import norm


class InkGenerator:
    """
    Class for generating fake ink images by randomly extracting a patch from handwritten images
    and overlaying on top of a given image based on given parameters
    """
    #get_template
    TEMPLATE_THRESH = 10
    #_get_distributed parameters
    DIST_PROB = [0.3,0.7]
    #generate_fake parameters
    CROP_KERNEL = 81
    IMG_KERNEL = 21
    #get_random_patch parameters
    AREA_THRESH = 0.3
    ANGLE = 30
    SCALE = 3
    HEIGHT_RANGE = [0.25,0.95]
    WIDTH_RANGE = [0.25,0.95]
    LOOP_BREAK = 30
    #main_process parameters
    TYPE_PROB = [0.5,0.8]
    MASK_THRESH = 0.15
    #Alpha for opacity
    ALPHA = [0.5,1]

    def __init__(self,ink_template,colors)->None:
        """
        ink_template(torch.utils.data.Dataset): Handwritten dataset
        colors(List[Tuple[str,str]]): List of tuples of colors, with tuple having colors with range from lighter to darker shade
        """
        self.ink_template = ink_template
        self.n_templ = len(self.ink_template)
        self.colors = colors
        
    @staticmethod
    def _get_distributed(n):
        '''
        Has three different type of variations of marker to emulate the pen marks
        1) linear
        2) Bell curve/Inverted Bell curve
        3) Plain
        '''
        p = torch.rand(1).item()
        q = torch.rand(1).item()
        if q<InkGenerator.DIST_PROB[0]:
            #Linear
            Y = np.linspace(0,1,n)
            x_range_corr = np.linspace(0,1,n)
        elif q<InkGenerator.DIST_PROB[1]:
            #Bell Curve
            x_range = np.linspace(-1,1,n)
            Y = norm.pdf(x_range,0,1)
            Y = (Y-min(Y))/(max(Y)-min(Y)+0.00001)
            x_range_corr = (x_range-min(x_range))/(max(x_range)-min(x_range)+0.00001)
        else:
            x_range_corr = np.linspace(0,1,n)
            Y = torch.rand(1).item()*np.ones(n)
        if p>0.5:
            Y = 1 - Y
        return x_range_corr,Y

    @staticmethod
    def _generate_fake(img,crop,color_matrix):
        '''
        Based on the given patch, augments data by generating fake images using the color matrix provided
        '''
        H,W,_ = np.shape(img)
        crop = cv2.resize(crop, (H,W), interpolation = cv2.INTER_AREA)
        crop_blur = cv2.GaussianBlur(crop.astype(np.float),(InkGenerator.CROP_KERNEL,InkGenerator.CROP_KERNEL),cv2.BORDER_DEFAULT)
        crop_blur = cv2.blur(crop.astype(np.float),(InkGenerator.IMG_KERNEL,InkGenerator.IMG_KERNEL))

        img_color = color_matrix.copy()
        img_color = cv2.resize(img_color, (H,W), interpolation = cv2.INTER_AREA)
        img_color = cv2.GaussianBlur(img_color,(InkGenerator.IMG_KERNEL,InkGenerator.IMG_KERNEL),cv2.BORDER_DEFAULT)
        img_color = cv2.blur(img_color,(InkGenerator.IMG_KERNEL,InkGenerator.IMG_KERNEL))
        
        mask = crop_blur/255
        patch = img_color.copy()
        
        #Generate the fake image by overlaying op top of other
        alpha=torch.distributions.uniform.Uniform(InkGenerator.ALPHA[0],InkGenerator.ALPHA[1]).sample().item()
        noise_img = img.copy()/255
        A = np.clip(1-mask,1-alpha,1)
        B = np.clip(mask,0,alpha)
        noise_img = (A[:,:,np.newaxis])*img/255 + patch*B[:,:,np.newaxis]
        return noise_img,mask,alpha

    def get_template(self):
        h,w=0,0
        while min(h,w)<InkGenerator.TEMPLATE_THRESH:
            n = torch.randint(0,self.n_templ,(1,)).item()
            image = self.ink_template[n]
            (T, thresh) = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
            x_min,x_max = np.sort(np.where(thresh!=0))[0][[0,-1]]
            y_min,y_max = np.sort(np.where(thresh!=0))[1][[0,-1]]
            w = y_max-y_min
            h = x_max-x_min
            x = y_min
            y = x_min
            rect2 = thresh[y:y+h,x:x+w]
        
        return rect2



    def get_random_patch(self, image)->np.array:
        """
        Randomly fetches a patch from a given image
        """ 
        h,w = image.shape
        center = (w / 2, h / 2)
        white_pixel = 0
        i = 0
        while white_pixel<InkGenerator.AREA_THRESH: #To ensure it doesnt get stuck infinitely
            angle = torch.randint(0,InkGenerator.ANGLE,(1,)).item()
            scale = torch.randint(1,InkGenerator.SCALE,(1,)).item()
            # Perform the rotation
            M = cv2.getRotationMatrix2D(center, angle, scale)
            image = cv2.warpAffine(image, M, (w, h))
            crop_height = torch.randint(int(InkGenerator.HEIGHT_RANGE[0]*h),int(InkGenerator.HEIGHT_RANGE[1]*h),(1,)).item()
            crop_width = torch.randint(int(InkGenerator.WIDTH_RANGE[0]*w),int(InkGenerator.WIDTH_RANGE[1]*w),(1,)).item()
            max_x = image.shape[1] - crop_width
            max_y = image.shape[0] - crop_height

            x = torch.randint(0, max_x,(1,)).item()
            y = torch.randint(0, max_y,(1,)).item()

            crop = image[y: y + crop_height, x: x + crop_width]
            white_pixel = np.sum(crop>0)/(crop_height*crop_width)
            #Flip randomly
            p = torch.rand(1).item()
            if p<0.5:
                q = torch.rand(1).item()
                if q<0.5:
                    crop = crop[:,::-1]
                else:
                    crop = crop[::-1,:]
            i+=1
            if i>InkGenerator.LOOP_BREAK:
                #if nothing works, just fully mask it
                crop[:,:] = 255
        return crop

    def _get_cgradient(self,c1,c2,size):
        '''
        For two mpl colour names, gets the color gradient according to the distribution obtained
        c1: Name of colour 1
        c2: Name of colour 2
        
        '''
        C1 = np.array(mpl.colors.to_rgb(c1))
        C2 = np.array(mpl.colors.to_rgb(c2))
        p1,p2 =  torch.rand(2).numpy()
        c1 = p1*C1 + (1-p1)*C2
        c2 = p2*C2 + (1-p2)*C1
        h,w = size

        color_matrix = np.zeros((h,w,3))
        #Generate color matrix
        #Top or bottom
        p = torch.rand(1).item()
        if p<0.5:
            _,mix = self._get_distributed(w)
            for e,i in enumerate(mix):
                c3 = c1*i+(1-i)*c2
                color_matrix[:,e,:] = c3
        else:
            _,mix = self._get_distributed(h)
            for e,i in enumerate(mix):
                c3 = c1*i+(1-i)*c2
                color_matrix[e,:,:] = c3

        return color_matrix

    def generate(self,img):
        '''
        Main process of generating the marker img with masks
        Parameters:
            img: Main image where the ink marks are generated
        Returns:
            crop: Cropped patch
            color_matrix2: Color of the patch
            noise_img: Final preprocessed image
            mask: Mask of the ink
            flag: to know the type of augmentation
                    0: Full covering
                    1: Partial covering till one end
                    2: A streak or a blob
        '''
        rect2 = self.get_template()

        #Extent of cover by the streak
        p = torch.rand(1).item()
        if p<InkGenerator.TYPE_PROB[0]:
            #Denotes fully covering
            flag = 0
            crop = 255*np.ones(np.shape(rect2))
        elif p<InkGenerator.TYPE_PROB[1]:
            #Denotes partially covering
            flag = 1
            crop = self.get_random_patch(rect2)
            coords = np.where(crop>0)
            p = torch.rand(1).item()
            if p<0.5:
                for i in range(len(coords[1])):
                    crop[:coords[0][i],coords[1][i]] = 255
            else:
                for i in range(len(coords[0])):
                    crop[coords[0][i],:coords[1][i]] = 255
            if torch.rand(1).item()<0.5:
                temp_crop = 255-crop
                crop_height,crop_width = np.shape(temp_crop)
                if (np.sum(temp_crop>0)/(crop_height*crop_width)) > 0.3:
                    crop = 255-crop
        else:
            #Denotes just streak/blob
            flag = 2
            crop = self.get_random_patch(rect2)

        #Pick random color pair
        n_colors = len(self.colors)
        color_idx = torch.randint(0, n_colors,(1,)).item()
        c1,c2 = self.colors[color_idx]
        #Combine all the components together
        color_matrix2 = self._get_cgradient(c1,c2,np.shape(crop))
        noise_img,mask,alpha = self._generate_fake(img,crop,color_matrix2)
        
        return crop,color_matrix2,noise_img,(mask>InkGenerator.MASK_THRESH)*1,flag,alpha
    
    def get_plots(self, img, figsize:tuple=(10,10)):
        '''
        Based on list of images, get subplots
        IMG = list of images you wish to plot
        format = number of rows
        figsize_given = size of figure
        '''
        fig = plt.figure(figsize=figsize)
        crop,color_matrix2,noise_img,mask,flag,alpha = self.generate(img)
        args = [img,crop,mask,color_matrix2,noise_img]
        n = len(args)
        for i in range(n):
            plt.subplot(1,n,i+1)
            img = cv2.resize(args[i], (256,256), interpolation = cv2.INTER_AREA)
            plt.imshow(img)
        return fig
