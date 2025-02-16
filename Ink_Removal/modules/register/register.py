"""
The code was developed by 
Wenchao Han
Sunnybrook Research Institute
University of Toronto Medical Biophysics
wenchao.han@sri.utoronto.ca
"""
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

class ImageRegister:
    """
    Given an image, brings it to right format and registers the object to the destined image
    """
    scale_percent = 100   # percent of original size
   # construct the image class for registration
    def __init__(self, I_path=None, fname=None, image=None):
        self.path = I_path     # file path for the image
        self.fname = fname     # file name for the image
        self.im = image         # image np array
        self.gray = None       # gray level image
        self.resized = None    # down sampled image
        self.prepared = None   # preprocessed image for registration

    # set the down sample percentage for the the class variable(scale_percent)
    @classmethod
    def set_downsample_percent(cls, percent):
        cls.scale_percent = percent

    # allow the read image from full path
    @classmethod
    def from_full_path(cls, fullpath):
        fname = fullpath.split('/')[-1]
        path = fullpath.replace(fname, '')
        return cls(path, fname)

    def read_img(self):
        if self.im is None:
            self.im = cv2.imread(os.path.join(self.path, self.fname))

    def downsample_image(self, clear=False, print_scale=False, mode=None):
        if mode is not None:
            self.im = mode
        if self.im is None:
            self.read_img()
        width = int(self.im.shape[1] * self.scale_percent / 100)
        height = int(self.im.shape[0] * self.scale_percent / 100)
        # resize image
        self.resized = cv2.resize(self.im, (width, height), interpolation=cv2.INTER_CUBIC)
        if print_scale:
            print('image is down sampled by:', format(self.scale_percent/100, '.1%'))
        if clear:
            self.im = None
        return self.resized

    # convert image from color to grey, and down sample the image
    def prepare_img_registration(self, clear_all=True):
        if self.im is None:
            self.read_img()
        self.gray = cv2.cvtColor(self.im, cv2.COLOR_RGB2GRAY)
        self.prepared = self.downsample_image(False, mode=None)
        if clear_all:
            self.gray = None
            self.resized = None

    # perform image registration using SIFT features
    def perform_registration(self, I2, draw_match=False):
        """
        Performs registration from self to given target image
        """
        # find keypoints using SIFT feature detector
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(self.prepared, None)
        kp2, des2 = sift.detectAndCompute(I2.prepared, None)


        ## need to be re-defined ##
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        if len(good) > 3:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            Mat, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            ## rescaling need to be sperated
            S = np.identity(3)
            upsample_ratio = 100 / self.scale_percent
            S[0, 0] = upsample_ratio
            S[1, 1] = upsample_ratio
            M = S @ Mat @ np.linalg.inv(S)
            self.M = M
        if draw_match:
            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                               singlePointColor=None,
                            #    matchesMask=mask,  # draw only inliers
                               flags=2)
            img3 = cv2.drawMatches(self.prepared, kp1, I2.prepared, kp2, good, None, **draw_params)
            plt.imshow(img3), plt.show()

        return good, M

    # warp the image using the homograph matrix
    def warp_img(self, M, target_size):
        if self.im is None:
            self.read_img()
        self.warped = cv2.warpPerspective(self.im, M, target_size)

    # draw the ROI on the image
    def draw_region(self, y1, y2, x1, x2):
        pt_test = np.float32([[x1, y1], [x1, y2], [x2, y2], [x2, y1]])
        if self.im is None:
            self.read_img()
        self.im = cv2.cvtColor(self.im, cv2.COLOR_BGR2RGB)
        img2_t = cv2.polylines(self.im, [np.int32(pt_test)], True, (255, 0, 0), 18, cv2.LINE_AA)
        plt.imshow(img2_t), plt.show()
        self.im = self.read_img()
        return img2_t

    def crop_ROI(self, y1, y2, x1, x2):
        if self.im is None:
            self.read_img()
        return self.im[y1:y2, x1:x2]

    # project the region to the target image
    @staticmethod
    def project_coordinates(M, x1, y1, x2, y2):
        pt_test = np.float32([[x1, y1], [x1, y2], [x2, y2], [x2, y1]])
        pt_in = pt_test.reshape((-1, 1, 2))
        dst_in = cv2.perspectiveTransform(pt_in, M)
        dst_in_int = np.int32(dst_in)
        top_left_x = min([dst_in_int[0, 0, 0], dst_in_int[1, 0, 0], dst_in_int[2, 0, 0], dst_in_int[3, 0, 0]])
        top_left_y = min([dst_in_int[0, 0, 1], dst_in_int[1, 0, 1], dst_in_int[2, 0, 1], dst_in_int[3, 0, 1]])
        bot_right_x = max([dst_in_int[0, 0, 0], dst_in_int[1, 0, 0], dst_in_int[2, 0, 0], dst_in_int[3, 0, 0]])
        bot_right_y = max([dst_in_int[0, 0, 1], dst_in_int[1, 0, 1], dst_in_int[2, 0, 1], dst_in_int[3, 0, 1]])
        return top_left_y, bot_right_y, top_left_x, bot_right_x

if __name__=="__main__":
    #Read images
    orig_image = cv2.imread("img1.png")
    reg_image = cv2.imread("img2.png")

    # two image object for registration
    ImageRegister.set_downsample_percent(100)
    I_realHE = ImageRegister(image=orig_image)
    I_virtualHE = ImageRegister(image=reg_image)

    # prepare two images for registration
    I_virtualHE.prepare_img_registration()
    I_realHE.prepare_img_registration()
    
    #Plotting original images
    fig = plt.figure(num="Orig Images")
    plt.subplot(1,2,1)
    plt.imshow(I_realHE.prepared)
    plt.subplot(1,2,2)
    plt.imshow(I_virtualHE.prepared)
    plt.show()
    
    # perform registration and plot the matches
    good, M = I_virtualHE.perform_registration(I_realHE, draw_match=True)
    
    #Plot registered image
    fig = plt.figure(num="Registered Images")
    plt.subplot(1,3,1)
    plt.imshow(I_virtualHE.im)
    I_virtualHE.warp_img(M, (I_realHE.im.shape[1], I_realHE.im.shape[0]))
    plt.subplot(1,3,2)
    plt.imshow(I_virtualHE.warped)
    plt.subplot(1,3,3)
    plt.imshow(I_realHE.im)
