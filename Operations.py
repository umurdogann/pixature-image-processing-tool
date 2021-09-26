# -*- coding: utf-8 -*-
# region IMPORTED LIBRARIES
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import cv2
from skimage import color, data, restoration
from tkinter import filedialog as fd
from scipy.signal import convolve2d
import math
import tkinter as tk
from PIL import Image, ImageTk, ImageOps
import copy
import os
from scipy.interpolate import UnivariateSpline
from pilgram import css
from pilgram import util
# endregion

# region GLOBAL VARIABLES
imagePath = ''
img = np.zeros(0)
cvImg = np.zeros(0)
opImg = np.zeros(0)
panel = []
undoList = list()
redoList = list()
# endregion

# region DISPLAY IMAGE ON FRAME FUNCTION


def DisplayImage(image=opImg):
    im = Image.fromarray(image)
    im = ResizeBaseHeight(im)
    imgtk = ImageTk.PhotoImage(image=im)
    panel.configure(image=imgtk)
    panel.image = imgtk
    pass
# endregion

# region UNDO/REDO IMAGE FUNCTIONS


def Redo():
    global cvImg, opImg, redoList, undoList
    if(len(redoList) > 0):
        undoList.insert(0, copy.deepcopy(cvImg))
        cvImg = copy.deepcopy(redoList[0])
        opImg = copy.deepcopy(cvImg)
        redoList.pop(0)
        DisplayImage(opImg)
        pass
    pass


def Undo():
    global cvImg, opImg, redoList, undoList
    if(len(undoList) > 0):
        redoList.insert(0, copy.deepcopy(cvImg))
        cvImg = copy.deepcopy(undoList[0])
        opImg = copy.deepcopy(cvImg)
        undoList.pop(0)
        DisplayImage(opImg)
        pass
    pass
# endregion

# region SAVING IMAGE FILE AS PNG FUNCTIONS


def SaveImageFileAs():
    global cvImg, opImg, panel, undoList, redoList
    file = fd.asksaveasfile(mode='w', defaultextension=".png", filetypes=(
        ("PNG file", "*.png"), ("All Files", "*.*")))
    if file:
        temp = copy.deepcopy(cvImg)
        abs_path = os.path.abspath(file.name)
        print(abs_path)
        b, g, r = cv2.split(cvImg)
        cvImg = cv2.merge((r, g, b))
        cv2.imwrite(abs_path, cvImg)
        cvImg = copy.deepcopy(temp)
        tk.messagebox.showinfo('Info', 'Image has been saved successfully!')
        pass
    pass


def SaveImageFile():
    global cvImg, imagePath
    temp = copy.deepcopy(cvImg)
    b, g, r = cv2.split(cvImg)
    cvImg = cv2.merge((r, g, b))
    cv2.imwrite(imagePath, cvImg)
    cvImg = copy.deepcopy(temp)
    tk.messagebox.showinfo('Info', 'Image has been saved successfully!')
    pass
# endregion

# region LOCAL RESET AND SAVE FUNCTIONS


def SaveImage():
    global cvImg, opImg, undoList, redoList
    tempImg = copy.deepcopy(cvImg)
    undoList.insert(0, tempImg)
    redoList = []
    cvImg = copy.deepcopy(opImg)
    pass


def ResetImage():
    global opImg, cvImg
    opImg = copy.deepcopy(cvImg)
    DisplayImage(opImg)
    pass
# endregion

# region RESIZE IMAGE FOR DYNAMIC FRAME


def ResizeBaseWidth(image):
    basewidth = 300
    wpercent = (basewidth/float(image.size[0]))
    hsize = int((float(image.size[1])*float(wpercent)))
    return image.resize((basewidth, hsize), Image.ANTIALIAS)


def ResizeBaseHeight(image):
    baseheight = 300
    hpercent = (baseheight/float(image.size[1]))
    wsize = int((float(image.size[0])*float(hpercent)))
    return image.resize((wsize, baseheight), Image.ANTIALIAS)
# endregion

# region LOAD IMAGE FUNCTION


def LoadImage():
    global cvImg, opImg, img
    global imagePath
    imagePath = str(fd.askopenfilename(filetypes=(
        ("Images", "*.jpg;*.png;*.jpeg"), ("PNG", "*.png"), ("JPEG", "*.jpeg"))))
    if(imagePath == ''):
        return False
    img = cv2.imread(imagePath)
    b, g, r = cv2.split(img)
    cvImg = cv2.merge((r, g, b))
    opImg = cv2.merge((r, g, b))
    DisplayImage(opImg)
    return True
# endregion

# region GRAYSACLE FUNCTION


def GrayScale():
    global cvImg
    cvImg = np.dot(cvImg[..., :3], [0.2989, 0.5870, 0.1140])
# endregion

# region ROTATE FUNCTION


def Rotate(degree):
    global cvImg
    global opImg
    opImg = cvImg.copy()
    opImg = ndimage.rotate(opImg, degree)
    DisplayImage(opImg)
# endregion

# region MIRROR FUNCTION


def Mirror(axis):
    global cvImg, opImg

    if(axis == 'x'):
        opImg = cv2.flip(opImg, 0)
        DisplayImage(opImg)
    if(axis == 'y'):
        opImg = cv2.flip(opImg, 1)
        DisplayImage(opImg)
    if(axis == 'xy'):
        opImg = cv2.flip(opImg, -1)
        DisplayImage(opImg)
# endregion

# region CROP FUNCTION


# operation : left,right,top,bottom,horizantal(left-right), vertical(top-bottom),all
imgCropTemp = []


def CropInit():
    global imgCropTemp, cvImg
    imgCropTemp = cvImg.copy()
    pass


def CropTempSave():
    global imgCropTemp, opImg
    imgCropTemp = opImg.copy()
    pass


def Crop(operation, percentage):
    global cvImg, opImg, imgCropTemp
    i_perc = int(percentage)/100
    if(operation == '' or i_perc < 0):
        return
    if(i_perc >= 0.5 and (operation == 'horizontal' or operation == 'vertical' or operation == 'all')):
        i_perc = 0.49
        pass

    lenX = imgCropTemp.shape[1]
    lenY = imgCropTemp.shape[0]
    perX = math.floor(lenX*i_perc)
    perY = math.floor(lenY*i_perc)
    if(operation == 'left'):
        opImg = imgCropTemp[:, perX:lenX]
    elif(operation == 'right'):
        opImg = imgCropTemp[:, 0:lenX-perX]
    elif(operation == 'top'):
        opImg = imgCropTemp[perY:lenY, :]
    elif(operation == 'bottom'):
        opImg = imgCropTemp[0:lenY-perY, :]
    elif(operation == 'horizontal'):
        opImg = imgCropTemp[:, perX:lenX-perX]
    elif(operation == 'vertical'):
        opImg = imgCropTemp[perY:lenY-perY, :]
    elif(operation == 'all'):
        opImg = imgCropTemp[perX:lenX-perX, perY:lenY-perY]
    DisplayImage(opImg)
    pass
# endregion

# region BRIGHTNESS & DARKNESS FUNCTION


def Brightness(value):
    global cvImg, opImg
    opImg = cv2.addWeighted(cvImg, 1, np.zeros(
        cvImg.shape, cvImg.dtype), 0, value)
    DisplayImage(opImg)
# endregion

# region CONTRAST FUNCTION


def Contrast(alpha=1.0):
    global cvImg, opImg
    opImg = cvImg.copy()
    if(alpha < 0.):
        alpha = 0.
    elif(alpha > 3.):
        alpha = 3.
    beta = 0.
    opImg = cv2.addWeighted(cvImg, alpha, np.zeros(
        cvImg.shape, cvImg.dtype), 0, beta)
    DisplayImage(opImg)
# endregion

# region BLUR FUNCTION


def Blur(method, k_x=5, k_y=5, bits=0):
    global cvImg, opImg
    if(k_x % 2 == 0):
        k_x += 1
    if(k_y % 2 == 0):
        k_y += 1
    if(method == 'gaussian'):
        opImg = cv2.GaussianBlur(cvImg, (k_x, k_y), 0)
    elif(method == 'averaging'):
        opImg = cv2.blur(cvImg, (k_x, k_y))
    elif(method == 'median'):
        opImg = cv2.medianBlur(cvImg, k_x)
    DisplayImage(opImg)
# endregion

# region DEBLUR FUNCTION (NOT WORKING CORRECTLY)


def Deblur():
    global cvImg
    cvImg = cv2.cvtColor(cvImg, cv2.COLOR_BGR2GRAY)
    psf = np.ones((5, 5)) / 25
    cvImg = convolve2d(cvImg, psf, 'same')
    cvImg += 0.1 * cvImg.std() * np.random.standard_normal(cvImg.shape)
    cvImg = restoration.wiener(cvImg, psf, 1, clip=False)
    DisplayImage()
# endregion

# region INVERT FUNCTION


def Invert():
    global cvImg, opImg
    opImg = (255-opImg)
    DisplayImage(opImg)
# endregion

# region HISTOGRAM NORMALIZATION FUNCTION


def HistogramNormalization(shift=0., cap=1., bits=0):
    global cvImg, opImg
    opImg = cvImg / 2**bits
    opImg = (opImg - np.amin(opImg))/np.amax(opImg)
    opImg[opImg > cap] = cap
    opImg -= shift
    opImg[opImg < 0] = 0.
    opImg = opImg / (cap-shift)
    opImg = (opImg * 255).astype(np.uint8)
    DisplayImage(opImg)
    pass
# endregion

# region MORPHOLOGICAL TRANSFORMATION FUNCTION


def MorphoTransform(method, kx=5, ky=5):
    global cvImg, opImg
    if(kx % 2 == 0):
        kx += 1
    if(ky % 2 == 0):
        ky += 1
    kernel = np.ones((kx, ky), np.uint8)
    if(method == 'opening'):
        opImg = cv2.morphologyEx(cvImg, cv2.MORPH_OPEN, kernel)
    elif(method == 'closing'):
        opImg = cv2.morphologyEx(cvImg, cv2.MORPH_CLOSE, kernel)
    elif(method == 'gradient'):
        opImg = cv2.morphologyEx(cvImg, cv2.MORPH_GRADIENT, kernel)
    elif(method == 'tophat'):
        opImg = cv2.morphologyEx(cvImg, cv2.MORPH_TOPHAT, kernel)
    elif(method == 'blackhat'):
        opImg = cv2.morphologyEx(cvImg, cv2.MORPH_BLACKHAT, kernel)
    DisplayImage(opImg)
    pass
# endregion

# region COLOR CHANNELS FUNCTION


def ColorChannels(r_val, g_val, b_val):
    global cvImg, opImg
    r, g, b = cv2.split(cvImg)
    r = cv2.addWeighted(r, 1, np.zeros(r.shape, r.dtype), 0, r_val)
    g = cv2.addWeighted(g, 1, np.zeros(g.shape, g.dtype), 0, g_val)
    b = cv2.addWeighted(b, 1, np.zeros(b.shape, b.dtype), 0, b_val)
    opImg = cv2.merge((r, g, b))
    DisplayImage(opImg)
    pass
# endregion

# region RESIZE IMAGE FUNCTION


def resize(image, w_height=500):
    aspect_ratio = float(image.shape[1])/float(image.shape[0])
    w_width = w_height/aspect_ratio
    image = cv2.resize(image, (int(w_height), int(w_width)))
    return image
# endregion

# region AUTOMATIC BRIGHTNESS AND CONTRAST FUNCTION


def AutomaticBrightnessAndContrast(clip_hist_percent=1):
    global opImg, cvImg
    gray = cv2.cvtColor(cvImg, cv2.COLOR_BGR2GRAY)
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)
    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    auto_result = cv2.convertScaleAbs(cvImg, alpha=alpha, beta=beta)
    opImg = auto_result
    DisplayImage(opImg)
    pass
# endregion

# region downsideUpFilter FUNCTION


def downsideUpFilter():
    global opImg
    global img

    kernel = np.array([[1, -1, 0], [-1, 4, -1], [-1, 0, -1]])
    # applying the kernel to the input image
    opImg = cv2.filter2D(img, -1, kernel)
    DisplayImage(opImg)
    pass
# endregion

# region SoftBWfilter FUNCTION


def SoftBWfilter():
    global opImg
    global img

    # allow the filter to process 30 times
    count = 30
    for _ in range(count):
        # smoothening images and reducing noise
        img_color = cv2.bilateralFilter(img, 10, 7, 3)
    opImg = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
    DisplayImage(opImg)
    pass
# endregion

# region cartoonizerEffectFilter FUNCTION


def cartoonizerEffectFilter():
    global opImg
    global img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # bluring
    gray = cv2.medianBlur(gray, 3)
    # edges were exposed
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 11)
    # smoothening images and reducing noise
    color = cv2.bilateralFilter(img, 9, 100, 100)
    # combining edges and color images
    opImg = cv2.bitwise_and(color, color, mask=edges)
    opImg = cv2.cvtColor(opImg, cv2.COLOR_BGR2RGB)

    DisplayImage(opImg)
    pass
# endregion

# region asheFilter FUNCTION


def asheFilter():
    global opImg
    global img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # bluring
    gray_blur = cv2.GaussianBlur(gray, (25, 25), 200)
    # divide blured image and gray image
    opImg = cv2.divide(gray, gray_blur, scale=100.0)
    DisplayImage(opImg)


    pass
# endregion

# region BRossFilter FUNCTION


def BRossFilter():
    global opImg
    global img

    # smoothening images and reducing noise
    img = cv2.bilateralFilter(img, 9, 100, 100)
    # changing the color channel in a different way
    b, g, r = cv2.split(img)
    opImg = cv2.merge((r, g, b))
    DisplayImage(opImg)


    pass
# endregion

# region NegativeFilter FUNCTION


def negativeFilter():
    global opImg
    global img

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # collapsed into one dimension
    k = img_gray.flatten()
    L = max(k)  # getting max value

    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            # getting reverse version of gray image
            img_gray[i, j] = L - img_gray[i, j]
    opImg = img_gray
    DisplayImage(opImg)


    pass
# endregion

# region coolFilter FUNCTION


def LUT_func(x, y):
    # Reduced to a single dimension
    spl = UnivariateSpline(x, y)
    return spl(range(256))


def coolFilter():
    global opImg
    global img

    incLUT = LUT_func([0, 64, 128, 192, 256], [0, 70, 140, 210, 256])
    decLUT = LUT_func([0, 64, 128, 192, 256], [0, 30, 80, 120, 192])

    c_b, c_g, c_r = cv2.split(img)
    # colormap that stored in a 256 x 1 color image  applied to an image using a lookup table LUT
    c_b = cv2.LUT(c_b, incLUT).astype(np.uint8)
    c_r = cv2.LUT(c_r, decLUT).astype(np.uint8)
    # decreasing the red color channel revealing the blue color channel
    img_rgb = cv2.merge((c_r, c_g, c_b))
    # Saturation was reduced to make these colors brighter than normal blue perception
    c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV))
    c_s = cv2.LUT(c_s, decLUT).astype(np.uint8)
    img_hsv = cv2.merge((c_h, c_s, c_v))
    opImg = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    DisplayImage(opImg)


    pass
# endregion

# region carbonPaperFilter FUNCTION


def carbonPaperFilter():
    global opImg
    global img

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # To obtain clear image in threshold
    img_blur = cv2.medianBlur(img_gray, 3)
    # edges were exposed
    opImg = cv2.adaptiveThreshold(
        img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)

    DisplayImage(opImg)


    pass
# endregion

# region warmFilter FUNCTION


def warmFilter():
    global opImg
    global img

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    increaseLUT = LUT_func([0, 64, 128, 192, 256], [0, 70, 140, 210, 256])
    decreaseLUT = LUT_func([0, 64, 128, 192, 256], [0, 30, 80, 120, 192])
    # colormap that stored in a 256 x 1 color image  applied to an image using a lookup table LUT
    c_r, c_g, c_b = cv2.split(img)
    # decreasing the blue color channel increasing the red color channel
    c_r = cv2.LUT(c_r, increaseLUT).astype(np.uint8)
    c_b = cv2.LUT(c_b, decreaseLUT).astype(np.uint8)
    img_rgb = cv2.merge((c_r, c_g, c_b))
    c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV))
    # Saturation was reduced to make these colors brighter than normal blue perception
    c_s = cv2.LUT(c_s, decreaseLUT).astype(np.uint8)
    img_hsv = cv2.merge((c_h, c_s, c_v))
    opImg = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    DisplayImage(opImg)

    pass
# endregion

# region masterSketcherFilter FUNCTION

def dodge_img(x,y):
    return cv2.divide(x,255-y,scale=256)
def burn_img(image, mask):
    return 255 - cv2.divide(255-image, 255-mask, scale=256)

def change_brightness(image, value=30):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,value)
    v[v > 255] = 255
    v[v < 0] = 0
    final = cv2.merge((h, s, v))
    image = cv2.cvtColor(final, cv2.COLOR_HSV2BGR)
    return image

def masterSketcherFilter():
    global opImg
    global img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # intensity 0
    img_bit = cv2.bitwise_not(gray)
    # bluring
    img_blur = cv2.GaussianBlur(img_bit, (21, 21), sigmaX=0, sigmaY=0)
    # converts the image to a faded image
    img_d = dodge_img(gray, img_blur)
    # image getting more dark
    final = burn_img(img_d, img_blur)
    # change brightness convert BGR then convert Gray
    gray = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
    # result approaches the drawing view, the image is dimmed
    final = change_brightness(gray, value=-10)
    opImg = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)

    DisplayImage(opImg)
    pass
# endregion

# region coloredMasterSketcherFilter FUNCTION
def change_saturation(image, value=30):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s,value)
    s[s > 255] = 255
    s[s < 0] = 0
    final = cv2.merge((h, s, v))
    image = cv2.cvtColor(final, cv2.COLOR_HSV2BGR)
    return image

def coloredMasterSketcherFilter():
    global opImg
    global img

    # vivid colors in the final image are rendered realistic
    image = change_saturation(img, value=-40)
    img_bit = cv2.bitwise_not(image)
    img_blur = cv2.GaussianBlur(img_bit, (21, 21), sigmaX=0, sigmaY=0)
    img_d = dodge_img(image, img_blur)
    final = burn_img(img_d, img_blur)
    opImg = change_brightness(final, value=-5)
    DisplayImage(opImg)

    pass
# endregion

# region embossFilter FUNCTION


def embossFilter():
    global opImg
    global img

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # emboss filter
    kernel = np.array(([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]), np.float32)
    # filter applied
    opImg = cv2.filter2D(src=img, kernel=kernel, ddepth=-2)
    DisplayImage(opImg)
    pass
# endregion

# region DownsideNeonFilter FUNCTION


def downsideNeonFilter():
    global opImg
    global img

    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # applying the kernel to the input image
    kernel = np.array(
        ([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]), np.float32)
    filter = cv2.filter2D(src=img, kernel=kernel, ddepth=-1)
    # applying the kernel2 to the input filtered image it makes the image sharper and neon colored
    kernel2 = np.array(
        ([[0, 2, 0], [-2, 5, -1], [0, -1, 0]]), np.float32)
    opImg = cv2.filter2D(src=filter, kernel=kernel2, ddepth=-5)
    DisplayImage(opImg)


    pass
# endregion


# region markedFilter FUNCTION


def markedFilter():
    global opImg
    global img

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # inverse version of color chanel
    img_bit = cv2.bitwise_not(img)
    # bluring
    blured = cv2.GaussianBlur(img_bit, (17, 53), sigmaX=8, sigmaY=10)
    # divide blured img and blured image but inverse version of blured image
    opImg = cv2.divide(img, 255 - blured, scale=256)
    DisplayImage(opImg)


    pass
# endregion

# region OTHER FILTERS
def blackSunny():
    global opImg
    global img
    #the width and length of the picture was found with .shape
    h, w = img.shape[:2]
    #With the cv2.cvtColor () function, the picture is
    #converted from the BGR color scale to the GRAY color scale
    imageG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #created a matrix of 1s suitable for the size of the picture.
    imm1 = np.ones((h, w), np.uint8) * 128
    #created a matrix of 0's suitable for the size of the picture
    imm0 = np.zeros((h, w), np.uint8)
    #created two kernels with k1 and k2 values
    #k1 is for the left side of the picture
    k1 = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]])
    #k2 is for the right side of the picture
    k2 = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])
    #k1 kernel value determined by the cv2.filter2D()
    # function has been combined with gray image
    img1 = cv2.filter2D(imageG, -1, k1)
    # Cv2.add () function added img1 and 1's matrix together.
    img1 = cv2.add(img1, imm1)
    # k2 kernel value determined by the cv2.filter2D()
    # function has been combined with gray image
    img2 = cv2.filter2D(imageG, -1, k2)
    # Cv2.add () function added img2 and 1's matrix together.
    img2 = cv2.add(img2, imm1)
    #reach each pixel of the picture
    for x in range(h):
        for y in range(w):
            #, the pixels of img1 and img2 were compared with the max()
            # function, and the large value was assigned to the 0’s matrix.
            imm0[x, y] = max(img1[x, y], img2[x, y])

    #openCV to PIL
    image = Image.fromarray(imm0)
    #converted to RGB format
    image = util.or_convert(image, 'RGB')
    #added new values to the picture
    image1 = util.radial_gradient(image.size, [(209, 187, 141), (53, 2, 10), (28, 3, 17)], [.3, .86, 1])
    #combined image and image1
    image = css.blending.overlay(image, image1)
    # converted to RGB format
    image = util.or_convert(image, 'RGB')
    #assign new values to the values in the image
    image1 = util.fill(image.size, [35, 35, 35])
    #combined image and image1
    image1 = css.blending.multiply(image, image1)
    #to find the color values of the image
    m = util.radial_gradient_mask(image.size, length=5.8, scale=2.6)
    # combined image, image1 and m
    image = Image.composite(image, image1, m)
    #adjusting the color values of the picture with contrast and saturate.
    image = css.saturate(image, 1.1)
    image = css.contrast(image, .8)
    opImg = css.saturate(image, 3.01)
    #convert PIL to openCV
    opImg = np.array(opImg)
    #call DisplayImage() function
    DisplayImage(opImg)


    pass


def vividNeon():
    global opImg
    global img

    #sobel() function to detect the edges of the image and im variables
    def sobel(image):
        #applied vertical sobel filter to sobel1
        sobel1 = cv2.Sobel(image, cv2.CV_8U, 0, 1, ksize=3)
        #applied horizontal sobel filter to sobel2
        sobel2 = cv2.Sobel(image, cv2.CV_8U, 1, 0, ksize=3)
        #We combined the vertical and horizontal sobel filter
        # applied to sobel1 and sobel2 variables
        return cv2.bitwise_or(sobel1, sobel2)

    #GaussianBlur () filter is applied to image
    image = cv2.GaussianBlur(img, (5, 5), 0)
    #To get a negative image
    im = 255 - image
    #call sobel() function with image
    im1 = sobel(image)
    #call sobel() function with im
    im2 = sobel(im)
    #combine the images of im1 and im2 with the Sobel filter applied.
    image = cv2.addWeighted(im1, 2.5, im2, 2, -2)
    #removed the picture from 255 to restore the picture
    opImg = 255 - image
    #call DisplayImage()
    DisplayImage(opImg)


    pass


def lala():
    global opImg
    global img

    #The lookup table was created using the UnivariateSpline() function
    def tab(i, j):
        #takes two array and generates a new callable value
        res = UnivariateSpline(i, j)
        return res(range(256))
    #color1 and color2 values get new values by calling the tab function
    color1 = tab([0, 50, 150, 200], [0, 40, 80, 160])
    color2 = tab([0, 64, 128, 256], [0, 70, 140, 210])
    #the picture is divided into red, green and blue color channels.
    red, green, blue = cv2.split(img)
    #Color1 and red values are matched using the LUT()
    #function for the new red value
    red = cv2.LUT(red, color1).astype(np.uint8)
    #Color2 and blue value are matched using the LUT()
    #function for the new blue value.
    blue = cv2.LUT(blue, color2).astype(np.uint8)
    #combine red green and blue
    image = cv2.merge((red, green, blue))
    #RGB color scale of the picture is converted to BGR
    opImg = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #call DisplayImage()
    DisplayImage(opImg)


    pass


def wonderland():
    global opImg
    global img
    #converted from the BGR color channel to the GRAY color channel
    imageG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Median blurring in 5 kernel sizes was applied in the picture
    imageM = cv2.medianBlur(imageG, 5)
    #thick edges were found
    imageE = cv2.adaptiveThreshold(imageM, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)
    #simplifies the colors while preserving the edges
    imageD = cv2.edgePreservingFilter(img, flags=2, sigma_s=64, sigma_r=0.25)
    #Smoothed image and thick edges were added to each other
    image = cv2.bitwise_and(imageD, imageD, mask=imageE)
    #RGB color scale of the picture is converted to BGR
    opImg = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #call DisplayImage
    DisplayImage(opImg)


    pass


def sundown():
    global opImg
    global img

    #image size has been edited in the def boyut() function
    def boyut(image, h=500):
        #By calculating the appropriate height and width value,
        #the picture is resized
        ratio = float(image.shape[1]) / float(image.shape[0])
        w = h / ratio
        image = cv2.resize(image, (int(h), int(w)))
        return image
    #call boyut function
    image = boyut(img, 500)
    #We set the kernel value with the np.array() function to sharpen the image.
    k = np.array([[1, -1, 0.4], [-1, 4.1, -1], [-1, 0, -1]])
    #combined the determined kernel value with the image with the function cv2.filter2D().
    image = cv2.filter2D(image, -10, k)
    #reduced the noise in the picture with the cv2.bilateral Filter()
    opImg = cv2.bilateralFilter(image, 0, 0, 0)
    opImg = cv2.cvtColor(opImg, cv2.COLOR_RGB2BGR)
    #call DisplayImage
    DisplayImage(opImg)


    pass


def glossy():
    global opImg
    global img

    # image size has been edited in the def boyut() function
    def boyut(image, h=500):
        # By calculating the appropriate height and width value,
        # the picture is resized
        ratio = float(image.shape[1]) / float(image.shape[0])
        w = h / ratio
        image = cv2.resize(image, (int(h), int(w)))
        return image
    # call boyut function
    image = boyut(img)
    #set the kernel value with the np.array() function to sharpen the image
    k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    #combined the determined kernel value with the image
    image = cv2.filter2D(image, -1, k)
    #reduced the noise in the picture
    image = cv2.bilateralFilter(image, 9, 75, 75)
    #conver image RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # openCV to PIL
    image = Image.fromarray(image)
    # converted to RGB format
    image = util.or_convert(image, 'RGB')
    #assigned new values to the values in the visual
    image1 = util.fill(image.size, [35, 35, 35])
    #combined image and image1
    image1 = css.blending.multiply(image, image1)
    #find the color values of the image
    m = util.radial_gradient_mask(image.size, length=5.8, scale=2.6)
    #combined image, image1 and m
    image = Image.composite(image, image1, m)
    #color values of the picture are changed with the saturate and contrast.
    image = css.saturate(image, 1.1)
    opImg = css.contrast(image, 1.5)
    #convert openCV to PIL
    opImg = np.array(opImg)
    #call DisplayImage
    DisplayImage(opImg)


    pass


def handDrawn():
    global opImg
    global img

    # image size has been edited in the def boyut() function
    def boyut(image, h=500):
        # By calculating the appropriate height and width value,
        # the picture is resized
        ratio = float(image.shape[1]) / float(image.shape[0])
        w = h / ratio
        image = cv2.resize(image, (int(h), int(w)))
        return image
    # call boyut function
    image = boyut(img)
    #Down holds the number of steps required to scale down
    down = 2
    #bil variable holds the number of steps of the image to be filtered double-sided
    bil = 50
    color = image
    #A subsampling image was obtained using the Gaussian
    # Pyramid with the function cv2.pyrDown()
    for i in range(down):
        color = cv2.pyrDown(color)
    #the picture is free of noise by creating for loop with the variable bil
    for j in range(bil):
        color = cv2.bilateralFilter(color, 9, 9, 7)
    #the top sampling of the image is obtained (image is enlarged).
    for t in range(down):
        color = cv2.pyrUp(color)
    #Color scale of the picture has been returned to GRAY
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #image is blurred
    blur = cv2.medianBlur(gray, 3)
    #thick edges were found
    edge = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
    #height and width of the image was found with .shape
    w, h, p = color.shape
    #image was resized
    edge = cv2.resize(edge, (h, w))
    # convert image GRAY to BGR
    edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)
    # openCV to PIL
    edge = Image.fromarray(edge)
    # converted to RGB format
    edge = util.or_convert(edge, 'RGB')
    #played with the contrast and brightness values
    image = css.contrast(edge, 1.6)
    image = css.brightness(image, .8)
    # converted to RGB format
    image = util.or_convert(image, 'RGB')
    #assigned new values to the values in the image
    im = util.fill(image.size, [250, 180, 149, .60])
    #combined image and im
    im2 = css.blending.darken(image, im)
    #assigned new values to the values in the image
    im3 = util.fill(image.size, [1, 69, 149, .5])
    #combined im2 and im3
    image = css.blending.lighten(im2, im3)
    # color values of the picture are changed with brightness, saturate and contrast.
    image = css.sepia(image, .3)
    image = css.contrast(image, 1.4)
    image = css.brightness(image, 1.04)
    opImg = css.saturate(image, 1.3)
    # convert openCV to PIL
    opImg = np.array(opImg)
    # call DisplayImage
    DisplayImage(opImg)


    pass


def oldTown():
    global opImg
    global img

    # image size has been edited in the def boyut() function
    def boyut(image, h=500):
        # By calculating the appropriate height and width value,
        # the picture is resized
        ratio = float(image.shape[1]) / float(image.shape[0])
        w = h / ratio
        image = cv2.resize(image, (int(h), int(w)))
        return image
    # call boyut function
    image = boyut(img)
    #and height value of the picture was found with .shape
    w, h, t = image.shape
    #reach each pixel value in the image
    for i in range(w):
        for j in range(h):
            #Red, green and blue values for each pixel in for
            # loop are calculated separately.
            red = image[i, j, 2] * 0.0 + image[i, j, 1] * \
                0.500 + image[i, j, 0] * 0.400
            green = image[i, j, 2] * 0.0 + image[i, j, 1] * \
                0.400 + image[i, j, 0] * 0.300
            blue = image[i, j, 2] * 0.0 + image[i, j, 1] * \
                0.300 + image[i, j, 0] * 0.200
            #If these values are not greater than 255, the
            # calculated values have been replaced with the values of the
            # pixels in the picture.
            if red > 255:
                image[i, j, 2] = 255
            else:
                image[i, j, 2] = red
            if green > 255:
                image[i, j, 1] = 255
            else:
                image[i, j, 1] = green
            if blue > 255:
                image[i, j, 0] = 255
            else:
                image[i, j, 0] = blue
    # convert image RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # openCV to PIL
    image = Image.fromarray(image)
    # converted to RGB format
    image = util.or_convert(image, 'RGB')
    #values of the picture are changed
    image1 = util.radial_gradient(image.size, [(209, 187, 141), (53, 2, 10), (28, 3, 17)], [.3, .86, 1])
    #combined image and image1
    image = css.blending.overlay(image, image1)
    # color values of the picture are changed with contrast.
    opImg = css.contrast(image, 1.)
    # convert openCV to PIL
    opImg = np.array(opImg)
    # call DisplayImage
    DisplayImage(opImg)


    pass


def atDawn():
    global opImg
    global img

    #lookup table was created using the UnivariateSpline()
    # function in the def spline() function.
    def spline(i, j):
        #The UnivariateSpline() function takes an array of two
        # values and generates a new callable value.
        splin = UnivariateSpline(i, j)
        return splin(range(256))
    #image is divided into red, green and blue color channels.
    red, green, blue = cv2.split(img)
    #For the new red value, the values sent to the spline() function
    # using the LUT() function and the green value are matched.
    red = cv2.LUT(green, spline([0, 64, 128, 192, 256], [
                  0, 70, 140, 210, 256])).astype(np.uint8)
    #For the new blue value, the values sent to the spline() function
    # using the LUT() function and the red value are matched
    blue = cv2.LUT(red, spline([0, 50, 100, 150, 256], [
                   0, 30, 80, 120, 192])).astype(np.uint8)
    #combined red, green and blue
    image = cv2.merge((red, green, blue))
    #converted the image from RGB to HSV
    #image is divided into hue, saturation and value color channels
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
    #For the hue value, the values sent to the spline() function
    # using the LUT () function and the value are matched.
    h = cv2.LUT(v, spline([0, 64, 128, 192, 256], [
                0, 70, 140, 210, 256])).astype(np.uint8)
    #combined hue, saturation and value
    #converted it from HSV to RGB
    image = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2RGB)
    # convert image RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # openCV to PIL
    image = Image.fromarray(image)
    # converted to RGB format
    image = util.or_convert(image, 'RGB')
    #assigned new values to the values in the image
    im1 = util.fill(image.size, [0, 68, 204])
    #combined image and im1
    im1 = css.blending.screen(image, im1)
    #combined image, im1
    image = Image.blend(image, im1, .3)
    #color values of the picture are changed with contrast, hue_rotate and saturate.
    image = css.brightness(image, 1.1)
    image = css.hue_rotate(image, -10)
    opImg = css.saturate(image, 1.7)
    # convert openCV to PIL
    opImg = np.array(opImg)
    # call DisplayImage
    DisplayImage(opImg)


    pass


def candyGirl():
    global opImg
    global img

    # lookup table was created using the UnivariateSpline()
    # function in the def spline() function.
    def spline(i, j):
        # The UnivariateSpline() function takes an array of two
        # values and generates a new callable value.
        splin = UnivariateSpline(i, j)
        return splin(range(256))
    ##image is divided into red, green and blue color channels.
    red, green, blue = cv2.split(img)
    #For the new blue value, the values sent to the spline() function
    # using the LUT() function and the blue value are matched
    blue = cv2.LUT(blue, spline([0, 64, 128, 192, 256], [
                   0, 70, 140, 210, 256])).astype(np.uint8)
    #For the new green value, the values sent to the spline() function
    # using the LUT() function and the green value are matched.
    green = cv2.LUT(green, spline([0, 64, 128, 192, 256], [
                    0, 30, 80, 120, 192])).astype(np.uint8)
    #combined red, green and blue
    image = cv2.merge((red, green, blue))
    #converted the image from RGB to HSV
    #image is divided into hue, saturation and value color channels
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
    #For the Saturation value, the values sent to the spline() function
    # using the LUT() function and the hue value are matched
    s = cv2.LUT(h, spline([0, 64, 128, 192, 256], [
                0, 70, 140, 210, 256])).astype(np.uint8)
    # combined hue, saturation and value
    # converted it from HSV to RGB
    image = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2RGB)
    #convert image RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # openCV to PIL
    image = Image.fromarray(image)
    # converted to RGB format
    image = util.or_convert(image, 'RGB')
    #assigned new values
    im1 = util.fill(image.size, [243, 106, 188, .3])
    #combined image and im1
    image = css.blending.screen(image, im1)
    # color values of the picture are changed with contrast, brightness and saturate
    image = css.contrast(image, 1.1)
    image = css.brightness(image, 1.1)
    opImg = css.saturate(image, 1.3)
    # convert openCV to PIL
    opImg = np.array(opImg)
    # call DisplayImage
    DisplayImage(opImg)


    pass
# endregion
