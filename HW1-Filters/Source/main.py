# Instructions:
# For question 1, only modify function: histogram_equalization
# For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
# For question 3, only modify function: laplacian_pyramid_blending

import os
import sys
import cv2
import numpy as np


def help_message():
    print("Usage: [Question_Number] [Input_Options] [Output_Options]")
    print("[Question Number]")
    print("1 Histogram equalization")
    print("2 Frequency domain filtering")
    print("3 Laplacian pyramid blending")
    print("[Input_Options]")
    print("Path to the input images")
    print("[Output_Options]")
    print("Output directory")
    print("Example usages:")
    print(sys.argv[0] + " 1 " + "[path to input image] " +
          "[output directory]")  # Single input, single output
    print(sys.argv[0] + " 2 " + "[path to input image1] " +
          "[path to input image2] " +
          "[output directory]")  # Two inputs, three outputs
    print(sys.argv[0] + " 3 " + "[path to input image1] " +
          "[path to input image2] " +
          "[output directory]")  # Two inputs, single output


# ===================================================
# ======== Question 1: Histogram equalization =======
# ===================================================

# def histogram_equalization(img_in):
#     # extracting dimensions
#     width, height, layers = img_in.shape
#     img_out = np.zeros([width,height,layers])
#     # Write histogram equalization here
#     for i in range(0,layers):
#         #build normalized histogram and cdfs
#         hist = cv2.calcHist([img_in],[i],None,[256],[0,256])/(width*height)
#         cdf = np.cumsum(hist)
#         #Mapping into new img_out
#         reshapedImg = img_in[:,:,i].reshape(width*height)
#         reshapedImg  = np.floor(np.multiply(cdf[reshapedImg],255))
#         img_out[:,:,i]  = reshapedImg.reshape(width,height) #result
#     return True, img_out


def histogram_equalization(img_in):
    # extracting dimensions
    width, height, layers = img_in.shape
    #img_out = np.zeros([width,height,layers])
    # Write histogram equalization here
    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_in)
    img_in = cv2.merge((h, s, v))
    #build normalized histogram and cdfs
    hist = cv2.calcHist([v], [0], None, [256], [0, 256]) / (width * height)
    cdf = np.cumsum(hist)
    #Mapping into new img_out
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    v_new = cdf[v]
    img_out = cv2.merge([h, s, v_new])
    img_out = cv2.cvtColor(img_out, cv2.COLOR_HSV2BGR)
    return True, img_out


def Question1():

    # Read in input images
    input_image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)

    # Histogram equalization
    succeed, output_image = histogram_equalization(input_image)

    # Write out the result
    output_name = sys.argv[3] + "1.jpg"
    cv2.imwrite(output_name, output_image)

    return True


# ===================================================
# ===== Question 2: Frequency domain filtering ======
# ===================================================


def ft(im, newsize=None):
    dft = np.fft.fft2(np.float32(im), newsize)
    return np.fft.fftshift(dft)


def ift(shift):
    f_ishift = np.fft.ifftshift(shift)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back)


def low_pass_filter(img_in):
    width, height, layers = img_in.shape
    img_out = np.zeros([width, height, layers])
    for i in range(0, layers):
        #converting into float32 format
        img_in[:, :, i] = np.float32(img_in[:, :, i])
        #2D Discrete Fourier Transform (DFT) on image which returns real and imaginary matrix of same size
        dft = cv2.dft(np.float32(img_in[:, :, i]), flags=cv2.DFT_COMPLEX_OUTPUT)
        #shifting to origin
        dft_shift = np.fft.fftshift(dft)
        #creating window
        rows, cols = img_in[:, :, i].shape
        crow, ccol = rows / 2, cols / 2
        #masking the dft_shift
        mask = np.zeros((rows, cols, 2), np.uint8)
        mask[crow - 10:crow + 10, ccol - 10:ccol + 10] = 1
        fshift = dft_shift * mask
        # inverting the shift
        f_ishift = np.fft.ifftshift(fshift)
        # inverse dft
        temp = cv2.idft(f_ishift)
        img_out[:, :, i] = cv2.magnitude(temp[:, :, 0], temp[:, :, 1])
    img_out = np.divide(img_out * 255, np.max(img_out))
    return True, img_out


def high_pass_filter(img_in):
    #similar to low pass filter
    width, height, layers = img_in.shape
    img_out = np.zeros([width, height, layers])
    for i in range(0, layers):
        img_in[:, :, i] = np.float32(img_in[:, :, i])
        dft = cv2.dft(np.float32(img_in[:, :, i]), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        rows, cols = img_in[:, :, i].shape
        crow, ccol = rows / 2, cols / 2
        #reverse masking
        mask = np.ones((rows, cols, 2), np.uint8)
        mask[crow - 10:crow + 10, ccol - 10:ccol + 10] = 0
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        temp = cv2.idft(f_ishift)
        img_out[:, :, i] = cv2.magnitude(temp[:, :, 0], temp[:, :, 1])
    img_out = np.divide(img_out * 255, np.max(img_out))
    return True, img_out


def deconvolution(im):
    gk = cv2.getGaussianKernel(21, 5)
    gk = gk * gk.T
    imf = ft(im, (im.shape[0], im.shape[1]))  # make sure sizes match
    gkf = ft(gk, (im.shape[0], im.shape[1]))  # so we can multiple easily
    imconvf = np.divide(imf, gkf)
    # now for example we can reconstruct the blurred image from its FT
    blurred = ift(imconvf) * 255
    return True, blurred


def Question2():

    #Read in input images
    input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)
    input_image2 = cv2.imread(sys.argv[3],
                              cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    # Low and high pass filter
    succeed1, output_image1 = low_pass_filter(input_image1)
    succeed2, output_image2 = high_pass_filter(input_image1)

    # Deconvolution
    succeed3, output_image3 = deconvolution(input_image2)

    # Write out the result
    output_name1 = sys.argv[4] + "2.jpg"
    output_name2 = sys.argv[4] + "3.jpg"
    output_name3 = sys.argv[4] + "4.jpg"
    cv2.imwrite(output_name1, output_image1)
    cv2.imwrite(output_name2, output_image2)
    cv2.imwrite(output_name3, output_image3)

    return True


# ===================================================
# ===== Question 3: Laplacian pyramid blending ======
# ===================================================


def laplacian_pyramid_blending(A, B):
    n = 5
    print A.shape, B.shape
    if(A.shape[0] < A.shape[1]):
        A = A[:, :A.shape[0]]
        B = B[:A.shape[0], :A.shape[0]]
    else:
        A = A[:A.shape[1],:]
        B = B[:A.shape[1],:A.shape[1]]
    #referred from opencv tutorial
    #Write laplacian pyramid blending codes here
    #img_out = img_in1 # Blending result
    # generate Gaussian pyramid for A
    G = A.copy()
    gpA = [G]
    print G.shape
    for i in xrange(n):
        G = cv2.pyrDown(G)
        print G.shape
        gpA.append(G)
    # generate Gaussian pyramid for B
    G = B.copy()
    gpB = [G]
    print G.shape
    for i in xrange(n):
        G = cv2.pyrDown(G)
        print G.shape
        gpB.append(G)
    # generate Laplacian Pyramid for A
    print "laplaclian pyramids"
    lpA = [gpA[n-1]]
    for i in xrange(n-1, 0, -1):  
        print gpA[i].shape
        GE = cv2.pyrUp(gpA[i])
        print GE.shape
        if(gpA[i-1].shape != GE.shape):
            GE = GE[:gpA[i-1].shape[1],:gpA[i-1].shape[1]]
        L = cv2.subtract(gpA[i - 1], GE)
        lpA.append(L)
    # generate Laplacian Pyramid for B
    lpB = [gpB[n-1]]
    for i in xrange(n-1, 0, -1):
        GE = cv2.pyrUp(gpB[i])
        print GE.shape
        if(gpB[i-1].shape != GE.shape):
            GE = GE[:gpB[i-1].shape[1],:gpB[i-1].shape[1]]
        L = cv2.subtract(gpB[i - 1], GE)
        lpB.append(L)
    # Now add left and right halves of images in each level
    LS = []
    for la, lb in zip(lpA, lpB):
        rows, cols, dpt = la.shape
        ls = np.hstack((la[:, 0:cols / 2], lb[:, cols / 2:]))
        LS.append(ls)
    # now reconstruct
    ls_ = LS[0]
    for i in xrange(1, n):
        ls_ = cv2.pyrUp(ls_)
        print ls_.shape
        print LS[i].shape
        if(LS[i].shape != ls_.shape):
            ls_ = ls_[:LS[i].shape[1],:LS[i].shape[1]]
        ls_ = cv2.add(ls_, LS[i])
    # image with direct connecting each half
    img_out = np.hstack((A[:, :cols/2], B[:, cols/2:]))
    cv2.imwrite('Pyramid_blending2.jpg', ls_)
    return True, ls_


def Question3():

    # Read in input images
    input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)
    input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR)

    # Laplacian pyramid blending
    succeed, output_image = laplacian_pyramid_blending(input_image1,
                                                       input_image2)

    # Write out the result
    output_name = sys.argv[4] + "5.jpg"
    cv2.imwrite(output_name, output_image)

    return True


if __name__ == '__main__':
    question_number = -1

    # Validate the input arguments
    if (len(sys.argv) < 4):
        help_message()
        sys.exit()
    else:
        question_number = int(sys.argv[1])

        if (question_number == 1 and not (len(sys.argv) == 4)):
            help_message()
            sys.exit()
        if (question_number == 2 and not (len(sys.argv) == 5)):
            help_message()
            sys.exit()
        if (question_number == 3 and not (len(sys.argv) == 5)):
            help_message()
            sys.exit()
        if (question_number > 3 or question_number < 1 or len(sys.argv) > 5):
            print("Input parameters out of bound ...")
            sys.exit()

    function_launch = {
        1: Question1,
        2: Question2,
        3: Question3,
    }

    # Call the function
    function_launch[question_number]()
