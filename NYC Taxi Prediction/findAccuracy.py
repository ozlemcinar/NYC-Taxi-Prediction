from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
import cv2

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err
def compare_images(imageA, imageB, title):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = measure.compare_ssim(imageA, imageB)
    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap = plt.cm.gray)
    plt.axis("off")
    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap = plt.cm.gray)
    plt.axis("off")
    # show the images
    plt.show()


AprilL = cv2.imread("04-lucrative.png")
#contrast = cv2.imread("desktop/04-lucrative.png")
TestDataL = cv2.imread("test-lucrative.png")
# convert the images to grayscale
AprilL = cv2.cvtColor(AprilL, cv2.COLOR_BGR2GRAY)
TestDataL = cv2.cvtColor(TestDataL, cv2.COLOR_BGR2GRAY)
# initialize the figure
fig = plt.figure("Images")
images = ("AprilL", AprilL), ("TestDataL", TestDataL)
# loop over the images
for (i, (name, image)) in enumerate(images):
    # show the image
    ax = fig.add_subplot(1, 3, i + 1)
    ax.set_title(name)
    plt.imshow(image, cmap = plt.cm.gray)
    plt.axis("off")
    
AprilP = cv2.imread("test-pickup.png")
#contrast = cv2.imread("desktop/04-lucrative.png")
TestDataP = cv2.imread("04-pickup.png")
# convert the images to grayscale
AprilP = cv2.cvtColor(AprilP, cv2.COLOR_BGR2GRAY)
TestDataP = cv2.cvtColor(TestDataP, cv2.COLOR_BGR2GRAY)
# initialize the figure
fig = plt.figure("Images")
images = ("AprilP", AprilP), ("TestDataP", TestDataP)
# loop over the images
for (i, (name, image)) in enumerate(images):
    # show the image
    ax = fig.add_subplot(1, 3, i + 1)
    ax.set_title(name)
    plt.imshow(image, cmap = plt.cm.gray)
    plt.axis("off")
    
    
    
# show the figure
plt.show()
# compare the images

compare_images(AprilL, TestDataL, "April vs. TestData for lucrative")
compare_images(AprilP, TestDataP, "April vs. TestData for pickup")



