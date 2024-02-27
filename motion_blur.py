from motionblur.motionblur import *
import PIL

# Initialise Kernel
kernel = Kernel(size=(31, 31), intensity=0.7)

# Display kernel
kernel.displayKernel()

# Get kernel as numpy array
kernel.kernelMatrix

# Save kernel as image. (Do not show kernel, just save.)
kernel.displayKernel(save_to="./datasets/NIND_motion_kernel/13_clean_kernel1.png", show=False)

# load image or get image path
# image1_path = "./datasets/pic3.png"
image2 = PIL.Image.open("datasets/NIND/13_clean.png")

# apply motion blur (returns PIL.Image instance of blurred image)
# blurred1 = kernel.applyTo(image1_path)

blurred2 = kernel.applyTo(image2)

# if you need the dimension of the blurred image to be the same
# as the original image, pass `keep_image_dim=True`
blurred_same = kernel.applyTo(image2, keep_image_dim=True)

# show result
blurred_same.show()

# or save to file
blurred_same.save("./datasets/NIND_blur/13_clean_blur1.png")