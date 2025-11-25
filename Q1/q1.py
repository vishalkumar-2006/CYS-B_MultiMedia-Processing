from PIL import Image

def grayscale(image):
    width, height = image.size

    # Create a new grayscale image
    quant_img = Image.new("L", (width, height))

    # Iterate through each pixel
    for y in range(height):
        for x in range(width):
            r, g, b = image.getpixel((x, y))
            gray = int((max(r, g, b) + min(r, g, b)) / 2)  # Lightness method
            quant_img.putpixel((x, y), gray)

    return quant_img    

def apply_median_cut(image,colors=16):
    # Quantize using Median Cut (method = 0)
    quant_img = image.quantize(colors, method=Image.MEDIANCUT)
    return quant_img

def apply_octree(image,colors=16):
    # Quantize using Octree (method = 2)
    quant_img = image.quantize(colors, method=Image.FASTOCTREE)
    return quant_img

# Load source image
input=Image.open("input.jpg")

#Median Cut 
median_img = apply_median_cut(input)
median_rgb = median_img.convert("RGB")
#median_img.save("without.png") #without converting you can save in png or gif format
median_rgb.save("q1_new_median.jpg")
#Octree
octree_img = apply_octree(input)
octree_rgb = octree_img.convert("RGB")
octree_rgb.save("q1_new_octree.jpg")
#Grayscale
gray_img=grayscale(input)
gray_img.save("q1_grayscale.jpg")
print("quantization completed")
