from PIL import Image

def grayscale(image):
    width, height = image.size

    quant_img = Image.new("L", (width, height))

    for y in range(height):
        for x in range(width):
            r, g, b = image.getpixel((x, y))
            gray = int((max(r, g, b) + min(r, g, b)) / 2)  
            quant_img.putpixel((x, y), gray)

    return quant_img    

def apply_median_cut(image,colors=16):
    quant_img = image.quantize(colors, method=Image.MEDIANCUT)
    return quant_img

def apply_octree(image,colors=16):
    quant_img = image.quantize(colors, method=Image.FASTOCTREE)
    return quant_img

input=Image.open("input.jpg")

median_img = apply_median_cut(input)
median_rgb = median_img.convert("RGB")
median_rgb.save("q1_new_median.jpg")
octree_img = apply_octree(input)
octree_rgb = octree_img.convert("RGB")
octree_rgb.save("q1_new_octree.jpg")
gray_img=grayscale(input)
gray_img.save("q1_grayscale.jpg")

print("quantization completed")
