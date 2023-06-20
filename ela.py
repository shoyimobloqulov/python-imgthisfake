from functions import *


# 1 - rasm
Image.open("datasets/train/real/output.png")
convert_to_ela_image('datasets/train/real/output.png', 90).save("datasets/train/real/output.png")

# 2 - rasm
Image.open('datasets/train/fake/output1.jpg')
convert_to_ela_image('datasets/train/fake/output1.jpg', 90).save("datasets/train/fake/output1.jpg")

