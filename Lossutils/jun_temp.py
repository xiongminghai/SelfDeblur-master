from PIL import Image
I = Image.open('D:\chenxiaojun\DIP-bsd-premodel-test\data\denoising\different1.png')
I.show()
L = I.convert('L')
L.show()
L.save('D:\chenxiaojun\DIP-bsd-premodel-test\data\denoising\pics5\gray_different1.png')