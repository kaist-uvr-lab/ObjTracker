import numpy as np
import glob
from PIL import Image

def dataProcessor(filePath, outputPath):
    origImages = glob.glob(filePath + '/*.jpg')
    segImages = glob.glob(filePath + '/*_seg.png')
    
    