"""
Convert an MNIST PNG file to MLflow JSON format.
"""
import sys
import json
import numpy as np
from PIL import Image

def main(path):
    data = np.asarray(Image.open(path))
    data = data.reshape((1, 28 * 28))
    columns = [ f"col_{c}" for c in range(0,data[0].shape[0]) ]
    dct = { "columns" : columns, "data" : [ data[0].tolist()] }
    print(json.dumps(dct,indent=2)+"\n") 

if __name__ == "__main__":
    main(sys.argv[1])
