import sys
import os
import morpheus

os.chdir(os.path.dirname(os.path.abspath(__file__)))

if len(sys.argv) < 2:
    print("Usage: python3 downloader.py <model>")
    print("Supported models: neo-small, neo-medium, neo-large, neox, dialo-small, dialo-medium, dialo-large, rag, blender-small, blender-medium, blender-large, aeona")
    exit()
model = sys.argv[1]

model_downloader = morpheus.GPTNeo(model=model)