from openimages.download import download_images
from pathlib import Path

def init_test(image_qtd=50, image_path="./tests/images"):
    if not Path("./tests/images/person").exists():
        print('\n Downloading images...')
        download_images(dest_dir=image_path, 
                        class_labels=["Person"],
                        exclusions_path="./tests/images/exclusions.txt",
                        limit=image_qtd
                        )
        print('\n Images downloaded!')