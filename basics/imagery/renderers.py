import base64
from io import BytesIO
from typing import List, Union

from flytekit.types.file import FlyteFile
from PIL import Image


class ImageRenderer:
    """Converts a FlyteFile or PIL.Image.Image object to an HTML string with the image data
    represented as a base64-encoded string.
    """

    def to_html(cls, image_src: Union[FlyteFile, Image.Image]) -> str:
        img = cls._get_image_object(image_src)
        return cls._image_to_html_string(img)

    @staticmethod
    def _get_image_object(image_src: Union[FlyteFile, Image.Image]) -> Image.Image:
        if isinstance(image_src, FlyteFile):
            local_path = image_src.download()
            return Image.open(local_path)
        elif isinstance(image_src, Image.Image):
            return image_src
        else:
            raise ValueError("Unsupported image source type")

    @staticmethod
    def _image_to_html_string(img: Image.Image) -> str:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        return f'<img src="data:image/png;base64,{img_base64}" alt="Rendered Image" />'


class FancyGrid:
    def to_html(self, html_elements: List[str]) -> str:
        grid_items = "\n".join([f'<div class="grid-item">{element}</div>' for element in html_elements])
        return f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                .grid-container {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                    grid-gap: 10px;
                    padding: 10px;
                }}
                .grid-item {{
                    padding: 10px;
                    background-color: #f1f1f1;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    overflow: auto;
                    max-height: 300px; /* Adjust this value to set the maximum height of the grid item */
                }}
                .grid-item img {{
                    display: block;
                    max-width: 100%;
                    max-height: 100%;
                }}
            </style>
        </head>
        <body>
            <div class="grid-container">
                {grid_items}
            </div>
        </body>
        </html>
        '''


