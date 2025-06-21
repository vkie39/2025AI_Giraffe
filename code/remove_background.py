import os
from PIL import Image #pip install Image
from rembg import remove #pip install pillow imagehash
                         #pip install onnxruntime

def process_images(input_dir='human_face', output_dir='after_rmbackground_humanFace'):
    os.makedirs(output_dir, exist_ok=True)
    image_extensions = ('.jpg', '.jpeg', '.png')

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(image_extensions):
            in_path = os.path.join(input_dir, filename)
            out_path = os.path.join(output_dir, filename)
            try:
                with Image.open(in_path) as img:
                    img = img.convert('RGBA')
                    result = remove(img)
                    # Create white background image
                    bg = Image.new('RGB', result.size, (255, 255, 255))
                    bg.paste(result, mask=result.split()[3])
                    bg.save(out_path)
            except Exception as e:
                print(f'Error processing {filename}: {e}')


if __name__ == '__main__':
    process_images()