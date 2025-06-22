from flask import Flask, request, send_file
from PIL import Image, ImageDraw
import os
import tempfile

app = Flask(__name__)

@app.route('/mix', methods=['POST'])
def mix():
    giraffe = request.files['giraffe']
    human = request.files['human']
    mixing_level = int(request.form['mixing_level'])
    giraffe_seed = int(request.form.get('giraffe_seed', 0))
    human_seed = int(request.form.get('human_seed', 0))

    temp_dir = tempfile.mkdtemp()
    giraffe_path = os.path.join(temp_dir, 'giraffe.jpg')
    human_path = os.path.join(temp_dir, 'human.jpg')
    output_path = os.path.join(temp_dir, 'mixed.png')

    giraffe.save(giraffe_path)
    human.save(human_path)

    # 실제 GAN 모델로 대체 가능
    img = Image.new('RGB', (512, 512), (200 + mixing_level, 150, 200 - mixing_level))
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), f'Mix: {mixing_level}%', fill=(0, 0, 0))
    draw.text((20, 50), f'Seed G: {giraffe_seed}', fill=(0, 0, 0))
    draw.text((20, 80), f'Seed H: {human_seed}', fill=(0, 0, 0))
    img.save(output_path)

    return send_file(output_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(port=8000)
