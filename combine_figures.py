from PIL import Image
import os
import subprocess


def ensure_images():
    imgs = {
        'schematic': 'neuron_diagram.png',
        'fit': 'f1_poly_fit.png',
        'metrics': 'f1_poly_metrics.png',
        'learning': 'f1_learning_curves.png'
    }
    missing = [v for v in imgs.values() if not os.path.exists(v)]
    if missing:
        print('Missing images:', missing)
        # generate schematic
        if not os.path.exists(imgs['schematic']):
            print('Generating schematic...')
            subprocess.run(['python', 'diagram_neuron.py'], check=True)
        # generate training visuals (this runs training if needed)
        if not os.path.exists(imgs['fit']) or not os.path.exists(imgs['metrics']) or not os.path.exists(imgs['learning']):
            print('Generating training plots (may take a moment)...')
            subprocess.run(['python', 'main.py'], check=True)


def combine_side_by_side(schematic_path='neuron_diagram.png', right_images=None, out_fname='combined_figure.png'):
    if right_images is None:
        right_images = ['f1_poly_fit.png', 'f1_poly_metrics.png', 'f1_learning_curves.png']

    left = Image.open(schematic_path).convert('RGBA')
    rights = [Image.open(p).convert('RGBA') for p in right_images]

    # target widths
    left_w = 600
    right_w = 600

    # Resize left to left_w
    left_h = int(left_h := left.size[1] * (left_w / left.size[0]))
    left = left.resize((left_w, left_h), Image.LANCZOS)

    # Resize rights to right_w keeping aspect ratio
    rights_resized = []
    for im in rights:
        h = int(im.size[1] * (right_w / im.size[0]))
        rights_resized.append(im.resize((right_w, h), Image.LANCZOS))

    # compute total right height
    total_right_h = sum(im.size[1] for im in rights_resized)

    # final canvas size: width = left_w + right_w, height = max(left_h, total_right_h)
    canvas_h = max(left_h, total_right_h)
    canvas_w = left_w + right_w

    canvas = Image.new('RGBA', (canvas_w, canvas_h), (255, 255, 255, 255))

    # paste left centered vertically
    left_y = (canvas_h - left_h) // 2
    canvas.paste(left, (0, left_y), left)

    # paste rights stacked at right column, top aligned
    y = 0
    for im in rights_resized:
        canvas.paste(im, (left_w, y), im)
        y += im.size[1]

    canvas.convert('RGB').save(out_fname, dpi=(150, 150))
    print(f'Saved combined figure to {out_fname}')
    # overlay input data size if metrics file exists
    try:
        import json
        if os.path.exists('f1_poly_metrics.json'):
            with open('f1_poly_metrics.json', 'r') as fh:
                metrics = json.load(fh)
            n_train = metrics.get('n_train')
            if n_train is not None:
                # reopen and draw text
                from PIL import ImageDraw, ImageFont
                img = Image.open(out_fname).convert('RGBA')
                draw = ImageDraw.Draw(img)
                # try a larger bold font for visibility
                try:
                    font = ImageFont.truetype('DejaVuSans-Bold.ttf', 22)
                except Exception:
                    try:
                        font = ImageFont.truetype('DejaVuSans.ttf', 22)
                    except Exception:
                        font = None
                text = f'Input data size (n_train): {n_train}'
                # place at top-right with padding
                w, h = img.size
                tw, th = draw.textsize(text, font=font)
                pad_x = 14
                pad_y = 8
                rect_coords = [(w - tw - pad_x, 6), (w - 6, 6 + th + pad_y)]
                draw.rectangle(rect_coords, fill=(255, 255, 255, 230))
                draw.text((w - tw - pad_x + 4, 8), text, fill='black', font=font)
                img.convert('RGB').save(out_fname, dpi=(150,150))
                print(f'Overlayed input data size onto {out_fname}')
    except Exception:
        pass
    # Also save an SVG wrapper that embeds the PNG as a data URI so you have an SVG file
    try:
        import base64
        from io import BytesIO

        with open(out_fname, 'rb') as f:
            png_data = f.read()
        b64 = base64.b64encode(png_data).decode('ascii')
        svg_name = os.path.splitext(out_fname)[0] + '.svg'
        # Determine image dimensions
        with Image.open(out_fname) as im:
            w, h = im.size

        svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <image width="{w}" height="{h}" href="data:image/png;base64,{b64}"/>
</svg>
'''
        with open(svg_name, 'w') as fh:
            fh.write(svg_content)
        print(f'Saved combined figure SVG to {svg_name}')
    except Exception:
        pass


if __name__ == '__main__':
    ensure_images()
    combine_side_by_side()
