"""Read-only label audit: enlarged original crops, with old points only as locators."""
from pathlib import Path
import json
from PIL import Image, ImageDraw

ROOT = Path('/home/ccw100/Downloads/solar_test/reference_example')
OUT = Path('/home/ccw100/repos/EU-hack/ML/runs/solar_reference_audit')

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = json.loads((ROOT/'tracking.json').read_text())
    for frame in labels['frames']:
        points = [p for p in frame['points'] if p['visibility']=='visible' and p['method']=='image_refined_diamond']
        if not points:
            continue
        original = Image.open(ROOT/frame['image_file']).convert('RGB')
        canvas = Image.new('RGB', (1200, ((len(points)+2)//3)*330), '#181818')
        draw = ImageDraw.Draw(canvas)
        for i, p in enumerate(points):
            x, y = p['x_px'], p['y_px']
            radius = 35 if int(frame['frame_id'].split(':')[1])<890 else 55
            x0, y0 = round(x)-radius, round(y)-radius
            crop = original.crop((x0, y0, x0+2*radius, y0+2*radius)).resize((280,280))
            ox, oy = (i%3)*400, (i//3)*330
            canvas.paste(crop, (ox, oy+35))
            # Tick marks outside the center preserve visibility of the actual feature.
            cx, cy = ox+(x-x0)*280/(2*radius), oy+35+(y-y0)*280/(2*radius)
            draw.line((cx-25,cy,cx-10,cy), fill='red', width=1)
            draw.line((cx+10,cy,cx+25,cy), fill='red', width=1)
            draw.line((cx,cy-25,cx,cy-10), fill='red', width=1)
            draw.line((cx,cy+10,cx,cy+25), fill='red', width=1)
            draw.text((ox+3,oy+3),f"u{p['u']}v{p['v']} old=({x:.2f},{y:.2f})",fill='white')
            draw.text((ox+3,oy+17),f'crop x0={x0} y0={y0} side={2*radius}px',fill='white')
        dest=OUT/(frame['frame_id'].split(':')[1]+'.jpg')
        canvas.save(dest, quality=95)
        print(dest)

if __name__=='__main__':main()
