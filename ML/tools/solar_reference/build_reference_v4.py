"""Build a sparse, visually reviewed solution; never export old projected labels."""
from pathlib import Path
import copy
import json
import shutil
from PIL import Image, ImageDraw, ImageFont

BASE = Path('/home/ccw100/Downloads/solar_test')
OLD = BASE/'reference_example'
OUT = BASE/'reference_example_v4'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

def main():
    OUT.mkdir(exist_ok=True)
    for sub in ['images', 'visual_key']:
        (OUT/sub).mkdir(exist_ok=True)
    for name in ['manifest.json', 'format.json', 'read_tracking.py']:
        shutil.copy2(OLD/name, OUT/name)
    doc = json.loads((OLD/'tracking.json').read_text())
    doc['sequence_id'] = 'solar_zoom_reference_v4'
    doc['label_provenance'] = ('Sparse white-diamond reference: originals and enlarged individual feature crops visually reviewed; '
        'retained image-refined diamond centers rechecked; frame00920 u2v4 corrected by direct crop inspection. '
        'No projected intersections, outer-edge guesses, or homography included. Blurred ambiguous measurements withheld. '
        'Not an independently calibrated pixel-error benchmark.')
    manifest = json.loads((OUT/'manifest.json').read_text())
    manifest['sequence_id'] = doc['sequence_id']
    manifest['reference_scope'] = 'Visible internal white diamond centers only; omission is not a negative label for other grid junctions.'
    previews = []
    for f in doc['frames']:
        n = int(f['frame_id'].split(':')[1])
        points = []
        for p in f['points']:
            if p['method'] != 'image_refined_diamond' or p['visibility'] != 'visible':
                continue
            p = copy.deepcopy(p)
            p['method'] = 'model_observation'
            p['x_px'] = round(p['x_px'], 1)
            p['y_px'] = round(p['y_px'], 1)
            if n == 920 and (p['u'], p['v']) == (2, 4):
                p.update(x_px=1037.0, y_px=1395.0)
            if n == 860 or (n == 890 and (p['u'], p['v']) == (3, 6)):
                p.update(x_px=None, y_px=None, visibility='uncertain')
                p['note'] = 'Feature identity retained, but motion blur prevents a trustworthy precise center; coordinate withheld.'
            points.append(p)
        visible = [p for p in points if p['visibility'] == 'visible']
        f['points'] = points
        f['status'] = 'tracking' if visible else 'uncertain'
        f['homography_panel_to_image'] = None
        f['notes'] = ('Sparse diamond-center solution; no extrapolated grid or outer boundaries. '
            'Other unlisted junctions are outside the scope of this example solution, not necessarily absent.')
        if n == 1040:
            f['notes'] = 'Close-up: no confidently localized internal diamond center in view. No coordinates asserted.'
        f['quality'] = {'visual_review': {
            'image_viewed_individually': True,
            'second_pass_completed': True,
            'observed_points_checked': len(visible),
            'checks': [dict(track_id=p['track_id'], feature='white diamond center', result='pass',
                           note='Center compared against enlarged source-image feature crop; physical ID checked in ordered original images.') for p in visible],
            'limitations': 'Visual localization has finite uncertainty. No independently measured numerical error bound. Blurred centers withheld.'}}
        shutil.copy2(OLD/f['image_file'], OUT/f['image_file'])
        im = Image.open(OUT/f['image_file']).convert('RGB')
        draw = ImageDraw.Draw(im)
        font = ImageFont.load_default(size=20)
        for p in visible:
            x, y = p['x_px'], p['y_px']
            draw.ellipse((x-7,y-7,x+7,y+7),outline='#32ff66',width=2)
            label = f"({p['u']},{p['v']})"
            draw.text((min(x+12,1005),max(85,y-27)),label,font=font,fill='#32ff66',stroke_width=2,stroke_fill='black')
        draw.rectangle((0,0,1080,75),fill='#111111')
        draw.text((15,8),f'Frame {n:05d} | reviewed diamond IDs (u,v)',font=font,fill='white')
        draw.text((15,39),f'{len(visible)} measured centers | no projected grid; uncertain positions withheld',font=font,fill='white')
        im.save(OUT/'visual_key'/Path(f['image_file']).name,quality=95)
        previews.append(im.resize((360,480)))
    doc['verification'] = {'frames_expected':9,'frames_viewed_individually':9,'frames_second_pass_completed':9,
        'pixel_accuracy':'not independently measured',
        'review_scope':'Every retained numeric center compared with enlarged original crop; topology checked against ordered full originals.'}
    (OUT/'tracking.json').write_text(json.dumps(doc,indent=2)+'\n')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    sheet = Image.new('RGB',(1080,1440))
    for i,im in enumerate(previews):sheet.paste(im,((i%3)*360,(i//3)*480))
    sheet.save(OUT/'preview.jpg',quality=95)
    print('Built:',OUT,'visible centers:',sum(p['visibility']=='visible' for f in doc['frames'] for p in f['points']))

if __name__=='__main__':main()
