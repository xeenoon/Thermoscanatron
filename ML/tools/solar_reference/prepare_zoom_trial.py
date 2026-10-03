"""Derive a 100-frame benchmark from packet027 without modifying any packet."""
from pathlib import Path
import json
import zipfile

BASE=Path('/home/ccw100/Downloads/solar_test')
OUT=BASE/'luna_xhigh_zoom_trial'

def main():
    dest=OUT/'input'
    dest.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(BASE/'analysis_027.zip') as z:
        prefix='analysis_027/'
        manifest=json.loads(z.read(prefix+'manifest.json'))
        template=json.loads(z.read(prefix+'tracking_template.json'))
        keep={f['image_file'] for f in manifest['frames'][:100]}
        for entry in z.namelist():
            assert entry.startswith(prefix)
            relative=entry[len(prefix):]
            if relative.startswith('images/') and relative not in keep:continue
            if relative in ['manifest.json','tracking_template.json']:continue
            target=dest/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            raw=z.read(entry)
            if relative=='README.md':
                raw=raw.replace(b'118 original images',b'100 original images')
            target.write_bytes(raw)
        manifest.update(sequence_id='luna_xhigh_zoom_trial_027',source_frame_end=3168,task_frame_count=100,
                        trial_of_packet='analysis_027',frames=manifest['frames'][:100])
        template.update(sequence_id=manifest['sequence_id'],frames=template['frames'][:100])
        (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (dest/'tracking_template.json').write_text(json.dumps(template,indent=2)+'\n')
    (OUT/'trial_setup.json').write_text(json.dumps(dict(model='gpt-5.6-luna',reasoning_effort='xhigh',
        parent_packet='analysis_027.zip',source_frames=[3069,3168],frame_count=100,
        sequence='Broad panel view to tight upper-cell crop; left edge partly cropped in initial frame.',
        purpose='Track original physical grid identities as zoom removes global panel boundaries; do not renumber visible points.',
        solved_example_overlap=False,grading_model='gpt-5.6-sol',
        input_directory=str(dest),output_file=str(OUT/'tracking.json')),indent=2)+'\n')
    print(dest)

if __name__=='__main__':main()
