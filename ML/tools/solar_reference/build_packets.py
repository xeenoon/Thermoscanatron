"""Export 100 disjoint 118-frame packets, preserving the complete extracted order."""
import concurrent.futures
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile
from PIL import Image
import jsonschema

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parents[1]/'data/panel/20261003_220702_906/frames'
OUT=Path('/home/ccw100/Downloads/solar_test')
EXAMPLE=OUT/'reference_example_v4'
ANCHORS=[800,830,11701]
RECORDING='20261003_220702_906'

def encoded(obj):return (json.dumps(obj,indent=2,allow_nan=False)+'\n').encode()
def digest(raw):return hashlib.sha256(raw).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--replace-verified-build',action='store_true')
    args=parser.parse_args()
    prior={}
    if args.replace_verified_build:
        previous=json.loads((OUT/'packets_index.json').read_text())
        assert previous['recording_id']==RECORDING and previous['total_packets']==100
        prior={r['zip_file']:r['sha256'] for r in previous['packets']}
    files=sorted(SOURCE.glob('*.jpg'))
    assert [p.name for p in files]==[f'{n:05d}.jpg' for n in range(1,11801)]
    schema=json.loads((EXAMPLE/'format.json').read_text())
    schema_validator=jsonschema.Draft202012Validator(schema)
    solution=json.loads((EXAMPLE/'tracking.json').read_text())
    schema_validator.validate(solution)
    spec=importlib.util.spec_from_file_location('reader',EXAMPLE/'read_tracking.py')
    reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
    p=reader.Packet(EXAMPLE)
    reference_check=reader.validate(p,solution,complete=True);p.close()
    docs={
        'README.md':(HERE/'PACKET_INSTRUCTIONS.md').read_bytes(),
        'format.json':(EXAMPLE/'format.json').read_bytes(),
        'read_tracking.py':(EXAMPLE/'read_tracking.py').read_bytes(),
    }
    example={f'example/{p.relative_to(EXAMPLE).as_posix()}':p.read_bytes()
             for p in sorted(EXAMPLE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    audit=HERE.parents[1]/'runs/solar_packet_review/AUDIT_V4.md'
    example['example/INDEPENDENT_VISUAL_AUDIT.md']=audit.read_bytes()
    anchor_bytes={n:(SOURCE/f'{n:05d}.jpg').read_bytes() for n in ANCHORS}
    all_frames=[]
    for n,path in enumerate(files,1):
        with Image.open(path) as im:assert im.size==(1080,1440)
        all_frames.append(dict(source_frame=n,frame_id=f'{RECORDING}:{n:05d}',width=1080,height=1440,
                               timestamp_seconds=round((n-1)*1001/30000,6),sha256=digest(path.read_bytes())))
    print('Source preflight: 11800 ordered original JPEGs; example validated.',flush=True)

    def build(k):
        start=(k-1)*118+1;end=k*118;name=f'analysis_{k:03d}';prefix=name+'/'
        frames=[]
        for index,source in enumerate(all_frames[start-1:end]):
            frames.append(dict(source,sequence_index=index,recording_sequence_index=source['source_frame']-1,
                               image_file=f"images/{index+1:06d}_frame_{source['source_frame']:05d}.jpg"))
        manifest=dict(schema_version=1,sequence_id=name,recording_id=RECORDING,packet_index=k,packet_count=100,
            source_frame_start=start,source_frame_end=end,task_frame_count=118,
            order_policy='Consecutive original extracted frames; no omitted, repeated or reordered task frames across packets.',
            timestamp_provenance='Nominal CFR extracted-image timeline at30000/1001FPS, not exact video PTS.',
            context_is_separate=True,example_is_separate=True,frames=frames)
        template=dict(schema_version=1,sequence_id=name,recording_id=RECORDING,
            panel_spec=dict(cols=4,rows=9,diamond_rows=[2,4,6,8]),
            label_provenance='Unlabeled task template; inspect each original image and produce your own labels.',
            frames=[dict(frame_id=f['frame_id'],image_file=f['image_file'],status='unlabeled',points=[]) for f in frames])
        schema_validator.validate(template)
        context=[];context_files={}
        for role,boundary in [('start',start),('end',end)]:
            if role=='start':choices=[n for n in ANCHORS if n<=boundary];a=max(choices) if choices else min(ANCHORS)
            else:choices=[n for n in ANCHORS if n>=boundary];a=min(choices) if choices else max(ANCHORS)
            relative=f'context/{role}_full_panel_frame_{a:05d}.jpg'
            context_files[relative]=anchor_bytes[a]
            context.append(dict(role=f'{role}_context_only',image_file=relative,source_frame=a,
                frame_id=f'{RECORDING}:{a:05d}',relative_to_task_boundary_frames=a-boundary,
                temporal_relation='before' if a<boundary else 'after' if a>boundary else 'same',
                full_cell_area_visible=True,cell_area_unoccluded=True,
                review='Four active-cell-area corners individually rechecked in original image by root; hand touches outer frame only. Context can be temporally distant.',sha256=digest(anchor_bytes[a])))
        context_manifest=dict(recording_id=RECORDING,not_task_frames=True,
            warning='Context may be future or reused. Do not treat it as chronological task frames or causal inference input.',frames=context)
        entries={**docs,**example,**context_files,'manifest.json':encoded(manifest),
                 'tracking_template.json':encoded(template),'context/manifest.json':encoded(context_manifest)}
        target=OUT/(name+'.zip');partial=OUT/(name+'.zip.partial')
        if target.exists():
            if not args.replace_verified_build or prior.get(target.name)!=digest(target.read_bytes()):
                raise FileExistsError(f'Refusing to overwrite unverified or externally changed {target}')
        with zipfile.ZipFile(partial,'x',compression=zipfile.ZIP_STORED) as z:
            for rel,raw in entries.items():z.writestr(prefix+rel,raw)
            for f in frames:
                raw=(SOURCE/f"{f['source_frame']:05d}.jpg").read_bytes()
                assert digest(raw)==f['sha256'], 'Source changed during export'
                z.writestr(prefix+f['image_file'],raw)
        with zipfile.ZipFile(partial) as z:
            assert len(z.namelist())==len(entries)+118
            assert prefix+'tracking.json' not in z.namelist()
            for rel,raw in entries.items():assert z.read(prefix+rel)==raw
            for f in frames:assert digest(z.read(prefix+f['image_file']))==f['sha256']
        # Exercise the actual shipped reader on every archive, not just ad hoc checks.
        p=reader.Packet(partial)
        result=reader.validate(p,template);p.close()
        assert result['frames']==118 and result['points']==0 and result['unlabeled_frames']==118
        partial.replace(target)
        return dict(packet_index=k,zip_file=target.name,sequence_id=name,source_frame_start=start,source_frame_end=end,
                    task_frame_count=118,bytes=target.stat().st_size,sha256=digest(target.read_bytes()),
                    context_source_frames=[c['source_frame'] for c in context],validation=result)

    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for future in concurrent.futures.as_completed([pool.submit(build,k) for k in range(1,101)]):
            results.append(future.result())
            if len(results)%10==0:print(f'Built and verified {len(results)}/100 ZIPs',flush=True)
    results.sort(key=lambda r:r['packet_index'])
    covered=[n for r in results for n in range(r['source_frame_start'],r['source_frame_end']+1)]
    assert covered==list(range(1,11801))
    index=dict(schema_version=1,recording_id=RECORDING,total_packets=100,frames_per_packet=118,
               task_frames_total=11800,coverage='All extracted source frames exactly once in recording order.',
               example_frames_per_packet=9,context_images_per_packet=2,packets=results)
    (OUT/'packets_index.json').write_bytes(encoded(index))
    (OUT/'source_frame_manifest.json').write_bytes(encoded(dict(recording_id=RECORDING,frames=all_frames)))
    report=dict(valid=True,packets=100,unique_task_frames=11800,missing_frames=0,duplicate_task_frames=0,
        root_prefilled_tracking_files=0,task_frames_per_packet=118,source_image_hashes_verified=True,
        schema_and_shipped_reader_validated=True,example_identical_in_all_packets=True,reference_validation=reference_check,
        context_and_example_excluded_from_task_stream=True,context_anchor_source_frames=ANCHORS,
        context_corner_review='Root independently checked four corners on final anchors; rejected earlier cropped candidates.',
        total_zip_bytes=sum(r['bytes'] for r in results))
    (OUT/'build_validation.json').write_bytes(encoded(report))
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
