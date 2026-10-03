"""Validate the review reference and create ONE ZIP, not the future mass packets."""
import copy
import importlib.util
import json
from pathlib import Path
import zipfile

ROOT = Path('/home/ccw100/Downloads/solar_test/reference_example')
spec = importlib.util.spec_from_file_location('reader', ROOT / 'read_tracking.py')
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


def main():
    packet = reader.Packet(ROOT)
    labels = reader.load_labels(packet, None)
    print('Reference:', reader.validate(packet, labels, complete=True))
    template = json.loads(packet.read('tracking_template.json'))
    print('Blank template:', reader.validate(packet, template))
    import jsonschema
    schema = json.loads(packet.read('format.json'))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(labels, schema)
    jsonschema.validate(template, schema)
    bad = copy.deepcopy(labels)
    bad['frames'][0], bad['frames'][1] = bad['frames'][1], bad['frames'][0]
    cases = [('reordered frames', bad)]
    bad = copy.deepcopy(labels)
    point = next(p for p in bad['frames'][0]['points'] if p['visibility'] == 'visible')
    point['x_px'] = 1080
    cases.append(('out-of-bounds pixel', bad))
    bad = copy.deepcopy(labels)
    bad['frames'][0]['points'][0]['track_id'] = 'panel0:u99:v99'
    cases.append(('wrong grid identity', bad))
    cases.append(('unfinished template', template))
    for name, malformed in cases:
        try:
            reader.validate(packet, malformed, complete=True)
        except ValueError:
            print('Correctly rejected:', name)
        else:
            raise AssertionError(name)
    packet.close()
    paths = sorted(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    target = ROOT.parent / 'reference_example.zip'
    if target.exists():
        raise FileExistsError(f'Will not overwrite existing review ZIP: {target}')
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in paths:
            z.write(path, path.relative_to(ROOT.parent))
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(paths)
    packed = reader.Packet(target)
    print('ZIP:', reader.validate(packed, reader.load_labels(packed, None), complete=True))
    packed.close()
    streamed = list(reader.iter_frames(target))
    assert len(streamed) == 9
    assert [e['source_frame'] for e, _, _ in streamed] == list(range(800, 1041, 30))
    assert all(raw[:2] == b'\xff\xd8' for _, raw, _ in streamed)
    print(f'Ready: {target} ({target.stat().st_size:,} bytes, {len(paths)} files)')


if __name__ == '__main__':
    main()
