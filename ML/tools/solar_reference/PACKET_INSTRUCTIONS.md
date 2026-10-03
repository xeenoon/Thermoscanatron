# Ordered solar-panel tracking task

read every image in this folder in the order that they are named and place the tracking data as defined by format.json

**view each image seperately and analyse. Prove your work passes before handing back**

## Your input and output

Your task is ONLY the 118 original images in the root `images/` folder, in ascending filename order, matching the root `manifest.json`. This packet is one consecutive section of the latest recording. Preserve every frame and its source identity, including no-panel, uncertain, or lost frames. Do not reorder, silently skip, or add frames.

`format.json` defines the output format. `tracking_template.json` is an EMPTY structure with your frame identities populated. There is intentionally NO root `tracking.json` answer. Create it from your images.

Return **only your newly created `tracking.json`** matching `format.json`. Do **NOT** draw labels, generate overlays, modify images, or return annotated images. Keep the manifest's sequence and recording IDs unchanged.

## Separate context and solved example

- `context/` contains full-panel views selected from an individually verified set for this packet's start and end. Their actual source frame numbers, signed offsets and temporal relationship are in `context/manifest.json`. They are NOT the actual first/last task frames and are NOT additional frames to label. A context image may be temporally distant, come from before OR after the task, and be reused in several packets.
- `example/` is a separate solved wide-to-close-up sequence. Read its `README.md`, original images, `tracking.json`, and `visual_key/` to understand coordinates and stable grid identity. That JSON answers the EXAMPLE ONLY. Never submit it as your packet's output. Always match `frame_id` and `image_file` to the root manifest.
- The example's solution is deliberately sparse: reviewed internal white-diamond centers, no extrapolated full grid. Its visual key illustrates these points, not an extra task sequence. Uncertain positions are withheld. The example is a reviewed annotation, not a calibrated guarantee of zero pixel error.
- Context and example are aids for offline labeling. Future context must not be used as if it were causal input when evaluating an online inference tracker. Recordings may contain the example frames in their ordinary chronological section as well; they remain task frames when listed in the root manifest.

## Physical grid and pixel coordinates

- Panel: 4 cell columns × 9 cell rows. Boundary intersections use `u=0..4` left-to-right and `v=0..9` top-to-bottom on the physical panel. The origin is the upper-left of the cell area, inside the metal frame.
- White diamonds occur at internal columns `u=1,2,3` and rows `v=2,4,6,8`. Thin metallic busbars inside cells are NOT column boundaries. Odd row boundaries have ordinary gaps.
- Coordinates use ORIGINAL 1080×1440 pixels: x right, y down, `(0,0)` at the upper-left image corner. Convert cropped/resized coordinates back before returning them.
- `panel0:u1:v2` must always identify the same physical junction. As the camera zooms in, do NOT renumber the remaining visible cells or points from zero. Preserve physical identity even as panel edges leave view. Cell names `Ccolumn Rrow`, when shown in the example, are zero-based cells, not boundary intersections.
- Chamfer tips are not intersections of extended cell edges. Extrapolated corners are inferred, not directly observed.

## Required individual review

1. Open EACH original task image separately, in order. Inspect at native resolution or using unannotated detail crops where needed. A contact sheet, existing predictions, or an inferred plane do not substitute for seeing that image. If an image cannot be accessed, state the limitation; do not claim inspection.
2. Select confidently visible grid junctions, preferably diamond centers, independently from each image. Use `method: "model_observation"` for directly selected points. Track physical identities through the sequence. Do not force a complete 50-point grid or omit clear junctions merely to avoid checking them.
3. Use `"inferred"` or `"tracked_homography"` for geometric estimates; never present them as observations. Being inside image bounds does not prove a projected feature is visible. Do not report a homography unless you actually estimated and checked it.
4. If position or identity is unreliable, use `visibility: "uncertain"` with null coordinates, or omit that point. Out-of-frame and occluded points also require null coordinates with the appropriate visibility. Keep every frame record; explicit uncertain/lost/no-panel statuses are valid.
5. Make a SECOND visual pass over each image and check EVERY point reported as observed against the actual feature. Correct misplaced points; verify identities across adjacent frames, especially when zooming, tilting or losing panel edges. Recheck ambiguity instead of copying a prior visibility decision.
6. Validate against `format.json`, then run the local semantic validator when execution is available. Fix failures and rerun. A schema pass checks structure, NOT pixel accuracy. Small flow/homography residuals are not an independent accuracy test.

## Verification evidence inside the returned JSON

For each frame add `quality.visual_review` with `image_viewed_individually`, `second_pass_completed`, `observed_points_checked`, `limitations`, and `checks`. Each check contains `track_id`, the observed `feature`, `result` (`"pass"` or `"uncertain"`), and a brief factual `note`. Include at least three spatially spread observed points when available, or every observed point if fewer than three exist. Explain no-point frames. These sample records do not replace checking all observed points.

Add a top-level `verification` object with `frames_expected`, `frames_viewed_individually`, `frames_second_pass_completed`, `schema_validation`, `semantic_validation`, and `pixel_accuracy`. Record actual tools/commands and results, or `"not_run"` with a reason. If pixel accuracy is not independently measured, say `"not independently measured"`. These extra fields are permitted by the schema.

Provide concise observable evidence, not private reasoning. Never invent inspection, validation, timing or error measurements. Self-reported checks are subject to independent image review. There is no arbitrary minimum thinking time: finish the checks before handing back.

## Local validation and ordered reading

After placing your returned file beside this README:

```bash
python3 read_tracking.py validate . --labels tracking.json --complete
python3 read_tracking.py stream . --labels tracking.json
```

The validator checks frame identity/order, source-image hashes, bounds, visibility and grid IDs. `--complete` rejects still-unlabeled frames. The stream command returns one JSON object per task frame, joining metadata to labels. It never includes example/context images in the task stream.

The reader also accepts a ZIP path and a separate `--labels` path. For local inference, `iter_frames(packet_path, labels_path)` yields `(frame_metadata, original_jpeg_bytes, annotation)` in order. Before labeling, pass the blank template as `labels_path`. Keep state across sections of the same recording rather than resetting merely because the packet filename changes.
