# EU-hack

## Browser webcam demo

A Python Gradio/FastRTC server runs the solar-panel tracker and hand-segmentation models while a laptop or phone browser streams the camera.

```sh
cd ML
uv sync --extra cpu --extra web
uv run --extra cpu --extra web segkit-panel-web
```

Open <http://127.0.0.1:7860>. See [ML/README.md](ML/README.md#browser-webcam-demo) for model overrides.

## 3D housing & structural studies

CAD, STL and print files for the phone + thermal camera housing are in [3d/](3d/). The raw SolidWorks simulation output files are gitignored because they're large and can be regenerated.

We ran two SolidWorks static studies. Each one applies 100 N to opposite sides of the casing, which simulates someone forcing the sliding attachment open or shut.

| Study | Fixed points | Max stress (von Mises) | Max displacement | Max strain |
|---|---|---|---|---|
| 1 | Large section of the phone attachment | 13.18 MPa | 0.0486 mm | 0.206% |
| 2 | Only the phone-gripping hooks (more realistic), with revised CAD | **1.25 MPa** | **0.0149 mm** | **0.026%** |

Study 2 is the more realistic support case and uses the revised design. Compared with study 1 it shows **~90% less stress, ~69% less displacement and ~87% less strain**, which means the revised housing is much stiffer and spreads the load better.

What the numbers mean:
- **Von Mises stress**: how close the material is to yielding (permanently bending or breaking).
- **Displacement**: how far the structure moves under the load.
- **Strain**: how much the material stretches or squashes locally.
