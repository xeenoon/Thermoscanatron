# Thermoscanatron

## Device components

- MLX90640: $32\times24$ thermal pixels.
- ESP32-S3: factory-calibrated temperature conversion and USB streaming.
- Android app: local ExecuTorch inference, tracking and temperature overlays; the phone also powers the sensor.
- SolidWorks housing: a 3D-printable phone attachment. The revised 100 N simulation gave 1.25 MPa maximum stress and 0.0149 mm displacement.

## Scanner mapping: hand recognition and calibration

HandSegNet uses a MobileNetV3-Small encoder with a U-Net-style decoder and presence head, approximately 1.70 million parameters, at $256\times256$ input resolution.

Our offline labeller combines SegFormer-B2 human parsing with BiRefNet foreground masks and MediaPipe hand landmarks. Landmark-guided cropping removes sleeves; clothing predictions veto false skin regions. These automatically generated masks transfer heavier models' predictions into the compact phone model through supervised pseudo-label training.

The hand estimates the **relative camera pose**. Approximating an open hand as a front-facing plane of area $A_h=130\,\mathrm{cm}^2$, its mask area $a$ gives depth:

```math
Z=f\sqrt{A_h/a},\qquad
P=ZK^{-1}[u,v,1]^T.
```

Here $f$ is focal length in pixels, $K$ is the RGB intrinsic matrix and $(u,v)$ is the hand centroid. For thermal-camera centre $c$ and rotation $R$:

```math
P_T=R^T(P-c),\qquad
R=R_y(\psi)R_x(\theta)R_z(\phi).
```

We fit yaw $\psi$, pitch $\theta$ and roll $\phi$, plus translation, by robust least-squares matching projected hand centroids to thermal warm-blob centroids. The thermal lens uses an equidistant projection: image radius is proportional to ray angle. We then refine pose and lens scale by maximising correlation between projected hand coverage and thermal warmth. Near/far hand motion provides parallax; timestamp alignment compensates sensor delay.

**Training and labelling:** Hand/skin model training totalled **1.84 hours across 13 logged runs**, including validation. The final run used 5,288 training frames, 1,356 validation frames and 2,000 additional negatives. A complete hand-labelling runtime was not retained.

## Solar scanning: recognition, geometry and temperature

PanelNet uses a MobileNetV3-Small encoder and U-Net-style decoder, approximately 1.69 million parameters. The phone uses 192-pixel inputs for fast inference and a 384-pixel recovery model. Outputs include cell masks, gridlines, presence and periodic cell coordinates:

```math
(\sin 2\pi u,\cos 2\pi u,\sin\pi v,\cos\pi v).
```

Column phase repeats every cell; row phase repeats every two rows to encode the panel's alternating diamond pattern. Temporal tracking and gyroscope motion preserve integer cell identities through close-ups.

Our offline labeller combines KLT optical flow, SIFT rematching, gridline/diamond constraints and forward/backward tracking. A hidden Markov model with Viterbi decoding resolves cell numbering across the recording. Its homographies generate dense training targets for the phone network: geometric pseudo-labelling, rather than direct neural teacher/student distillation.

The panel-to-image homography gives its plane pose:

```math
H\sim K[w r_1,\;h r_2,\;t],\qquad n=r_1\times r_2,
```

where $w,h$ are cell dimensions, $r_1,r_2$ are panel axes and $t$ is its origin. A calibrated thermal ray $d$ intersects that plane at:

```math
X=c+\frac{n^T(t-c)}{n^Td}d.
```

Projecting $X-t$ onto the panel axes identifies the cell. Nine rays sample each thermal pixel's footprint; a cell reading accepts only pixels whose entire sampled footprint lies within that cell, excluding boundaries and background.

For accepted thermal pixels $S_i$ in cell $i$ and panel pixels $S_P$:

```math
T_i=\mathrm{median}\{T_p:p\in S_i\},\qquad
\bar T_P=\frac{1}{|S_P|}\sum_{p\in S_P}T_p,\qquad
\Delta T_i=T_i-\bar T_P.
```

We define a **hotspot** as $\Delta T_i>5^\circ\mathrm C$. The implementation flags $|\Delta T_i|>5^\circ\mathrm C$, including cold anomalies. This is our prototype threshold. Readings are apparent surface temperatures with emissivity set to 0.95.

**Training and labelling:** Solar model training totalled **4.23 hours across 22 logged runs**, including validation. One training stage used 8,695 panel frames plus 1,291 negatives. Solar labelling totalled **4.56 logged hours across 74 completed runs on 3–4 October**. These cumulative development totals exclude unlogged work; interrupted labelling runs are additional.

## Demo result

The suspect panel measured 19 V, 4 V below its 23 V baseline; the healthy panel measured 23.7 V, 0.7 V above baseline. Together with the observed hotspots, the voltage deficit supports identifying the demo panel as faulty.
