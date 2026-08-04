# Phoenix QoS Clock-Ratio Discovery

## Verdict

The context-QoS path is usable for Phoenix clock selection, but the current
platform state exposes only two distinct reported H/MP-NPU ratios. The
three-point firmware/array clock model is therefore **insufficiently
identified** and no production scheduler change is authorized.

## Pinned run

- firmware: `amdnpu/1502_00/npu.dev.sbin` `1.5.5.391`;
- kernel: `7.1.5-custom+`;
- loaded `amdxdna.ko` SHA-256:
  `9b403eb8d34f0a66f385e6918bba1ebf86da5b527393280047588196b2d16297`;
- XRT: `2.26.0`, commit `e3cb1fa9e1bebc9beedd94c80c950dd31106f0c1`;
- fault XCLBIN SHA-256:
  `d25ab5b8b45a0119c7a62efbe291599020adf86e27609fdc01a6346637ab51b3`;
- qualified comparator template SHA-256:
  `8adb7894816c1671c1a958707672f577707e3520097270ff047716fca1aab675`;
- bridge QoS seam: `0f3d9a5d`.

The complete receipt and raw dispatch outputs are under
`build/experiments/phoenix-pm-clock-characterization/20260804T225006Z-qos-campaign/`.

## Observation

Each QoS session ran one priming dispatch, queried clocks while that context
remained alive, ran the measured dispatch through a fresh context with the same
QoS, and queried again. All ten priming/measured dispatches completed, and all
outputs matched the pinned expected bytes.

| QoS (`gops`, `fps`) | before MP-NPU/H MHz | after MP-NPU/H MHz |
|---|---:|---:|
| `(1, 1000)` | `400/800` | `400/800` |
| `(1, 1800)` | `600/1028` | `600/1028` |
| `(1, 2300)` | `600/1028` | `600/1028` |
| `(1, 3000)` | `600/1028` | `600/1028` |

The pinned XCLBIN declares `operations_per_cycle=2048`. Under the loaded open
driver's solver, `fps=2300` and `fps=3000` exceed the `600/1024` table point and
therefore request higher DPM indices. The SMU-reported clocks nevertheless
remain `600/1028`. Both power-profiles-daemon and the ACPI platform profile
were already `performance`. This establishes the observed collapse; it does
not establish why the SMU returns the cap.

The final no-QoS session and an independent post-run query both reproduced the
original state exactly: power mode `default`, MP-NPU `600 MHz`, H `1028 MHz`.
No threshold boundary search ran after the ratio gate failed.

## Consequence

Two ratios can calibrate two unknowns but cannot independently falsify the
candidate model. Do not fit the two points, treat repeated QoS requests as new
points, use `turbo` with clock gating disabled, partially backport driver power
modes, or change firmware/array scheduling from this result. The next step is a
new proof design that supplies an independent clock-domain constraint or a
third genuinely observable ratio.
