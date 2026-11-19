# 04 — Rock physics and fluid substitution

## Goal

Build the physics that relates lithology, porosity, and fluid to Vp, Vs, and density. Then decide whether fluid is detectable in Volve Hugin (**gate G3**).

## Inputs

- Clean, labelled logs from 02.
- Fluid properties from DSTs: Hugin oil density about 0.87–0.90 g/cm³, GOR about 70–100 Sm³/Sm³, reservoir temperature about 110–113 °C. Formation water is similar to Sleipner-area water.
- Mineral moduli from the literature, for example quartz and clay. Record the sources.

## Tasks

1. Compute fluid properties at reservoir pressure and temperature with Batzle–Wang or equivalent: brine, live oil, and optionally gas for comparison.
2. Implement Gassmann fluid substitution, including inverse substitution to the dry frame from in-situ logs.
3. **Fluid sensitivity test (G3):** For the Hugin sands in each training well, substitute to 100% brine and to oil at realistic Sw. Then:
   - Compare the changes in Vp, density, Vp/Vs, acoustic impedance, and shear impedance.
   - Pass both cases through the forward model from 06 using the calibrated wavelet and noise from 03.
   - Compare the brine-versus-oil difference in near, mid, and far amplitudes with the noise level, and with variability from porosity and Vsh in the same well.
4. Optionally fit a simple rock-physics model, such as a stiff-sand or constant-cement trend, for use in task 05.
5. Document the Vs estimation relation and its uncertainty when Vs is missing.

## Outputs

- `src/rock_physics.py`: fluid properties, Gassmann, and optional rock-physics models.
- `configs/rock_physics.yaml`.
- `outputs/fluid_sensitivity/`: brine-versus-oil logs and synthetic panels, a numerical summary table, and a short verdict.
- Record the G3 outcome in `outputs/decision_log.md`.

## Acceptance criteria

- Gassmann round trip works: in-situ → brine → in-situ reproduces the original logs.
- G3 gives a clear verdict, supported by numbers, on whether fluid is a viable target.

## Pitfalls

- Deep, cemented sandstone has a stiff frame, so the fluid effect may be small. That is an important finding, not a failure.
- Gassmann assumes connected porosity and low frequency. Be cautious in shaly or tight intervals.
- Do not tune fluid parameters until the result looks detectable.
