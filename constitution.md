# Ensemble / Monte Carlo AVO Uncertainty Workflow (Lithology & Fluid Prediction)

An **Ensemble / Monte Carlo AVO Uncertainty Workflow for Lithology & Fluid Prediction** estimates probabilities over lithology and pore-fluid states—and their joint combinations (such as brine sand, oil/gas/$\text{CO}_2$ sand, shale/caprock, or cemented/tight intervals)—along depth or two-way travel time, together with uncertainty and ambiguity metrics.

Rather than stopping at raw intercept ($A$) and gradient ($B$) values or AVO classes (I–IV) alone, this workflow propagates geological, petrophysical, rock-physics, and seismic uncertainties across $N$ realizations (e.g., 100–500 runs) to produce calibrated lithology and fluid posteriors, highlight where lithology and fluid signatures overlap, and—when paired with a short local seismic window—check consistency against observed seismic response.

## Scientific Motivation

In AVO analysis, predicting lithology and fluids directly from elastic responses is inherently non-unique:

* **Lithology vs. Fluid Trade-Offs:** A change from brine to gas/oil or $\text{CO}_2$ can produce amplitude and gradient shifts similar to changes in clay/shale content ($V_{\text{sh}}$), porosity, or cementation—and thin-bed tuning can further distort the apparent response.
* **Shear & Frame Uncertainty:** Missing or estimated shear-wave velocity ($V_s$), uncertain mineral moduli, or washout-affected density/sonic logs can shift predicted fluid/lithology probabilities even when apparent AVO trends look strong.
* **Acquisition & Calibration Limits:** Wavelet frequency, phase, noise floor, angle range, and local amplitude scaling affect how cleanly lithology and fluid states separate in $(A, B)$ or elastic attribute space.

By running Monte Carlo realizations over rock-physics, fluid-substitution, and seismic parameters—and optionally calibrating wavelet/noise/scale on a short seismic window near the well—you obtain posterior probabilities for lithology and fluid, posterior entropy / confidence, and value-of-information diagnostics showing what reduces classification ambiguity most.

## Data Needed

### 1. Well Log Data (`LAS` / `CSV` / `Parquet`)

* **Core Elastic Logs (Required):**
  * Compressional slowness (`DTC` / `DT` / `DTCO`) $\rightarrow$ P-wave velocity ($V_p$)
  * Shear slowness (`DTS` / `DTSM`) $\rightarrow$ S-wave velocity ($V_s$), or rock-physics estimation + uncertainty bounds when missing
  * Bulk density (`RHOB` / `RHOZ`) $\rightarrow$ rock bulk density ($\rho$)
* **Petrophysical & Lithology Logs (Required for Lithology & Gassmann Fluid Modeling):**
  * Gamma ray (`GR`) $\rightarrow$ shale volume ($V_{\text{sh}}$) and baseline lithology indicators
  * Porosity (`PHIT` / `PHIE`) or neutron/density pair (`NPHI` + `RHOB`)
  * Deep resistivity (`RT` / `ILD` / `LLD` / `AT90`) $\rightarrow$ baseline water saturation ($S_w$)
  * Optional core/cuttings/lithofacies labels (when available for benchmarking or defining lithology classes)
* **QC & Time-Depth Logs (Recommended):**
  * Caliper (`CALI`), bit size (`BS`), density correction (`DRHO`), and photoelectric factor (`PEF`) to flag bad-hole intervals and mineralogies
  * Checkshot / VSP / time-depth curve (`MD` / `TVD` to two-way travel time `TWT`)

### 2. Optional Calibration Data (Short Seismic Slice)

A narrow time window around the target interval and small spatial footprint (e.g., a single trace, super-gather, or local angle gather near the well) in `SEG-Y` or array format, used to estimate local wavelet/spectrum, noise/SNR, incident-angle range, and amplitude scaling.

### 3. Lithology, Fluid & Perturbation Config (`YAML` / `JSON`)

* **Target Lithology & Fluid Classes:** e.g., $\text{Lithology} \in \{\text{Sand}, \text{Shale}, \text{Tight/Carbonate}\}$ and $\text{Fluid} \in \{\text{Brine}, \text{Oil}, \text{Gas}/\text{CO}_2\}$ (or joint classes like Brine Sand, Hydrocarbon/$\text{CO}_2$ Sand, Shale).
* **Rock-Physics & Fluid Parameters:** Mineral moduli/densities, fluid bulk modulus/density and saturation ranges, dry-frame/Gassmann assumptions, and $V_s$ estimation uncertainty.
* **Seismic Perturbation Ranges:** Wavelet center frequency/phase/noise, angle window ($\theta_{\min}\text{–}\theta_{\max}$), and local calibration settings.

## End-to-End Workflow

1. **Ingest, QC & Label Baseline:** Load well logs, map depth (`MD`/`TVD`) to `TWT`, flag washouts/bad-hole intervals (`CALI`/`DRHO`), and assign baseline lithology/fluid states (from petrophysics, core/facies logs, or defined scenarios).
2. **Calibrate on Short Seismic Window (Optional):** Estimate local wavelet bandwidth/phase, noise floor, and amplitude scaling from a narrow seismic window around the well tie without overfitting to a single trace.
3. **Sample Monte Carlo Ensemble:** Generate $N$ realizations across lithology/rock-physics parameters (e.g., $V_{\text{sh}}$, porosity, $V_s$ relation/error), Gassmann fluid/saturation states, and seismic wavelet/noise/angle perturbations.
4. **Forward-Model Synthetic Angle Gathers:** Compute angle-dependent reflectivity (Zoeppritz or Aki-Richards) and convolve with wavelets to build synthetic angle gathers per realization.
5. **Invert / Classify Lithology & Fluids Across Realizations:** Extract AVO/elastic attributes ($A$, $B$, derived fluid/lithology indicators) and map each realization to predicted lithology and fluid states (via rock-physics template likelihoods, distance/boundary rules, or a trained classifier).
6. **Aggregate Posteriors & Diagnose Ambiguity (Deterministic + Agent):** Compute lithology probabilities, fluid probabilities, joint lithology-fluid probabilities, and uncertainty metrics (entropy, margin between top two classes, and lithology-vs-fluid confusion rates); use deterministic rules + an agent layer to flag ambiguous zones and recommend high-value data/constraints.

## Suggested Repository Structure

```text
monte-carlo-avo-litho-fluid/
├── README.md
├── pyproject.toml                       # or requirements.txt (numpy, scipy, pandas, lasio, segyio, matplotlib)
├── configs/
│   └── litho_fluid_config.yaml          # lithology/fluid classes, Gassmann & perturbation ranges, calibration
├── data/
│   ├── sample_well.las                  # or synthetic/public test well log
│   └── local_seismic_window/            # optional short trace/gather snippet around the well
├── src/
│   ├── io_qc.py                         # log loading, time-depth mapping, bad-hole flagging, baseline states
│   ├── rock_physics.py                  # Vp/Vs estimation, Gassmann fluid substitution, lithology/fluid sampling
│   ├── avo_forward.py                   # Zoeppritz / Aki-Richards reflectivity, wavelet convolution, A/B extraction
│   ├── calibration.py                   # optional wavelet, noise, and scale estimation from short seismic slice
│   ├── litho_fluid_inference.py         # maps synthetic AVO responses to lithology & fluid posteriors + uncertainty
│   └── agent_diagnostics.py             # deterministic ambiguity checks + agent summary of lithology/fluid risk
├── notebooks_or_scripts/
│   └── run_pipeline.py                  # CLI entry point generating lithology/fluid tracks, plots, and report
└── outputs/
    ├── litho_fluid_probs.png
    ├── ab_litho_fluid_cloud.png
    └── litho_fluid_report.md
```

## Key Outputs & Evaluation Metrics

### Primary Outputs (Per Depth/Time Sample or Interval)

* **Lithology Probabilities & Most-Likely Lithology:** e.g., $P(\text{Sand})$, $P(\text{Shale})$, $P(\text{Tight})$.
* **Fluid Probabilities & Most-Likely Fluid:** e.g., $P(\text{Brine})$, $P(\text{Oil})$, $P(\text{Gas}/\text{CO}_2)$.
* **Joint Lithology-Fluid Distribution:** e.g., distinguishing Brine Sand vs. Gas/$\text{CO}_2$ Sand vs. Shale.

### Uncertainty & Ambiguity Metrics

* **Prediction Entropy & Top-2 Margin:** Measures how split the ensemble is across candidate lithologies or fluids.
* **Lithology vs. Fluid Ambiguity Rate:** Tracks how often uncertainty stems from fluid/saturation changes vs. lithology/rock-physics/tuning shifts.
* **Calibration Agreement (When Using Short Seismic Slice):** Compares the observed seismic AVO response against the ensemble's predicted lithology/fluid region.