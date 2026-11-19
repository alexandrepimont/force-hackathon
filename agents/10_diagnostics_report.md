# 10 — Diagnostics, what-if analysis, and report

## Goal

Make the results explorable and explain them honestly. Provide an agent layer that runs real simulations on request and compiles the final report and demo.

## Inputs

- Outputs from all tasks, `outputs/decision_log.md`, and configs.

## Tasks

1. **What-if tool:** Let a user ask questions such as:
   - "What if Hugin porosity is 5 pp lower at well X?"
   - "What if Hugin is half as thick?"
   - "What if the oil is replaced by brine?"

   The agent edits a copy of the relevant config, runs tasks 04 → 06 → 07 for that well or location, and shows before-and-after synthetics and predictions. **No answer may be given without running the simulation.**
2. **Ambiguity diagnostics:**
   - Identify which scenarios produce near-identical near/mid/far responses but different labels.
   - Explain which assumptions dominate each ambiguity, such as wavelet, Vs, porosity, or thickness.
   - Use the deterministic checks from the constitution, such as entropy and top-2 margin, as inputs. The agent explains results; it does not invent them.
3. **Value of information, optional:** Rank which extra observation would most reduce ambiguity, for example measured Vs, better wavelet control, or one more well. State that the ranking depends on the model and the chosen scenarios.
4. **Report:** Write `outputs/report.md` with:
   - the question;
   - data used and its attribution;
   - the decision log;
   - blind-well A/B/C results;
   - the fluid-sensitivity verdict;
   - maps;
   - limitations;
   - next steps.
5. **Demo:** Prepare a 5-minute flow:
   1. Show the Gassmann panel.
   2. Show real-versus-synthetic data in a crossplot.
   3. Show the blind-well score table.
   4. Show the map.
   5. Run one live what-if example.

## Outputs

- `src/agent_diagnostics.py`, including the what-if runner.
- `outputs/report.md`.
- `outputs/demo/`.

## Acceptance criteria

- Every number in the report comes from a file in `outputs/`.
- Limitations are explicit, including:
  - the number of blind wells;
  - whether Vs was estimated or measured;
  - the timing of seismic and logs relative to production;
  - the assumptions behind the priors.

## Pitfalls

- Do not let the language model "interpret" beyond the simulations and data.
- Do not call model-conditional probabilities "true probabilities" unless the blind-well reliability results support it.
