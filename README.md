# F1-Pit-Stop-Strategy
# F1 Pit Stop Strategy Optimizer

Predicts optimal F1 pit stop strategies from real telemetry, and weighs them against the odds of a safety car — not just how they'd play out in a clean race.

Combines tire degradation modeling with a safety car expected-value layer to recommend strategies under uncertainty.

## Why this project

Most strategy models stop at "predict lap time, pick the fastest
strategy." This one goes further by asking: given that safety cars
are unpredictable but statistically common at certain tracks, which
strategy is actually best once you weigh in that uncertainty? It's
also validated against what actually happened in real races, rather
than just trusting the model's own output blindly.

## Built with

- Python 3
- [FastF1](https://github.com/theOehrly/Fast-F1) — real session telemetry
- pandas — data cleaning and pipeline
- scikit-learn (or statsmodels) — per-compound degradation regression
- matplotlib — degradation curves and strategy comparison plots

*(Pin exact versions in `requirements.txt`.)*

## How it works

1. **Data pipeline** (`data_loader.py`) - pulls real session data via
   FastF1, filters out laps that would corrupt a degradation model
   (pit in/out laps, safety car laps).
2. **Degradation modeling** (`degradation_model.py`) - fits a linear
   regression per tire compound: `lap_time ~ tyre_age + lap_number`.
   Lap number acts as a proxy for fuel load, since fuel burn-off and
   tire wear both affect lap time and need to be separated out.
3. **Strategy simulation** (`strategy_simulator.py`) - enumerates
   1-stop and 2-stop pit strategies and ranks them by predicted total
   race time, with a guardrail against recommending stint lengths that
   extrapolate beyond real observed data.
4. **Safety car expected value** (`safety_car_model.py`) - builds a
   per-track safety car probability curve from historical data and
   re-ranks strategies factoring in the value of a cheap pit stop
   under caution.
5. **Validation** (`validate_strategy.py`) - compares the model's top
   recommendation against what the real race winner actually ran.
6. **Visualization** (`visualize.py`) - degradation curve plots and a
   strategy comparison chart.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

FastF1 downloads real session data the first time you run each script,
then caches it locally so re-runs are fast.

## How to run it, in order

```bash
python src/data_loader.py          # confirm data loads and cleans correctly
python src/degradation_model.py    # fit per-compound degradation models
python src/strategy_simulator.py   # rank candidate strategies
python src/safety_car_model.py     # add the safety car EV layer
python src/validate_strategy.py    # compare model vs. real race outcome
python src/visualize.py            # generate the plots below
```

To test a different race, change `YEAR`/`GP`/`TOTAL_LAPS` at the
bottom of `validate_strategy.py` and `visualize.py`.

## Known limitations

- **MEDIUM compound gets skipped at both tracks tested so far** - it
  only had 8 real laps at Bahrain and a similarly small sample at
  Spain, well under the 20-lap minimum needed for a trustworthy fit.
  This means the model can never recommend MEDIUM even when it might
  genuinely be the best choice.
- **SOFT tires showed a near-zero or slightly negative degradation
  coefficient at both tracks**, even after correcting for fuel load.
  This could be a real effect (track surface evolution / rubber-in
  outweighing tire wear on low-degradation tracks like Spain) or an
  artifact of a fairly simple linear model - worth deeper investigation
  with more data.
- **Pit stop time loss is a flat estimate (22 seconds)**, not
  track-specific. Real pit lane length varies significantly by track.
- **The model only optimizes for lap time, not track position.** A
  real team also weighs the risk of losing places in traffic from an
  extra stop - something this model doesn't currently capture, which
  is part of why it sometimes recommends stops a real team wouldn't
  make (e.g. swapping worn Hards for fresh Hards mid-race).
- **Safety car probability is estimated from only 3 historical races**
  per track, which is a small sample for building a reliable
  probability curve.
- **Only tested on 2 tracks (Bahrain, Spain) so far.**

## Planned improvements

- [ ] Pull more seasons of data so MEDIUM (and other under-sampled compounds) can be reliably fit
- [ ] Try a polynomial degradation fit to capture the late-stint "cliff" effect
- [ ] Track-specific pit stop time loss instead of a flat estimate
- [ ] Model track position / traffic risk, not just raw lap time
- [ ] Test on more tracks with different characteristics (street circuits, high-speed circuits, wet races)
- [ ] Simple Streamlit dashboard: pick a race, see the model's recommendation and validation against reality interactively
