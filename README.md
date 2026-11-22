# Population Genetics Simulator

An interactive simulator of how allele frequencies change over time in two connected populations under the four main forces of evolution: **genetic drift**, **natural selection**, **migration**, and **mutation**.

Adjust the parameters with sliders and watch the frequency of allele **A** evolve, generation by generation, in each population.

## What it models

Each generation, for both populations:

1. **Selection.** The frequency of allele A is reweighted by the fitness of each genotype (AA, Aa, aa), using Hardy–Weinberg genotype proportions:
   p' = (p²·w_AA + pq·w_Aa) / w̄, where w̄ = p²·w_AA + 2pq·w_Aa + q²·w_aa
2. **Genetic drift.** The next generation is sampled at random from a binomial distribution, so smaller populations fluctuate more.
3. **Migration.** A fraction of each population's A alleles moves to the other population.
4. **Mutation.** Each allele can mutate A→a or a→A at the given rate.

Because drift and mutation are random, every run is different. Rerun with the same settings to see the range of outcomes.

## Parameters

| Parameter | Range | Default |
|---|---|---|
| Population 1 / 2 size | 10–500 | 100 / 100 |
| Initial frequency of A (pop 1 / pop 2) | 0–1 | 0.5 / 0.8 |
| Generations | 10–100 | 50 |
| Migration rate | 0–0.5 | 0.05 |
| Mutation rate | 0–0.1 | 0.01 |
| Fitness of AA / Aa / aa | 0–1 | 1.0 / 0.9 / 0.8 |

## Getting started

Requires Python 3.

```bash
git clone https://github.com/Nafay-Dev/PopulationGeneticsSimulator.git
cd PopulationGeneticsSimulator
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Interactive app (recommended):**

```bash
streamlit run streamlit_app.py
```

This opens the simulator in your browser.

**Script version** (uses the fixed defaults above and shows one matplotlib plot):

```bash
python genetic_simulator.py
```

## Things to try

- **Drift in small populations:** set both sizes to 10–20 with no selection (all fitness values 1.0). Allele A often gets lost or fixed by chance alone.
- **Selection wins in large populations:** set the sizes to 500, and the curves follow selection more smoothly.
- **Migration evens things out:** start the populations at very different frequencies and raise the migration rate. They converge.
- **Mutation keeps variation alive:** with a non-zero mutation rate, alleles rarely stay at exactly 0 or 1.

## Files

| File | Description |
|---|---|
| `streamlit_app.py` | Interactive web app with sliders and a live plot |
| `genetic_simulator.py` | Standalone script with fixed parameters |
| `requirements.txt` | Dependencies: `streamlit`, `numpy`, `matplotlib` |

## Built with

Python · NumPy · Matplotlib · Streamlit
