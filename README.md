# Peloton-Aware Breakaway Strategy Model

A physics-based and optimisation-driven model for analysing **cycling breakaway strategies** under realistic race conditions, including **terrain variation, rider physiology, and peloton dynamics**.

This project answers a key question:

> **When should a rider attack, and at what power, to beat the peloton?**

## Overview

This model simulates a cycling race over a **continuous course** and evaluates:

- Peloton dynamics (passive vs strategic)
- Rider endurance constraints
- Terrain-dependent resistance
- Breakaway feasibility and optimal timing

The model integrates a physical description of cycling motion with rider-specific endurance profiles and numerical optimisation techniques. It evaluates breakaway strategies by combining terrain-dependent dynamics, physiological constraints, and brute-force search over possible attack scenarios.

## Project Context

This project builds on earlier group work developed for the Tour de Oxford modelling study:
https://github.com/PabloSavina/MM-Tour-de-Oxford

The current implementation extends that work by moving from MATLAB-based experimentation to a modular Python framework, introducing rider-specific endurance models, flexible terrain profiles, and a systematic optimisation pipeline. In addition, the peloton is no longer treated as purely constant-power: the model now includes a strategically responsive peloton that adjusts effort based on terrain (e.g. increasing power on climbs) and race phase (e.g. late-race surges), allowing direct comparison between passive and adaptive group dynamics.

## Core Idea

At every point along the track, speed is determined by solving:

```math
\frac{P}{v} = \gamma v^2 + \mu_r m g + m g \sin(\theta(x))
```

which balances:

- aerodynamic drag
- rolling resistance
- gravity
- rider power

From this, total race time is computed via:

```math
T = \int \frac{1}{v(x)} \, dx
```

Breakaways are evaluated by splitting the race into:

- **Peloton phase** (before attack)
- **Solo phase** (after attack)

subject to rider endurance limits.

## Rider Models

Three archetypes are implemented:

| Rider Type | Model | Strength |
|-----------|------|--------|
| Sprinter | Exponential decay | High short-term power |
| Climber | Hyperbolic | Sustained climbing ability |
| Time-Trialist | Composite | Balanced endurance |

Each rider has a **power-duration relationship**.

### Sprinter
```math
P(t) = b + (M - b)e^{-kt}
```

### Climber
```math
P(t) = a t^{-d}
```

### Time-trialist
```math
\Delta t(P) = \alpha \cdot (\text{hyperbolic}) + (1-\alpha)\cdot (\text{exponential})
```

## Track Profiles

Supports multiple terrain types:

- Flat
- Hill / Steep hill / Sharp hill
- Double hill
- Mountainous (“hills and valleys”)
- Custom composite profiles

Terrain influences speed via slope:

```math
\theta(x)
```

## Peloton Models

### Passive Peloton
- Constant power output

### Strategic Peloton
- Increases effort on climbs
- Late-race surge
- Terrain-aware pacing

## Breakaway Optimisation

The model evaluates:

- Breakaway location `x_b ∈ [0,1]`
- Breakaway power `P_b`

Using:

- brute-force grid search
- feasibility constraint:

```math
T_{\text{solo}} \leq \text{endurance}(P_b)
```

Outputs:

- Optimal breakaway point
- Required power
- Total finishing time
- Time gained vs peloton

## Visualisations

The notebook produces:

### 1. Finishing Time vs Breakaway Point
- Separate plots for:
  - Passive peloton
  - Strategic peloton
- Shows optimal attack timing

### 2. Optimal Breakaway Points on Track
- Elevation profile
- Markers showing:
  - where each rider should attack
  - required power (W/kg)

### 3. Comparative Strategy Analysis
- Side-by-side rider comparisons
- Time advantage vs peloton

## Project Structure

```text
src/
│
├── backward_method.py      # Core physics + breakaway evaluation
├── grid_search.py          # Strategy optimisation
├── rider_models.py         # Rider endurance models
├── strategic_peloton.py    # Peloton behaviours
├── track_profiles.py       # Terrain definitions
├── plotting.py             # Visualisation utilities
│
notebooks/
└── demo.ipynb          # Main simulation + plots
```

## Installation

```bash
git clone https://github.com/ChamuV/peloton-aware-breakaway-model.git
cd peloton-aware-breakaway-model

pip install -r requirements.txt
```

## Usage

Run the notebook:

```bash
jupyter notebook notebooks/analysis.ipynb
```

Or use the API:

```python
from src.grid_search import run_grid_search_best
from src.rider_models import get_rider

rider = get_rider("climber")

result = run_grid_search_best(
    rider=rider,
    track_name="hill"
)

print(result)
```

## Example Output

```text
[CLIMBER]
Power:     5.6 W/kg
Break @    31 km
Breakaway: 134.96 min
Peloton:   135.72 min
Lead:      0.77 min

[SPRINTER]
Power:     10.3 W/kg
Break @    90 km
Lead:      1.84 min
```

## Key Insights

- Late breakaways dominate for high-power riders (sprinters)
- Climbers benefit from **earlier attacks on terrain**
- Strategic pelotons significantly reduce breakaway success
- Optimal strategy is highly sensitive to:
  - terrain profile
  - endurance curve
  - peloton behaviour

## Possible Extensions

- Drafting effects
- Stochastic race dynamics
- Multi-rider breakaways
- Game-theoretic peloton response
- Real-world race data calibration


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.


