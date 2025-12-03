# Health_care_ABC_Algorithm

**Artificial Bee Colony–based optimization for Multi-Depot Multi-Satellite Multi-Compartment Vehicle Routing in healthcare logistics**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This repository provides a reproducible Python implementation of an enhanced Artificial Bee Colony (ABC) algorithm for the Multi-Depot Multi-Satellite Multi-Compartment Vehicle Routing Problem (MDMS-MCVRP) in healthcare logistics. The framework supports two-echelon routing (depots → satellites → health facilities), multiple commodities, realistic capacity constraints, and detailed sensitivity analysis over both fleet size and capacity settings.

The code is structured for research use, with a clear separation between problem definition, algorithm core, solution representation, and experimental scripts.

---

## Problem Description

The MDMS-MCVRP considered here includes:

- Multiple primary depots supplying vaccines and medicines.
- Multiple satellites acting as intermediate consolidation points.
- Multi-compartment secondary vehicles with compartment-wise capacity limits.
- A two-echelon routing structure:
  - Primary routes: depot → satellites → depot  
  - Secondary routes: satellite → customers → satellite
- Operational constraints such as:
  - Mandatory visits of all satellites by primary vehicles.
  - Capacity limits at depots, satellites, and vehicle compartments.
  - Complete service of all healthcare facilities.

---

## Key Features

- ✅ Custom ABC algorithm tailored for multi-echelon, multi-commodity VRP.
- ✅ **ML-Enhanced Optimization** - Automated hyperparameter tuning using Grid Search and Bayesian methods
- ✅ Explicit modeling of primary and secondary (two-echelon) routes.
- ✅ Penalty-based constraint handling with repair operators.
- ✅ Sensitivity analysis over:
  - Number of secondary vehicles (per satellite and total fleet size).
  - Satellite capacities (W) and secondary compartment capacities (Q).
- ✅ Multiple-run statistics: best, mean, standard deviation, success rate.
- ✅ Visualization utilities for convergence, sensitivity heatmaps, fitness distributions, and route exports (e.g. to QGIS).

---




- `main.py` runs the core ABC on the full healthcare instance.
- `sensitivity_analysis_number_of_vehicles.py` studies how performance changes with different numbers of secondary vehicles (e.g. 4, 6, 8, 10, 12).
- `sensitivity_analysis_with_capacities.py` varies satellite capacity (W) and secondary vehicle capacity (Q) for a fixed “optimal” vehicle configuration (e.g. 2 vehicles per satellite).

---

## Algorithm Details

### Artificial Bee Colony (ABC)

The implemented ABC follows the standard three-phase structure with domain-specific modifications:

1. **Employed Bee Phase**  
   Local exploration around current solutions using custom neighborhood operators:
   - Satellite sequence perturbations.
   - Customer swap and relocate moves within and between routes.
   - Route-level refinements for both echelons.

2. **Onlooker Bee Phase**  
   Probabilistic selection of promising solutions based on fitness and focused intensification around them.

3. **Scout Bee Phase**  
   Diversification by replacing stagnated solutions with new randomized candidates.

### Core Design Choices

- Route-based encoding with explicit primary and secondary segments.
- Penalty-augmented fitness:
  - Travel distance plus weighted penalties for infeasibilities (capacity, satellite visits, assignment completeness).
- Repair mechanisms to:
  - Enforce satellite-visit constraints for primary vehicles.
  - Reduce or eliminate capacity violations where possible.
- Hyperparameters tuned via systematic experimentation (e.g. swarm size, iteration limit, employed/onlooker ratios, abandonment limit).

---

## Experimental Components

### Hyperparameter Tuning

 The **Bayesian Optimization** exhibited a **smoother and more consistent convergence curve** indicating focused exploration in promising hyperparameter regions.
- Despite this, its **higher standard deviation** is due to Bayesian sampling a **wider parameter space including some less optimal points** during exploration, resulting in greater variability across trials.
- The **Grid Search**, while less smooth, has a **lower standard deviation** because of exhaustive but fixed sampling over a uniform grid, which tends to produce steadier but less finely tuned results.
- The **noise in Grid Search’s convergence** reflects its uniform sampling and the presence of suboptimal parameter combinations.
- Ultimately, **Bayesian Optimization is more adaptive and efficient**, allowing faster convergence on competitive solutions despite variability, while **Grid Search trades consistency for exhaustive coverage**.
### Sensitivity Analysis – Number of Vehicles

Using `sensitivity_analysis_number_of_vehicles.py`, the code:

- Selects subsets of customers (e.g. 10, 20, 30, full set) to study scalability.
- Evaluates multiple vehicle configurations, such as:
  - Symmetric fleet sizes per satellite (e.g. {S1:2, S2:2}, {S1:3, S2:3}, …).
- Aggregates results to identify:
  - Mean and best fitness per configuration.
  - Success rates and variability.
  - An empirically “optimal” fleet size for the given healthcare instance.

### Sensitivity Analysis – Capacities

Using `sensitivity_analysis_with_capacities.py`, the code:

- Fixes the secondary fleet to an empirically optimal configuration (e.g. 4 vehicles: 2 per satellite).
- Varies:
  - Satellite capacities \(W\) across several levels.
  - Secondary vehicle compartment capacities \(Q\) across several levels.
- Produces:
  - Heatmaps of mean fitness over the \((W, Q)\) grid.
  - Capacity–performance curves and distributions.
  - Recommended capacity settings balancing feasibility and distance.

---

## Visualizations

Typical plots generated by the experiments include:

- **Convergence curves** of the ABC algorithm over iterations.
- **Heatmaps** for capacity and vehicle-count sensitivity.
- **Box plots and histograms** for fitness distributions across scenarios.
- **Route exports** (e.g. GeoJSON/CSV) suitable for visualization in QGIS or other GIS tools.

---

## Technical Specifications

- **Language**: Python 3.8+
- **Core dependencies** (see `requirements.txt` for full list):
  - NumPy, SciPy, Pandas
  - Matplotlib
  - (Optional) Optuna / other tools if you add advanced tuning
- **Problem type**: NP-hard combinatorial optimization
- **Algorithm class**: Swarm intelligence / bio-inspired metaheuristic
- **Use cases**:
  - Vaccine and medicine distribution
  - Multi-echelon supply chains
  - Academic benchmarking of metaheuristics

---

## Getting Started

### Installation

git clone https://github.com/khatrigoldy/Health_care_ABC_Algorithm.git cd Health_care_ABC_Algorithm
pip install -r requirements.txt


Place your instance data (distance matrices, demands, etc.) in the `data/` directory as expected by `problem.py`.

### Running the main experiment

python main.py

### Running the sensitivity analyses

Number of vehicles:

python sensitivity_analysis_number_of_vehicles.py


Capacities (satellite and secondary vehicles):

python sensitivity_analysis_with_capacities.py


---

## Research Contributions

This repository supports the following contributions:

1. A customized ABC variant for a realistic multi-depot, multi-satellite, multi-compartment healthcare VRP.
2. A reproducible experimental framework for:
   - Hyperparameter tuning,
   - Fleet-size sensitivity,
   - Capacity sensitivity.
3. Practical guidelines on how vehicle fleet size and capacity settings impact routing performance in a two-echelon healthcare network.

If you use this work in your research, please include an appropriate citation to the associated report or paper (add your BibTeX/APA entry here once finalized).

---

## Contact

**Author**: Gurnam Singh  
**Institution**: Kirori Mal College, University of Delhi  
**Department**: Department of Operational Research  
**GitHub**: [@khatrigoldy](https://github.com/khatrigoldy)  
**E‑mail**: khatrigoldy10@gmail.com  

For questions, suggestions, or collaboration, please open an issue or contact directly.

---

## License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.

---

*Developed as part of advanced research in Operations Research and swarm intelligence for healthcare logistics.*  
*If you find this repository useful, please consider starring it on GitHub.*  

