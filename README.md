# Cutting Stock Problem Multi-Agent Solution

A multi-agent system implementation for solving the Cutting Stock Problem (CSP) using various heuristic algorithms and optimization strategies.

## Overview

The Cutting Stock Problem involves cutting stock materials of fixed length into smaller pieces to meet specific demands while minimizing waste. This project implements a multi-agent approach using different heuristic methods to find optimal solutions.

## Features

- **Multiple Heuristic Algorithms:**
  - Simulated Annealing (SA)
  - Hill Climbing (HC)
  - Genetic Algorithm (GA)

- **Two Multi-Agent Strategies:**
  - **Collaborative:** Agents use the same algorithm with different parameters
  - **Hyper Meta-Heuristic:** Each agent uses a different algorithm

- **Interactive Web Interface:** Built with Gradio for easy testing and visualization

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cutting-stock-problem
```

2. Install required dependencies:
```bash
pip install numpy matplotlib gradio
```

## Quick Start

Run the application:
```bash
python app.py
```

The web interface will be available at `http://localhost:7860`

![image](https://github.com/user-attachments/assets/6b62ff5a-d3ca-4b50-bf1f-9e1df97940f0)

## Usage

### Input Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| **Stock Length** | Length of raw material | 100 |
| **Piece Lengths** | Required piece lengths (comma-separated) | 10,15,20 |
| **Piece Demands** | Quantity needed for each piece | 5,3,2 |
| **Number of Agents** | Number of agents (1-5) | 3 |
| **Strategy** | Agent interaction type | Collaborative/Hyper |
| **Algorithms** | Optimization algorithms | sa, hc, ga |

### Example Usage

```
Stock Length: 100
Piece Lengths: 10,15,20,25
Piece Demands: 5,3,2,4
Number of Agents: 3
Strategy: Hyper
Algorithms: hc,sa,ga
```

## Algorithms

### Simulated Annealing (SA)
- **Initial Temperature:** 1000
- **Cooling Rate:** 0.95
- **Iterations:** 1000

### Hill Climbing (HC)
- **Iterations:** 1000
- **Neighbors per step:** 5

### Genetic Algorithm (GA)
- **Population Size:** 30
- **Generations:** 50
- **Mutation Rate:** 0.2

## Performance Results

Based on experimental testing:

| Scenario | Strategy | Algorithms | Agents | Best Waste | Best Algorithm | Time (s) |
|----------|----------|------------|--------|------------|----------------|----------|
| 4 | Collaborative | GA | 1 | 10 | Genetic Algorithm | 0.02 |
| 7 | Hyper | GA | 3 | 5 | Genetic Algorithm | 0.02 |
| 8 | Hyper | GA, HC, SA | 5 | 5 | Genetic Algorithm | 0.10 |

#### Scenario 1
![image](https://github.com/user-attachments/assets/19093c8b-78d2-40ea-a894-f747a25b63a2)

#### Scenario 2
![image](https://github.com/user-attachments/assets/49ae897a-503b-424b-9f2b-35c0863574c9)
...
### Key Findings

- **Best Overall Performance:** Hyper Strategy with Genetic Algorithm
- **Lowest Waste:** 5 units (Scenarios 7 & 8)
- **Fastest Execution:** Genetic Algorithm consistently shows low processing times
- **Strategy Comparison:** Hyper strategy generally outperforms Collaborative strategy

## Architecture

```
├── CSPAgent (Base Class)
├── Algorithm Agents
│   ├── SimulatedAnnealingAgent
│   ├── HillClimbingAgent
│   └── GeneticAlgorithmAgent
├── MultiAgentSystem (Coordinator)
└── Gradio Interface (UI)
```

## Technical Details

### Data Structures
- **Stock Length:** Single integer value
- **Piece Lengths:** List of required piece lengths
- **Piece Demands:** List of demand quantities
- **Solution:** List of cutting patterns per stock roll

### Collaborative Strategy Parameters
- Different agents use varied parameter settings
- Agents periodically share best solutions
- Promotes exploration diversity

### Hyper Strategy Parameters
- Each agent uses different algorithms
- Combines strengths of multiple approaches
- Balances solution quality and computation time

## Visualization

The application provides visual representations of:
- Cutting patterns for each stock roll
- Waste optimization across different scenarios
- Algorithm performance comparisons


## Acknowledgments

- Inspired by classical optimization problems in operations research
- Multi-agent system concepts from distributed AI
- Heuristic algorithms from computational intelligence literature
