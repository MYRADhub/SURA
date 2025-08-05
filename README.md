# Decentralized Multi-Agent Path Planning and Goal Assignment with Large Language Models

## Overview

This project investigates the challenge of **decentralized goal assignment** and path planning for multiple agents in grid-world environments with obstacles. The focus is on developing, benchmarking, and analyzing strategies that enable a team of agents to coordinate efficiently—assigning unique goals to each agent while minimizing overall completion time.

We compare **classic heuristics**, **optimal centralized solvers**, **human strategies**, and, most notably, **Large Language Model (LLM)-based agents** (e.g., GPT-4, LLaVA) for their ability to solve the multi-agent goal assignment problem under decentralized conditions.

---

## Features

* **Multi-agent grid-world simulation** (configurable grid size, obstacles, agents, goals)
* **Decentralized assignment protocols** with minimal communication (goal ranking exchange only)
* **Automated evaluation** of human, heuristic, LLM, and optimal assignment strategies
* **Rich agent inputs**: grid visualization, distance tables, structured scenario prompts
* **Comprehensive metrics**: makespan (completion time), optimality gap, conflict resolution effectiveness, collisions
* **Modular Python codebase** for reproducible experiments

---

## How It Works

* Each scenario places \$k\$ agents and \$k\$ goals on an \$N \times N\$ grid with randomly placed obstacles.
* **Agents** must assign themselves to goals **without centralized negotiation**, sharing only their ranked goal preferences. Final assignments are determined via a fixed conflict-resolution rule (e.g., agent index priority).
* Once assigned, agents plan shortest collision-free paths (BFS) to their goals.
* Agents receive a structured prompt (for LLMs/humans) or distance data (for baselines) describing the scenario.
* **Performance is measured by the number of steps required until all agents reach their goals (makespan), as well as their gap from the optimal solution.**

---

## Coordination Strategies Implemented

1. **Greedy Assignment:** Each agent chooses its closest available goal; ties resolved by index.
2. **Random Assignment:** Agents assigned to goals randomly.
3. **Optimal (Ground Truth):** Centralized brute-force search provides a lower bound on makespan for comparison.
4. **LLM Agents:** Each agent receives a grid image and detailed prompt (via GPT-4, LLaVA, or similar), generates a ranked goal list, and assignments are finalized based on all agents' submitted rankings.
5. **Human Agents:** Human participants interact with the same visual/structured input as LLMs to assign agents to goals.

---

## Example Research Questions

* Can LLM agents, when given structured input and prompt guidance, match or outperform traditional heuristic methods in decentralized goal assignment?
* How sensitive is LLM performance to the structure and detail of prompt information (e.g., explicit distance tables)?
* How does team size and environment complexity affect the scalability of different coordination strategies?

---

## Technology Stack

* Python (core simulation, agent logic)
* OpenAI GPT-4.1, LLaVA, Ollama (LLM API access)
* Pydantic (for output parsing)
* Matplotlib (visualization)
* YAML (scenario/config management)

---

## File Structure

```
project-root/
├── core/           # Core environment, plotting, pathfinding, LLM interface
├── configs/        # YAML scenario files (easy/medium/hard)
├── agents/         # Heuristic, LLM, human agent implementations
├── results/        # Experiment logs, output CSVs
└── scripts/        # Evaluation and plotting utilities
```

---

## Getting Started

1. **Install requirements** (`pip install -r requirements.txt`)
2. **Configure scenarios** using YAML files in `/configs`
3. **Run experiments** with agents (see scripts/eval\_final.py)
4. **Compare and visualize** results (see scripts/plot\_human\_cases.py, etc.)

---

## Research Outcomes

* **LLM-based agents** (GPT-4.1) using explicit distance tables and structured prompts achieve near-optimal performance and outperform traditional heuristics in decentralized goal assignment.
* LLMs are highly sensitive to the quality and structure of input prompts.
* Provides a new benchmark for evaluating language-based, human, and heuristic strategies in decentralized multi-agent coordination.

---

MIT License.
All code and data are provided for academic and research use.

---

## Project Contributors

* Murad Ismayilov
* Edwin Meriaux
* Shuo Wen
* Gregory Dudek

Center for Intelligent Machines (CIM) Laboratory, McGill University