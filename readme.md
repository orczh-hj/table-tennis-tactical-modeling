# AI-Driven Tactical Recommendations for Table Tennis: Decision Optimization with Probabilistic Interaction Model and Technical Quantification System

## Abstract

Current tactics in table tennis predominantly rely on empirical knowledge, lacking systematic and adaptive decision-making frameworks. To address this gap, this study proposes an intelligent tactical decision-making system integrating probabilistic modeling, quantified technical proficiency, and deep reinforcement learning. A probabilistic interaction model is established to formalize tactical dynamics, explicitly defining decision variables—spin, drop point, and quality—while accounting for technical constraints between players. Central to the framework is the Technical Capability Parameter Table (TCPT), a novel quantification system that evaluates athletes' adaptability and stability across diverse ball conditions. Leveraging these components, the Multi-Head Hybrid-Decision Proximal Policy Optimization (MHHD-PPO) algorithm is developed to optimize hybrid action spaces (discrete tactical choices and continuous quality control) and exploit temporal dependencies in gameplay. Experiments demonstrate that agents trained with MHHD-PPO achieve a 63.5% win rate against baseline strategies, with real-world validation involving university athletes revealing a significant win rate improvement (e.g., 48% to 59% in Player B vs. Player C matchups). The system provides actionable tactical recommendations through three operational modes: (1) adaptive serve/return strategies tailored to opponent weaknesses, (2) dilemma-specific solution generation, and (3) self-play optimization. By bridging theoretical models with practical training paradigms, this work advances the intelligent development of table tennis tactics, offering coaches and athletes a data-driven tool for strategic refinement. The integration of probabilistic interaction modeling, technical proficiency quantification, and hybrid reinforcement learning establishes a replicable framework for tactical intelligence in dynamic sports.

## Files

**tactics_of_player_ABC.py**

The original decision-making logic of athletes A, B, and C

**Athlete TPCTs.xlsx**

TCPT of Player A, B, and C

**History log Player X agent-Player X.json**

Logs of athletes A, B, and C playing using agent-based strategies and raw decision-making logic

**History log X-X.json**

Match logs between three athletes, A, B, and C

**three_common_tactics.py**

The decision-making logic of `Backspin attack`, `Raly`, and `Defense`