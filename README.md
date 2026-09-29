# Agentic AI Orchestration for Collaborative Split Inference in Wireless Networks

**Sajjad Ahmad**  
Email: sajjadahmad.code@gmail.com

## Overview

This project studies adaptive split inference over a time-varying wireless channel.

A pretrained ResNet-20 model is divided between an edge device and an inference server. Intermediate features are transmitted through a simulated wireless channel, where the channel quality changes over time.

An agent observes the current channel condition and selects:

- the model split point
- the feature quantization level

The system is evaluated using three main metrics:

- Classification accuracy
- Communication cost
- Reconstruction-based privacy leakage

Two contextual-bandit approaches are implemented:

1. **Tabular Contextual Bandit** with discretized SNR states and epsilon-greedy exploration
2. **LinUCB** using continuous SNR as the context

## System Pipeline

```text
CIFAR-10 Image
      |
      v
Device
ResNet-20
      |
      v
Split Point
      |
      v
Feature Compression
      |
      v
Wireless Channel
(SNR-dependent noise)
      |
      v
Server
ResNet-20
      |
      v
Prediction

        ^
        |
      Agent
        |
   SNR -> cut + bits
```

The agent observes the current channel SNR and selects the split point and quantization level for transmission.

## Main Components

### 1. Pretrained Model

A pretrained **ResNet-20** model for CIFAR-10 is used.

The model is partitioned at four candidate points:

```text
c = 0, 1, 2, 3
```

The split points correspond to different depths of the network.

### 2. Wireless Channel

The wireless channel is simulated using a time-varying SNR between:

```text
0–25 dB
```

Intermediate features are quantized before transmission and corrupted with SNR-dependent Gaussian noise.

Available quantization levels are:

```text
2, 4, 8, 32 bits
```

### 3. Privacy Leakage

A lightweight reconstruction decoder is trained for each split point.

The decoder attempts to reconstruct the original CIFAR-10 image from the transmitted intermediate features.

Lower reconstruction error indicates that more input information can be recovered from the transmitted representation.

Measured reconstruction MSE:

| Cut | Reconstruction MSE |
|---:|---:|
| 0 | 0.004469 |
| 1 | 0.075495 |
| 2 | 0.180641 |
| 3 | 0.647271 |

A normalized leakage score is also calculated as a relative reconstruction-based privacy indicator.

### 4. Contextual Bandit

The first agent divides the SNR range into five states and selects one of 16 possible actions.

Each action consists of:

```text
(cut point, quantization bits)
```

The reward combines:

```text
accuracy
communication cost
privacy leakage
```

The agent is trained for 3,000 episodes and evaluated separately for 250 episodes with exploration disabled.

### 5. LinUCB

A second agent uses **LinUCB** instead of the tabular epsilon-greedy approach.

LinUCB uses the continuous SNR value directly as the context and selects actions using an upper-confidence-bound strategy.

The learned policy in this experiment was:

```text
0.0–22.5 dB  -> cut=3, 4 bits
25.0 dB      -> cut=3, 2 bits
```

Thus, the learned policy mainly uses the deepest split and reduces the quantization level at the highest measured SNR.

## Baselines

The agentic approaches are compared with three static strategies:

- **Always Cloud:** cut = 0
- **Always Edge:** cut = 3
- **Fixed Split:** cut = 2

All methods are evaluated under the same time-varying channel conditions.

## Results

### Overall Comparison

| Method | Avg. Accuracy | Avg. Bytes/Step |
|---|---:|---:|
| Always Cloud | 71.8% | 1,045,430.3 |
| Always Edge | 92.0% | 261,357.6 |
| Fixed Split | 91.4% | 522,715.1 |
| Agentic | 88.9% | 424,804.4 |
| **LinUCB** | **92.1%** | **121,634.8** |

The tabular agent reaches 88.9% average accuracy while transmitting fewer bytes than the fixed-split baseline.

The LinUCB agent reaches **92.1% average accuracy** with **121,634.8 bytes per step** in the evaluated experiment.

## Repository Structure

```text
agentic-split-inference-wireless/
│
├── README.md
├── notebook/
│   └── agentic_split_inference.ipynb
│
├── report/
│   └── report.pdf
│
├── results/
│   ├── fig1_accuracy_vs_snr.png
│   ├── fig2_bytes_over_time.png
│   ├── fig3_privacy_leakage.png
│   ├── fig4_reconstructions.png
│   ├── fig5_agent_policy.png
│   ├── fig6_agent_decisions.png
│   └── fig7_comparison_table.png
│
└── figures/
    └── system_architecture.png
```

## Requirements

The prototype was developed and tested using Python and Google Colab.

Main libraries:

```text
PyTorch
Torchvision
NumPy
Matplotlib
```

## Running the Notebook

Open the notebook in Google Colab and run the cells in order.

The notebook covers:

1. Environment setup
2. CIFAR-10 dataset
3. Pretrained ResNet-20
4. Model splitting
5. Wireless channel simulation
6. Reconstruction-based privacy analysis
7. Contextual bandit
8. Baseline evaluation
9. Results and visualization
10. Agent policy analysis
11. Final comparison
12. LinUCB agent
13. LinUCB policy analysis

## Key Findings

- Split inference allows intermediate representations to be transmitted instead of the complete model output.
- Deeper split points produced higher reconstruction error in the privacy probe used in this experiment.
- The original tabular agent improved after increasing training from 250 to 3,000 episodes and separating training from evaluation.
- LinUCB achieved 92.1% average accuracy in the final experiment.
- The LinUCB policy used the deepest split (`cut=3`) across nearly the full measured SNR range and reduced quantization from 4 bits to 2 bits at 25 dB.
- The final results show a trade-off between accuracy, communication cost, and reconstruction-based privacy leakage.

## Limitations and Future Work

The current prototype uses a simulated wireless channel and a small CIFAR-10 benchmark.

Future work includes:

- Systematic reward-weight sweeps
- More realistic wireless channel models
- Additional network context such as latency and queue length
- Stronger reconstruction attacks
- More extensive evaluation across datasets and models
- Further investigation of adaptive split-point selection

## License

This project is intended for research and educational purposes.
