# ABZ 2025 Case Study - RSS safety shield for single-lane scenarios

## Training and testing of agents

### Setup

Ensure you are using Python **3.10.12**,
as the library versions specified in `requirements.txt`
are incompatible with newer Python releases.

We recommend using **pyenv**
([GitHub repo](https://github.com/pyenv/pyenv))
to manage your Python version.

Once the correct Python version is set up,
install the required version of `highway-env` by running:

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Run

To facilitate the reproduction of our case study results,
agents for the Highway Environment can be trained or tested via the command line.

Since the training phase is time-consuming, you can skip it,
as the pre-trained models are already included in this repository.

```bash
# Train the base and adversarial single-lane agent models:
python SafetyControllerCaseStudy.py train

# Load the trained models, run simulations, and render their behavior
# under different conditions (with or without shielding,
# and against IDM or Aggressive vehicles).
python SafetyControllerCaseStudy.py test
```

## Implementation

To enhance reliability, we refactored the original code into a single Python module
with a unified fundamental class, reducing potential issues caused by code duplication.

Additionally, we fine-tuned the setup for greater accuracy:

- The single-lane configuration allows only the `SLOWER`, `IDLE`, and `FASTER` actions.
  We reported this to the ABZ 2025 case study organizers in this issue:
  [hhu-stups/abz2025_casestudy_autonomous_driving#15](https://github.com/hhu-stups/abz2025_casestudy_autonomous_driving/issues/15)
- Test runs use a fixed randomization seed, ensuring better reproducibility
  of our case study results.

### Agents

Our experiments involve the following agents:

- Single-lane Base agent: Implemented as the `HighwayAgent1laneBase` class.
- Single-lane Adversarial agent: Implemented as the `HighwayAgent1laneAdversarial` class.

All agents are derived from the `HighwayAgent` class.

### Safety shield

The safety shield is implemented in method `HighwayAgent.compute_rss`.

## Agent Models

The agent models are trained using machine learning,
retaining the model and training parameters provided in the original
case study repository:
[hhu-stups/abz2025_casestudy_autonomous_driving](https://github.com/hhu-stups/abz2025_casestudy_autonomous_driving).

In summary, we train a Deep Q-Network (DQN) reinforcement learning model
using the Stable-Baselines3 library in Python.
The Q-network consists of two hidden layers, each with 256 neurons.
A discount factor of 0.9 is applied, prioritizing short-term rewards.

All models are trained through interactions with IDM (Intelligent Driver Model) vehicles.
During test runs, the ego vehicle may encounter either IDM or aggressive vehicles
(as defined by the `AggressiveVehicle` class in `highway-env`).
This setup allows us to evaluate the agent's behavior in unfamiliar situations.


## Training and testing the original ABZ agent models

The original models for the ABZ case study have been retained in this repository:
see files `HighwayEnvironment_Adversarial.py`, `HighwayEnvironment_Base.py`,
`HighwayEnvironment_Single_Adversarial.py`, `HighwayEnvironment_Single_Base.py`.

The documentation to train and test them is available here:
[hhu-stups/abz2025_casestudy_autonomous_driving](https://github.com/hhu-stups/abz2025_casestudy_autonomous_driving)

