# LungStoryShort

A chest X-ray research project that will use ResNet-18 for image classification
and Faster R-CNN for lung opacity localization. The project will compare image
rankings from the classifier, the detector, and a combination of their scores.

The project is in its setup phase. Training and evaluation code has not yet been implemented.

## Environment

- Environment manager: Conda, with an environment named `lungstoryshort`.
- Python: `3.12.15`.
- Dependency versions were recorded from a local Conda environment on macOS ARM64.
- `requirements.txt` pins direct dependencies. Transitive dependencies are resolved by pip.

## Initial Setup

Install Conda, then open a terminal in the cloned repository's root directory,
which contains `requirements.txt`. Each team member should create a local environment:

```bash
conda create -n lungstoryshort python=3.12.15 pip
conda activate lungstoryshort
python -m pip install -r requirements.txt
```

Skip the first command if the `lungstoryshort` environment already exists.

Check the dependencies and Python interpreter:

```bash
python --version
python -m pip check
python -c "import sys, torch, torchvision, pydicom; print(sys.executable); print('torch:', torch.__version__); print('torchvision:', torchvision.__version__)"
```

`pip check` should report `No broken requirements found.` The interpreter path
should point to the `lungstoryshort` Conda environment.

## Daily Use

Activate the environment in each new terminal:

```bash
conda activate lungstoryshort
```

Deactivate the environment:

```bash
conda deactivate
```

In VS Code, select the `lungstoryshort` Conda environment as both the Python
interpreter and the notebook kernel.

## Team Workflow

Share `requirements.txt` through the repository. Each team member maintains a
separate local environment. After the dependency file changes, update the active environment:

```bash
python -m pip install -r requirements.txt
python -m pip check
```

When adding or upgrading a direct dependency, record its installed version in
`requirements.txt` and verify the installation. This file does not lock all
transitive dependencies or guarantee identical results across platforms.
For Windows, Linux, or NVIDIA GPU environments, follow the
[official PyTorch installation instructions](https://pytorch.org/get-started/locally/)
to select the appropriate build and check compatibility with the pinned versions.

## Current Structure

```text
RBC_Letssolveit_LungStoryShort/
├── dataset/
│   └── scripts/       # Data preparation code
├── models/            # Model code
├── .gitignore
├── Agents.md
├── readme.md
└── requirements.txt
```

`dataset/scripts/` and `models/` are currently empty. Add code as development proceeds.
Local virtual environments, Python caches, and notebook checkpoints are excluded
by `.gitignore`.
