# Local macOS 13 setup

The team `requirements.txt` pins torch 2.14.1 and torchvision 0.29.1. Their supported Mac packages cannot be installed on this macOS 13 machine. `requirements-macos13.txt` uses torch 2.11.0 and torchvision 0.26.0 instead, keeps the other direct project versions, and explicitly adds Pillow 12.3.0 for image previews and DICOM decoding. Do not treat model results from different environments as exactly interchangeable.

Miniforge is installed locally under `lungstoryshort/`, which is already excluded by `.gitignore`. The project environment is inside `lungstoryshort/envs/lungstoryshort`. The existing copied `.venv` is not used.

## Start working

Open a terminal in the project root, then run:

```sh
source ./lungstoryshort/etc/profile.d/conda.sh
conda activate "$PWD/lungstoryshort/envs/lungstoryshort"
python --version
python -m pip check
```

Expected Python version: 3.12.15. To reinstall project dependencies on this machine:

```sh
python -m pip install -r requirements-macos13.txt
```

To leave the environment:

```sh
conda deactivate
```

In VS Code, select Python: Select Interpreter and choose the interpreter at `lungstoryshort/envs/lungstoryshort/bin/python` inside this project. Select the same environment for notebooks. No shell startup file was changed.

## Manually push code to GitHub

These commands are for a person to run, in accordance with Agents.md. Run one step at a time from the project root. Stop if a command reports an error.

1. Inspect the current changes:

   ```sh
   git status
   ```

2. Create a branch for this work. If this name is already in use, choose a different new name:

   ```sh
   git switch -c preprocessing-setup
   ```

3. Stage only these source/configuration files:

   ```sh
   git add requirements-macos13.txt LOCAL_SETUP.md dataset/script/convert_rsna_annotations.py dataset/script/prepare_dataset.py dataset/script/make_review_examples.py
   git diff --cached --stat
   ```

   Read the file list. It should contain only the files you intend to share. Do not proceed if it includes environment folders, DICOM images, generated PNGs, or other unrelated files. Do not use `git add .` for this task. Existing staged changes, if any, also appear here; review them before committing.

4. Save a local commit, then upload the new branch:

   ```sh
   git commit -m "Add preprocessing scripts and macOS 13 setup"
   git push -u origin preprocessing-setup
   ```

5. Open the GitHub repository, choose Compare & pull request for your branch, and ask your teammate to review it. If GitHub asks you to sign in, complete the authentication yourself. Never paste passwords or access tokens into chat.

`commit` saves a named version locally; `push` uploads those commits to GitHub. Installing an environment does not itself upload anything. No Git commands were executed during this local setup.
