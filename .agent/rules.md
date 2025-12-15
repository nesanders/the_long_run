# Project Rules

## 1. Environment Policy
**CRITICAL:** You must ALWAYS use the `the_long_run` conda environment when running Python scripts or installing packages.
- **Do not** use the base environment.
- **Do not** use `pip` without ensuring the conda environment is active.
- **Command:** `conda run -n the_long_run python <script>` or `conda activate the_long_run` before running commands.

## 2. Synchronization Policy
The project consists of three tightly coupled components. You must ensure they remain synchronized:
1.  **Python Model (`lifetime_model.py`)**: The Source of Truth for all statistical parameters, priors, and simulation logic.
2.  **Frontend (`index.html`)**: Contains hardcoded JS versions of the model parameters for the calculators. **If you change the model parameters in Python, you MUST update the constants in `index.html` (e.g., `WEIGHTS`, `BETA_GENDER`, Means/Stds).**
3.  **Documentation (`README.md` / `index.html` text)**: The explanatory text must accurately reflect the mathematical model implemented in the code.

**Rule:** When modifying the model, always perform a sweep to update the HTML calculators and text descriptions to match.
