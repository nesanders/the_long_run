# Lessons Learned: The Challenge of Synchronization

This project served as a case study in **Agentic Coding** and **Full-Stack Data Science**. While the generative capabilities of the system are powerful, the primary friction point was **synchronization** across the three pillars of the project:

1.  **The Engine** (`lifetime_model.py`): The source of truth for data and logic.
2.  **The Frontend** (`index.html`): The presentation layer and interactive calculators.
3.  **The Documentation** (`README.md`): The user instruction manual.

## Key Takeaways

### 1. The "Drift" of Hardcoded Values
Early in the project, specific numbers (e.g., "Peak at 70k miles") were hardcoded into the HTML captions. As the model was refined (e.g., adding the Retention/Duration parameter), the simulation results changed, but the static text did not.
*   **Solution:** We moved to an **Automated Injection** pattern. The Python script now prints a valid JSON object (`FIG_STATS`), which is copy-pasted (or piped) into the JavaScript. The HTML captions now use `span` tags with IDs populated dynamically on load. This ensures the text *cannot* lie about the chart.

### 2. Terminology Consistency
We shifted terminology mid-stream from "Core" runners to "Dedicated" runners to better distinguish them from the "Recreational" tier.
*   **The Trap:** While the Python variables were updated, the HTML narrative retained the old "Core" language in several places.
*   **Lesson:** Semantic search (grep) is essential when renaming concepts. A "Find and Replace" must be rigorous across all file types (code, comments, UI text).

### 3. "Model-Doc" Parity
The blog post described "Genesis" and "Retention" patterns as narrative concepts. However, the user correctly challenged whether these were *actually* in the mathematical model.
*   **Reality Check:** Upon inspection, the "Retention" parameter was initially missing from the PyMC inference step.
*   **Alignment:** We had to explicitly add `mu_duration` to the Bayesian model and constrain it via the "Herron Limit" Potential. This aligned the *code* with the *story*. Documentation should not just describe what *should* be happening; it must reflect what *is* computed.

## Best Practices for Future Projects

*   **Single Source of Truth:** Where possible, generate documentation variables from the code itself.
*   **Explicit Observation Maps:** Clearly listing how latent variables map to real-world anchors (as we did in the "Likelihood Function" section) exposes gaps in logic immediately.
*   **Agentic Verification:** The "Fact Check" step—explicitly comparing anchors across files—was the most high-value action taken. It revealed discrepancies (27 vs 37 mpw) that would have otherwise gone unnoticed.
