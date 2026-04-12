# Two Cities in One: How Auckland CBD Changes Between Weekdays and Weekends

This project analyses pedestrian activity across Auckland CBD using hourly footfall data from Heart of the City Auckland, combined with weather data from the Open-Meteo Archive API.

## Files included

- `assignment1.qmd` – Quarto document for the assignment
- `assignment1.py` – Python version of the workflow
- `data/` – input CSV data files
- `README.md` – instructions for running the code

## How to run the code

This project uses `uv` for dependency management and Quarto for rendering the report.

### 1. Install dependencies

```bash
uv sync
```

### 2. Render the Quarto document

quarto render assignment1.qmd

To render the PDF version specifically:
```bash
quarto render assignment1.qmd --to pdf
```
### 3. Optional: run the Python script

```bash
uv run python assignment1.py
```

### Outputs

Running the project produces:
- a rendered Quarto report
- saved chart images in the figures/ folder
- interactive HTML maps:
- sensor_map.html
- sensor_cluster_map.html
