from pathlib import Path

import kagglehub

data_dir = Path(__file__).resolve().parent / "data" / "tpu-getting-started"
data_dir.mkdir(parents=True, exist_ok=True)

# Download latest version into ./data (gitignored)
path = kagglehub.competition_download(
    "tpu-getting-started",
    output_dir=str(data_dir),
)

print("Path to competition files:", path)
