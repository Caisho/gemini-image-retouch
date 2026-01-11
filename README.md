# SmartThings API Extractor

Add your description here

## Prerequisites

- Python 3.13 or higher
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Caisho/smartthings-api-extractor.git
   cd smartthings-api-extractor
   ```

2. **Create a virtual environment with the required Python version:**
   ```bash
   uv venv --python 3.13
   ```

3. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

4. **Install the project and development dependencies:**
   ```bash
   uv sync --extra dev
   ```

## Development

After setup, you can:

- Run tests: `pytest`
- Format code: `ruff format`
- Lint code: `ruff check`
- Check for dead code: `vulture src/`

## Usage

[Add usage instructions here]