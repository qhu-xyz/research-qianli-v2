# Research Template

A quantitative research template repository for xyzpower electricity trading research projects.

## Quick Start: Customizing for Your Research Project

This template uses `research-template` as a placeholder. Follow these steps to rename it for your specific research project (e.g., `research_basis_forecasting`, `research_ancillary_optimization`, etc.):

### 1. Identify Your Research Name

Choose a descriptive name following the pattern: `research_<topic>`

### 2. Rename Core Components

Update these files/directories with your research name:

#### **Directory Structure**
```bash
# Rename the Python package directory
mv research_template research_<your_topic>
```

#### **pyproject.toml** (lines to update)
```toml
[project]
name = "research-<your_topic>"              # Line 2
description = "research repository for xyzpower"  # Line 4 - optionally customize

[tool.ruff.lint.isort]
known-first-party = ["pbase", "pmodel", "psignal"]  # Line 154 - optionally add your package
```

#### **main.py** (lines to update)
```python
def main():
    print("Hello from research-<your_topic>!")  # Line 2
```

#### **Makefile** (lines to update)
```makefile
PROJECT_NAME := research-<your_topic>  # Line 2
```
