# Git Message Tags

This document defines the tags to be used in git commit messages for the SCARF project.

## Tags

- `feat`: A new feature has been added to the SCARF accelerator.
- `bugfix`: A bug has been fixed.
- `docs`: Documentation has been updated or added.
- `test`: Test cases as well as test related infrastructure have been added or modified.
  - DO NOT use this tag when `bugfix` or `feat` is also present.
  - Use this tag only when solely modifying test cases.
- `refactor`: Code has been refactored without changing its functionality.
- `chore`: Routine tasks such as code formatting, dependency updates, etc.
- `deps`: Python dependency updates or package management changes.
- `saes`: Changes specifically to SAES (Scene-Adaptive Early-Stopping) components.
- `fsdr`: Changes specifically to FSDR (Feature-Similarity Depth Reuse) components.
- `sim`: Simulator infrastructure and hardware modeling changes.
- `perf`: Performance optimization and profiling related changes.
