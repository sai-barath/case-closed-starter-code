# Agent Development Guide

## Build/Test Commands

Never run commands let me do it!!!!

## Code Style (Python)
- **Python**: 3.13+, use `uv` for dependency management
- **Imports**: Standard lib → third-party → local (Flask, requests, pytest from pyproject.toml)
- **Types**: Use type hints (`tuple[int, int]`, `Optional['Agent']`, `list[tuple[int, int]]`)
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants
- **Strings**: Double quotes preferred, f-strings for formatting
- **Error handling**: Check conditions before operations (e.g., `if not self.alive: return False`)
- **Comments**: Docstrings for classes/methods only when needed; avoid inline comments unless complex

## Code Style (Go)
- **Go**: Standard formatting with `gofmt`, camelCase for unexported, PascalCase for exported
- **Structs**: JSON tags required for API payloads (`json:"field_name"`)
- **Error handling**: Check errors immediately after operations, return early on error

## Game Logic
- Board: 18x20 torus (wraparound), use `_torus_check()` for coordinates
- Agents: Start with 2-cell trail, 3 boosts; trails grow permanently, no shrinking
- Collisions: Self-collision and trail-collision are fatal; head-on collision kills both (draw)
- Boost: `:BOOST` suffix on move (e.g., `"RIGHT:BOOST"`) moves twice, costs 1 boost
- Flask API: Endpoints `/`, `/send-state` (POST), `/send-move` (GET), `/end` (POST)
