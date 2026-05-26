# Contributing

Velocity-FL is a solo project; there's no contributor queue, no review rota, no issue triage. If you're reading this because GitHub linked you here from a PR: thanks for the interest, but drive-by patches may sit unreviewed.

If you do want to send one anyway:

- Install with `uv sync` (see `README.md` for the full bootstrap).
- Run `make validate` before opening a PR — that's the same matrix CI runs.
- Architecture and design notes live in `docs/` (rendered at the project's GitHub Pages site).
- Keep commits conventional (`feat:`, `fix:`, `ci:`, etc.) and scoped to one concern.
