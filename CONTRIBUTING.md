# Contributing to chatstore

Thank you for taking the time to contribute! Whether it's a bug report, a feature suggestion, or a pull request — all contributions are welcome.

---

## Reporting a Bug

1. **Search first** — check [existing issues](../../issues) to avoid duplicates.
2. **Open a new issue** using the **Bug Report** template.
3. Include the following:
   - Your Python version (`python --version`)
   - Your chatstore version (`pip show chatstore`)
   - Whether you are using the `[semantic]` extra
   - A **minimal reproducible example** — the shortest code that triggers the bug
   - The full error traceback

Good bug reports get fixed faster. The more specific, the better.

---

## Suggesting a Feature

1. Check the [roadmap in the README](README.md#roadmap) — it may already be planned.
2. Open an issue with the **Feature Request** label.
3. Describe:
   - The problem you are trying to solve
   - Your proposed solution
   - Any alternatives you considered

---

## Making a Pull Request

### Setup

```bash
git clone https://github.com/your-username/chatstore.git
cd chatstore
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -e ".[dev]"
```

### Before you submit

1. **Write or update tests** for your change in `tests/`.
2. **Run the full test suite** — all tests must pass:
   ```bash
   pytest tests/ -v
   ```
3. Keep changes focused — one PR per fix or feature.
4. Write a clear PR description explaining *what* and *why*.

### Commit message style

Use short, imperative messages:
```
fix: return None when session not found
feat: add async support for ChatService
docs: clarify sliding window behaviour in README
```

---

## Code Style

- Follow existing code style — no reformatting unrelated lines
- No new dependencies in the core package without discussion
- Keep `chatstore[semantic]` imports lazy (inside `if enable_semantic_search`) so v1 stays lightweight

---

## Questions?

Open a [GitHub Discussion](../../discussions) or file an issue with the **question** label.
