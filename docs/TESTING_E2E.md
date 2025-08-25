# End-to-End (E2E) and User Acceptance (UAT) Testing

This repo includes lightweight E2E/UAT tests that exercise the public CLI in realistic flows.

- Location: `tests/e2e/`
  - `test_cli_end_to_end.py` — happy-path CLI flows (examples, NER JSON, generate to file, model list)
  - `test_cli_errors_and_edges.py` — error handling and edge cases (no input, conflicting flags, PDF behavior)
  - `test_cli_performance.py` — lightweight performance and UAT smoke

## How to run locally

- Run all E2E tests:

```bash
pytest -m e2e
```

- Exclude performance test (faster):

```bash
pytest -m e2e -m "not performance"
```

- Run UAT-style smoke:

```bash
pytest -m uat
```

- Just this folder:

```bash
pytest tests/e2e -v
```

## CI integration

- `pytest.ini` already includes `tests/` in `testpaths` and defines strict markers including `e2e`, `uat`, and `performance`.
- The default CI run executes all tests; these E2E tests are designed to be fast and have no external dependencies. The performance test is capped and lenient for CI but can be deselected via markers if needed.

## User Acceptance Testing

For scriptable UAT scenarios and golden-output checks, see `docs/UAT.md`.

## Notes

- PDF handling in `inference ner` requires `pypdf`. If unavailable, the CLI prints a clear error message; tests are written to pass in either case.
- Generation uses small defaults through `TextGenerator` and adapters; no large downloads are required for these tests.
