# traincheck

![PyPI version](https://img.shields.io/pypi/v/traincheck.svg)

A configuration linter for distributed GPU training that catches misconfigurations before they cause failures.

* [GitHub](https://github.com/darkleia/traincheck/) | [PyPI](https://pypi.org/project/traincheck/) | [Documentation](https://darkleia.github.io/traincheck/)
* Created by [Victoria Besedina](-) | GitHub [@darkleia](https://github.com/darkleia) | PyPI [@darkleia](https://pypi.org/user/darkleia/)
* MIT License

## Features

* TODO

## Documentation

Documentation is built with [Zensical](https://zensical.org/) and deployed to GitHub Pages.

* **Live site:** https://darkleia.github.io/traincheck/
* **Preview locally:** `just docs-serve` (serves at http://localhost:8000)
* **Build:** `just docs-build`

API documentation is auto-generated from docstrings using [mkdocstrings](https://mkdocstrings.github.io/).

Docs deploy automatically on push to `main` via GitHub Actions. To enable this, go to your repo's Settings > Pages and set the source to **GitHub Actions**.

## Development

To set up for local development:

```bash
# Clone your fork
git clone git@github.com:your_username/traincheck.git
cd traincheck

# Install in editable mode with live updates
uv tool install --editable .
```

This installs the CLI globally but with live updates - any changes you make to the source code are immediately available when you run `traincheck`.

Run tests:

```bash
uv run pytest
```

Run quality checks (format, lint, type check, test):

```bash
just qa
```

## Author

traincheck was created in 2026 by Victoria Besedina.

Built with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
