"""Single source of truth for this project's version.

Not an installable package (no setup.py/pyproject.toml -- see
dev/architecture.md), so this is a standalone module per PEP 440/semver
convention rather than a package's __init__.py attribute. Update this
alongside a new CHANGELOG.md entry.
"""

__version__ = '1.1.0'
