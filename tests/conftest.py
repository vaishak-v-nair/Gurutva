"""Skip the research suite when the research stack is not installed.

CI found this within sixty seconds of existing, which is the whole argument
for CI. `src/forge_data.py` imports `h5py` and `pandas` at module level, and
h5py was declared in no requirements file at all -- it was simply installed on
the one laptop the 172 tests had ever run on. Eight test modules reach
forge_data through `inversion`, `mesh`, `published_model` or `s25_model`, so
on any clean machine they did not fail, they failed to *collect*, taking the
whole run down with them.

These are research tests in the honest sense: they read the delivered Utah
FORGE archive, which is gitignored and fetched rather than committed. They
cannot run on a fresh clone whatever is installed. So they are skipped as a
set, by the presence of h5py, rather than pretended to be product tests.

The product suite -- gates, verdict, prior, survey, joint, CLI, window, the
frozen binary -- has no such dependency and runs everywhere.
"""

import importlib.util

# Reach forge_data (directly or transitively) and therefore need h5py+pandas
# AND the delivered archive.
RESEARCH_SUITE = [
    "test_bridge.py",
    "test_forge_data.py",
    "test_inversion.py",
    "test_modes.py",
    "test_posterior.py",
    "test_prism.py",
    "test_provenance.py",
    "test_s25.py",
]

collect_ignore = []

if importlib.util.find_spec("h5py") is None:
    collect_ignore.extend(RESEARCH_SUITE)


def pytest_report_header(config):
    if collect_ignore:
        return (f"research suite skipped ({len(collect_ignore)} modules): "
                f"pip install -r requirements-research.txt")
    return None
