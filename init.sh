python3 -m venv .venv
source .venv/bin/activate
pip install -r ./requirements/microtorch-requirements.txt
pip install -r ./requirements/testing-requirements.txt
pip install . # to add microtorch as a package for testing