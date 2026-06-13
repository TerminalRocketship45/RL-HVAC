# Canonical CityLearn env setup. CityLearn 2.3.1 pins openstudio<=3.3.0 (unsatisfiable
# on PyPI for Python 3.11), so it is installed --no-deps after its deps come from the yml.
conda env create -f "$PSScriptRoot/environment-citylearn.yml"
conda run -n rlhvac-citylearn pip install "CityLearn==2.3.1" --no-deps
conda run -n rlhvac-citylearn python -c "from citylearn.citylearn import CityLearnEnv; from citylearn.wrappers import StableBaselines3Wrapper; print('citylearn ok')"
