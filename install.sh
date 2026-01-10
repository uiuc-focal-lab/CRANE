pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python jax jaxlib shap pytensor
pip uninstall -y omegaconf hydra-core  # if you don't need them

pip install --upgrade --force-reinstall --no-cache-dir numpy==1.26.4
pip install --upgrade --force-reinstall --no-cache-dir antlr4-python3-runtime==4.11.1


cd syncode/
pip install -e .

cd ../src/itergen/iter_syncode/
pip install -e .

cd ../../itergen/
pip install -e .

cd ../math_evaluator/latex2sympy/
pip install -e .