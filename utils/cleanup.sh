#!/usr/bin/env bash

echo "Converting code to Black style (see black.readthedocs.io)"

black -l 120 ../experiments/*.py
black -l 120 ../experiments/debug/*.py
black -l 120 ../ginkgo_rl/agents/*.py
black -l 120 ../ginkgo_rl/envs/*.py
black -l 120 ../ginkgo_rl/eval/*.py
black -l 120 ../ginkgo_rl/utils/*.py

echo "All done, have a nice day!"
