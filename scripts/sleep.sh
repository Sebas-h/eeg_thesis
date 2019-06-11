#!/usr/bin/env bash

echo $1 + $2

python -m experiment.run -s $1 -i $2
