#!/bin/bash

export SKIPSLOW=true
python -m unittest $1 $2 $3
