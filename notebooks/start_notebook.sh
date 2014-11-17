#!/bin/sh

export DYLD_FALLBACK_LIBRARY_PATH=/Users/davekale/anaconda/lib
export PYTHONPATH=../
ipython notebook $1
