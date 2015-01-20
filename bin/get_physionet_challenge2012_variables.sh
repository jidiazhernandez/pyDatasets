#!/bin/bash

DATADIR=$1
OUTDIR=$2

CURRDIR=`pwd`
cd $DATADIR

cat set-a/*.txt set-b/*.txt | sed 's/.*,\(.*\),.*$/\1/g' | sort | uniq > $OUTDIR/variables-from-data.txt
cd $CURRDIR
