#!/bin/bash


/build/video-demo/video-demo --input /tmp/image/image$1.jpg --data /affdex-sdk/data --draw=0 --numFaces=1 $@ &>/failure