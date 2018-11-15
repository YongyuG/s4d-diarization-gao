#!/bin/bash
source /root/anaconda3/bin/activate s4d
python server.py --host 0.0.0.0 $*
