#!/bin/bash
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
/data/algo/algo_pyproxy_agent -envname $ENV_NAME -etcd $ETCD_ADDR -lookupaddr $LOOKUPD_ADDR  /data/algo/run.sh

