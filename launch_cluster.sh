
#! /bin/bash

#sbatch --export=OFFLINE=0,KDE=0,BETA=0,OFFSET=0,T=1 cluster_code.sbatch
#sleep 1
sbatch --export=OFFLINE=1,KDE=0,BETA=0,OFFSET=0,T=1 cluster_code.sbatch
sleep 1
for beta in 1; do
  for off in 0.0 0.2 0.5 0.8; do
    for t in 1.0 0.5 0.2; do
      sbatch --export=OFFLINE=1,KDE=1,BETA=$beta,OFFSET=$off,T=$t cluster_code.sbatch
      sleep 1
    done
  done
done
