S4D Diarization Toolkit developed by Lium group

INTRODUCTION
============
SIDEKIT for diarization (s4d as short name) is an open source package extension of SIDEKIT for Speaker diarization .
It officially developed by LIUM group from France.

URL is aviliable at: https://lium.univ-lemans.fr/s4d/  
URL for SIDEKIT: https://projets-lium.univ-lemans.fr/s4d/  
Official github:  https://git-lium.univ-lemans.fr/Meignier/s4d  

I used their toolkit for doing some diarization experiment using kaldi to extract features

INSTALLATION
============
I recommend you use anaconda, and create a new working envrionment by:  
conda env create -f environment.yml -p <your env path>


Other things
============
I have deleted some libs that I dont need for building docker, if you want full version of s4d, please go to the official github.

example1.wav and example2.wav is some example wavform for testing.  
you can directly using "gl_bic_ahc_viterbi.py" if you want  
algo.py, is editted version for my docker build.

start.sh  
run.sh  
Dockerfile   
algo_pyproxy_agent  
server.py  
test.py  

These above files are all for docker build. You can try it if you want.

Yongyu Gao

