# source ~/.bashrc
# cd plankton/torch
# git pull
# ./classifier.sh

cp feeder.py ../..
cp crack.lua ../..

cd ../..
rm model_3.t7
wget http://54.173.20.85:8000/model_3.t7
nano feeder.py

python feeder.py
nohup cat filelist.txt|th crack.lua>output.txt &
