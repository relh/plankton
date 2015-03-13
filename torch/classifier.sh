source ~/.bashrc
cd plankton
git pull

cd torch
cp feeder.py ../..
cd ../..
wget http://54.152.122.132:8000/model_4.t7
rm model_3.t7
mv model_4.t7 model_3.t7
nano feeder.py

python feeder.py
cat filelist.txt|th crack.lua>output.txt

