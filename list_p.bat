@echo off
:set local
set baseDir=C:\openpose
set imgDir=C:\openpose\wallimg\%1
set csvDir=C:\openpose\holdposition\%1.csv
set height=%2

cd C:\openpose\generateCourse

python listGenerate1_p.py %imgDir% %csvDir% %height%
python listGenerate2_p.py %imgDir% %csvDir% %height%
python listGenerate3_p.py %imgDir% %csvDir% %height%
python listGenerate4_p.py %imgDir% %csvDir% %height%
python listGenerate5_p.py %imgDir% %csvDir% %height%
python listGenerate6_p.py %imgDir% %csvDir% %height%

cd /d %~dp0