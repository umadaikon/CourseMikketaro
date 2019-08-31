@echo off
:set local
set baseDir=C:\openpose
set imgDir=C:\openpose\wallimg\%1
set csvDir=C:\openpose\holdposition\%1.csv
set height=%2

cd C:\openpose\generateCourse

python listGenerate.py %imgDir% %csvDir% %height%

cd /d %~dp0