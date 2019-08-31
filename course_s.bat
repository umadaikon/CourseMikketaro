@echo off
set local
set baseDir=C:\openpose
set imgDir=C:\openpose\wallimg\%1
set csvDir=C:\openpose\holdposition\%1.csv
set height=%2
set user=%3

cd C:\openpose\generateCourse

python holdSelect_s.py %imgDir% %csvDir% %user% %height%

cd /d %~dp0