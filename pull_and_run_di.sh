#!/bin/sh

WORKdir=$PWD
echo " --- ${WORKdir} --- "

cd ${WORKdir}/webapp

docker pull aliheadou/tagssuggestionwebapp:v0

docker run -p 5000:5000 aliheadou/tagssuggestionwebapp:v0
