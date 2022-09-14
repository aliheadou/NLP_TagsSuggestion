#!/bin/sh

WORKdir=$PWD
echo " --- ${WORKdir} --- "

cd ${WORKdir}/webapp

docker build -t tagssuggestionwebapp .

docker run -p 5000:5000 tagssuggestionwebapp