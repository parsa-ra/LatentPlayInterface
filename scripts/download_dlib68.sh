#!/bin/bash

mkdir -p models && cd models
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
cd .. 
export LP_DLIB_PREDICTOR="$(pwd)/models/shape_predictor_68_face_landmarks.dat"