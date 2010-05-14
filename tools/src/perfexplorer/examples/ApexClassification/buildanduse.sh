#!/bin/bash

buildClassifier jaguar-apex-unoptimized-ratios-temporal.csv O3-temporal-svm.class
buildClassifier jaguar-apex-unoptimized-ratios-spatial.csv O3-spatial-svm.class

echo "-O3 temporal"
useClassifier O3-temporal-svm.class jaguar-bench-ratios.csv
echo "-O3 spatial"
useClassifier O3-spatial-svm.class jaguar-bench-ratios.csv
