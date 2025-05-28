# -*- coding: utf-8 -*-
#!/bin/bash

cd ./structures

echo "Starting calculations..."

for j in {101..200}
do
    cd ./$j

    file_name=$(ls | grep "one_dimension_$j.inp")

    echo "Running calculation for $j with file $file_name"

    cp2k.ssmp ${file_name} > ./one_dimension.out 2>&1

    python ../../read_bands.py

    rm ./one_dimension-RESTART*

    cd ../

    echo "Finished calculation for $j"
done

echo "All calculations are done."