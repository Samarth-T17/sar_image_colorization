#!/bin/bash
kaggle datasets download -d requiemonk/sentinel12-image-pairs-segregated-by-terrain
unzip sentinel12-image-pairs-segregated-by-terrain.zip > /dev/null 2>&1 && rm sentinel12-image-pairs-segregated-by-terrain.zip > /dev/null 2>&1