#!bin/bash

docker build --rm --tag emotts ./vctk_preprocessing

docker run --rm --shm-size="8G" -it -v $(pwd):/emotts/repo -v /media/public/nikita_u/data/:/emotts/repo/data emotts

