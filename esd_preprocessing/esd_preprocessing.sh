#!bin/bash

docker build --rm --tag esd ./esd_preprocessing
docker run --env language=english --rm -it -v $(pwd):/emotts/repo esd

