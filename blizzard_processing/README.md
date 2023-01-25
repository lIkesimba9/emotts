# Byers Preprocessing

## Data

put books [after preprocessing](https://github.com/Tomiinek/Blizzard2013_Segmentation) to data/books

## Prerequisites

* Docker: 20.10.7

## Usage

```bash
docker build --rm --tag byers ./blizzard_processing
docker run --rm -it -v $(pwd):/emotts/repo byers
```
