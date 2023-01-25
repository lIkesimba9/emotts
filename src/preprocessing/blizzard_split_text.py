from pathlib import Path
from shutil import copy

import click
from tqdm import tqdm


def split_text_file(filepath: Path, save_dir: Path, book_name: str):
    with open(filepath) as fil:
        for line in fil:
            sentence_number, sentence = line.split('|')[0], line.split('|')[1]
            open(save_dir / (book_name + "_" + filepath.stem + "_" + sentence_number + ".txt"), 'wt').write(sentence)
            


@click.command()
@click.option("--input-dir", type=Path, required=True,
              help="Directory to move audio from.")
@click.option("--output-dir", type=Path, required=True,
              help="Directory to move audio to.")
def main(input_dir: Path, output_dir: Path) -> None:
    files_total = 0
    for dir_path in tqdm(input_dir.iterdir()):
        for filepath in dir_path.iterdir():
            new_dir_path = (output_dir / (dir_path.name + "_" + filepath.stem))
            new_dir_path.mkdir(exist_ok=True, parents=True)
            split_text_file(filepath, new_dir_path, dir_path.name)
            files_total += 1

    print(f"{files_total} files were copied to {output_dir}")


if __name__ == "__main__":
    main()