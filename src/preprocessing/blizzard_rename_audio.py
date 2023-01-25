from pathlib import Path
from shutil import copy

import click
from tqdm import tqdm


@click.command()
@click.option("--input-dir", type=Path, required=True,
              help="Directory to move audio from.")
@click.option("--output-dir", type=Path, required=True,
              help="Directory to move audio to.")
def main(input_dir: Path, output_dir: Path) -> None:
    files_total = 0
    for dir_path in tqdm(input_dir.iterdir()):
        for filepath in dir_path.iterdir():
            charter_id = filepath.name.split('-')[0]
            new_dir_path = (output_dir / (dir_path.name + "_" + charter_id))
            new_dir_path.mkdir(exist_ok=True, parents=True)
            copy(str(filepath), new_dir_path / (dir_path.name + "_" + filepath.name.replace("-", '_')))
            files_total += 1

    print(f"{files_total} files were copied to {output_dir}")


if __name__ == "__main__":
    main()
