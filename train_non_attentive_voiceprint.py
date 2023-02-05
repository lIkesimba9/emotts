import argparse

from src.trainer_voiceprint_adversarial import Trainer as TrainerAdv
from src.train_config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="configuration file path"
    )
    args = parser.parse_args()
    config = load_config(args.config)

    trainer = TrainerAdv(config)
    trainer.train()


if __name__ == "__main__":
    main()
