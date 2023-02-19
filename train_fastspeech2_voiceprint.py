import argparse

from src.trainer_fastspeech2_voiceprint import Trainer
from src.train_config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="configuration file path"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    assert(config.data.energy_norm == config.variance_adapter_params.energy_norm)
    assert(config.data.pitch_norm == config.variance_adapter_params.pitch_norm)
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
