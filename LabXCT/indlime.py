from pytorch_lightning.cli import LightningCLI
from srgan import SRGAN
from dataset import IndLimeDataModule


def indlime_cli():
    LightningCLI(SRGAN, IndLimeDataModule)


if __name__ == "__main__":
    indlime_cli()
