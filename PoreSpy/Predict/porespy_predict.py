from pytorch_lightning.cli import LightningCLI
from srgan import SRGAN
from dataset import PoreSpyDataModule_2x


def porespy_cli():
    LightningCLI(SRGAN, PoreSpyDataModule_2x)


if __name__ == "__main__":
    porespy_cli()
