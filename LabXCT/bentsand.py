from pytorch_lightning.cli import LightningCLI
from srgan import SRGAN
from dataset import BentSandDataModule


def bentsand_cli():
    LightningCLI(SRGAN, BentSandDataModule)


if __name__ == "__main__":
    bentsand_cli()
