from pytorch_lightning.cli import LightningCLI

from openlsrx.data.dataset import PoreSpyDataModule_2x
from openlsrx.models.srgan import SRGAN


# def porespy_cli():
#     LightningCLI(SRGAN, PoreSpyDataModule)


def porespy_cli():
    LightningCLI(SRGAN, PoreSpyDataModule_2x)


if __name__ == "__main__":
    porespy_cli()
