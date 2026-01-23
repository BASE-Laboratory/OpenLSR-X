from pytorch_lightning.cli import LightningCLI

from openlsrx.data.dataset import BentSandDataModule
from openlsrx.models.srgan import SRGAN


def bentsand_cli():
    LightningCLI(SRGAN, BentSandDataModule)


if __name__ == "__main__":
    bentsand_cli()
