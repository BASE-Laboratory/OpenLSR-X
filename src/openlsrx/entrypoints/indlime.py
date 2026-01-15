from pytorch_lightning.cli import LightningCLI

from openlsrx.data.dataset import IndLimeDataModule
from openlsrx.models.srgan import SRGAN


def indlime_cli():
    LightningCLI(SRGAN, IndLimeDataModule)


if __name__ == "__main__":
    indlime_cli()
