from pytorch_lightning.cli import LightningCLI

from openlsrx.data.dataset import I13XCTDataModule
from openlsrx.models.srgan import SRGAN


def i13_xct_cli():
    LightningCLI(SRGAN, I13XCTDataModule)


if __name__ == "__main__":
    i13_xct_cli()
