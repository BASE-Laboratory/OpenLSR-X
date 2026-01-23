from openlsrx.cli import MyLightningCLI
from openlsrx.data.dataset import I13XCTDataModule
from openlsrx.models.srgan import SRGAN


def porespy_cli():
    MyLightningCLI(SRGAN, I13XCTDataModule)


if __name__ == "__main__":
    porespy_cli()
