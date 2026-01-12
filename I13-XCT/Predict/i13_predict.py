from lightningcli import MyLightningCLI
from srgan import SRGAN
from dataset import I13XCTDataModule


def porespy_cli():
    MyLightningCLI(SRGAN, I13XCTDataModule)


if __name__ == "__main__":
    porespy_cli()
