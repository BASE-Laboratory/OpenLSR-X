from pytorch_lightning.cli import LightningCLI
from srgan import SRGAN
from dataset import I13XCTDataModule


def i13_xct_cli():
    LightningCLI(SRGAN, I13XCTDataModule)


if __name__ == "__main__":
    i13_xct_cli()
