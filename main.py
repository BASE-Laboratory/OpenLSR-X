from pytorch_lightning.cli import LightningCLI
from srgan import SRGAN
from dataset import PoreSpy3D_Volume


def srgan_cli():
    LightningCLI(SRGAN, PoreSpy3D_Volume)


if __name__ == "__main__":
    srgan_cli()
