"""
This is a file to describe the creation of the SRGAN model I am using
"""

import pytorch_lightning as pl
import torch
from torch import nn, optim
import imageio
import os
from submodels import Generator, Discriminator
from loss import VGGLoss
import tifffile
import re


class SRGAN(pl.LightningModule):

    def __init__(
        self,
        learning_rate,
        adaptive_size,
        accumulate_n_batches,
        super_res_factor,
        num_blocks=10,
        use_vgg_loss=False,
        vgg_loss_weight=0.006,
        vgg_layers=18,
        sample_id=str(),
        strict_loading=True,
        sigmoid_k=1.0,
        lr_scheduler_type="constant",
        lr_scheduler_args=None,
    ):
        super().__init__()

        # sample name
        self.sample_id = sample_id

        # hyperparameters
        self.learning_rate = learning_rate
        self.adaptive_size = adaptive_size
        self.accumulate_n_batches = accumulate_n_batches
        self.sigmoid_k = float(sigmoid_k)

        # dictionary of accepted LR schedulers
        lr_scheduler_dict = dict(
            {"constant": {"lr": float(learning_rate)}, "CAWR": lr_scheduler_args}
        )
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_scheduler = lr_scheduler_dict[str(lr_scheduler_type)]

        # Generator hyperparameters
        self.super_res_factor = super_res_factor
        self.use_vgg_loss = use_vgg_loss
        self.num_blocks = num_blocks
        self.vgg_loss_weight = float(vgg_loss_weight)
        self.vgg_layers = vgg_layers

        # networks

        self.gen = Generator(
            num_blocks=num_blocks,
            super_res_factor=super_res_factor,
            sigmoid_k=sigmoid_k,
        )
        self.disc = Discriminator(adaptive_size=adaptive_size)

        # losses

        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.vgg = VGGLoss(self.vgg_layers)

        # optimizers

        self.automatic_optimization = False

        # loss lists

        self.gen_losses = torch.Tensor([])
        self.disc_losses = torch.Tensor([])

        # strict loading
        self.strict_loading = strict_loading

    def training_step(self, train_batch, batch_idx):
        """
        Loads training batch and batch index

        training batch -> torch.Tensor([B,H,W])
        unsqueeze(1) -> torch.Tensor([B,C,H,W])

        Gradient Accumulation is used every accumulate_n_batches:
        1) Gen/Disc Weights are initialised
        2) Gen gradients/weights not update in Disc Training
        3) Disc weights updated, gradients not zeroed here
        4) Disc gradients not re-calculated in Gen Training
            - gradients at end of disc training, same for input on next training step
            - necessary for accurate grad acc
        5) Gen gradients/weights updated
            - gradient at end of gen training, same for input on next training step
            - necessary for accurate grad acc
        6) Gen gradients zeroed if multiple of accumulate_n_batches
        7) Disc gradients zeroed if multiple of accumulate_n_batches
        """
        low_res, high_res = train_batch
        low_res = low_res.unsqueeze(1)
        high_res = high_res.unsqueeze(1)

        opt_gen, opt_disc = self.optimizers()
        gen_scheduler, disc_scheduler = self.lr_schedulers()

        gen = self.gen
        disc = self.disc
        mse = self.mse
        bce = self.bce
        vgg = self.vgg
        # l1 = self.l1

        # train discriminator

        self.toggle_optimizer(opt_disc)
        fake = gen(low_res).detach()  # detach to prevent gradients being calculated
        disc_real = disc(high_res)
        disc_fake = disc(fake)

        disc_loss_real = bce(
            disc_real,
            torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real),
        )

        disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))

        disc_loss = (disc_loss_fake + disc_loss_real) / self.accumulate_n_batches

        self.manual_backward(disc_loss)  # calculate gradients

        # update weights & zero grad every accumulate_n_batches
        if (batch_idx + 1) % self.accumulate_n_batches == 0:
            opt_disc.step()  # update disc weights
            opt_disc.zero_grad()  # zero gradients
            disc_scheduler.step()
        self.untoggle_optimizer(opt_disc)

        # save disc gradients
        disc_grads = [param.grad for param in disc.parameters()]

        # train generator

        self.toggle_optimizer(opt_gen)
        fake = gen(low_res)

        # call disc with grad as reset after gen training
        disc_fake = disc(fake)

        adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
        mse_loss = mse(fake, high_res)
        # l1_loss = l1(fake, high_res)

        # only calculate vgg_loss if 'use_vgg_loss' is set to True
        vgg_loss = (
            vgg(fake, high_res) * self.vgg_loss_weight if self.use_vgg_loss else 0
        )

        gen_loss = (adversarial_loss + mse_loss + vgg_loss) / self.accumulate_n_batches

        self.manual_backward(gen_loss)
        # update weights every accumulate_n_batches
        if (batch_idx + 1) % self.accumulate_n_batches == 0:
            opt_gen.step()
            opt_gen.zero_grad()
            gen_scheduler.step()
        self.untoggle_optimizer(opt_gen)

        # reset disc gradients to before gen training
        for param, grad in zip(disc.parameters(), disc_grads):
            param.grad = grad

        # append loss after every desired batch passsed through
        if self.global_rank == 0 and (batch_idx + 1) % self.accumulate_n_batches == 0:
            disc_loss = torch.Tensor([disc_loss])
            gen_loss = torch.Tensor([gen_loss])

            self.disc_losses = torch.cat((self.disc_losses, disc_loss), axis=0)
            self.gen_losses = torch.cat((self.gen_losses, gen_loss), axis=0)

    def on_train_epoch_end(self):
        """
        Defines what occurs at end of training epoch
        1) Take average of losses from each batch processed in epoch
        2) Write disc_loss and gen_loss to files
        3) Reset loss lists to empty/zero
        """
        # Input dim of loss lists is just length dataset_size/batch_size
        avg_disc_loss = torch.mean(self.disc_losses).item()
        avg_gen_loss = torch.mean(self.gen_losses).item()

        if self.global_rank == 0:
            with open("gen_loss.txt", "a") as f:
                f.write(str(avg_gen_loss))
                f.write("\n")

            with open("disc_loss.txt", "a") as f:
                f.write(str(avg_disc_loss))
                f.write("\n")

        self.disc_losses = torch.Tensor([])
        self.gen_losses = torch.Tensor([])

    @torch.no_grad()
    def validation_step(self, val_batch, batch_idx):
        """
        Performs the validation step. Passes low-res input through
        the partially trained model.
        Validation performed on single GPU

        val_batch -> torch.Tensor([B,H,W])
        """

        # disable training

        if self.global_rank == 0:
            low_res, high_res = val_batch

            low_res = low_res.unsqueeze(1)
            high_res = high_res.unsqueeze(1)
            # BHW -> BCHW

            upsampled = self.gen(low_res)

            # want to save image as .tiff as floating points
            up_tiff = "upsampled_" + str(batch_idx) + ".tiff"
            high_tiff = "high_" + str(batch_idx) + ".tiff"
            low_tiff = "low_" + str(batch_idx) + ".tiff"

            # take 1st item from the batch and squeeze channel
            # BCHW -> CHW -> HW
            upsampled = upsampled[0].squeeze(0).cpu().to(torch.float32).numpy()
            low_res = low_res[0].squeeze(0).cpu().numpy()
            high_res = high_res[0].squeeze(0).cpu().numpy()

            # make empty saved folder
            if not os.path.isdir("saved"):
                os.mkdir("saved")

            epoch = self.current_epoch
            sanity = self.trainer.sanity_checking

            if sanity:
                os.mkdir("sanity") if not os.path.isdir("sanity") else None
                os.chdir("sanity")
                (
                    os.mkdir(f"epoch_{epoch}")
                    if not os.path.isdir(f"epoch_{epoch}")
                    else None
                )
                imageio.imwrite(f"epoch_{epoch}/{up_tiff}", upsampled)
                imageio.imwrite(f"epoch_{epoch}/{high_tiff}", high_res)
                imageio.imwrite(f"epoch_{epoch}/{low_tiff}", low_res)
                os.chdir("../")

            elif not sanity:
                if not os.path.isdir(f"saved/epoch_{epoch}"):
                    os.mkdir(f"saved/epoch_{epoch}")
                imageio.imwrite(f"saved/epoch_{epoch}/{up_tiff}", upsampled)
                imageio.imwrite(f"saved/epoch_{epoch}/{high_tiff}", high_res)
                imageio.imwrite(f"saved/epoch_{epoch}/{low_tiff}", low_res)

    @torch.no_grad()
    def test_step(self, test_batch, batch_idx):
        """
        Basic test step, saving the high-res, low-res and upsampled images
        to a folder called 'test_saved' in the current directory.

        Performs the test step. Passes low-res input through
        the trained model from the testing dataset.
        Test performed on single GPU

        test_batch -> torch.Tensor([B,H,W])
        """
        low_res, high_res = test_batch
        low_res = low_res.unsqueeze(1)
        high_res = high_res.unsqueeze(1)

        upsampled = self.gen(low_res)

        # want to save image as .tiff as floating points
        up_tiff = "upsampled_" + str(batch_idx) + ".tiff"
        high_tiff = "high_" + str(batch_idx) + ".tiff"
        low_tiff = "low_" + str(batch_idx) + ".tiff"

        # take 1st item from the batch and squeeze channel
        # BCHW -> CHW -> HW
        upsampled = upsampled[0].squeeze(0).cpu().to(torch.float32).numpy()
        low_res = low_res[0].squeeze(0).cpu().numpy()
        high_res = high_res[0].squeeze(0).cpu().numpy()

        if not os.path.isdir("test_saved"):
            os.mkdir("test_saved")
        os.chdir("test_saved")

        # save the images in the test_saved folder
        imageio.imwrite(f"{up_tiff}", upsampled)
        imageio.imwrite(f"{high_tiff}", high_res)
        imageio.imwrite(f"{low_tiff}", low_res)

        os.chdir("../")  # back up to original folder

    @torch.no_grad()
    def predict_step(self, predict_batch, batch_idx):
        """
        Performs the prediction step. Passes low-res input through
        the trained model.
        Prediction performed on single GPU

        batch -> torch.Tensor([B,H,W])
        """
        low_res, high_res = predict_batch  # Both BHW

        low_res = low_res.unsqueeze(1)  # BHW -> BCHW to pass through model

        predicted = self.gen(low_res)

        # save the predicted image as a .tiff file in a 'predicted_' folder

        pred_axis = getattr(self.trainer.datamodule, "predict_axis", "unknown")
        pred_folder = (
            f"predicted_{pred_axis}" if pred_axis != "unknown" else "predicted"
        )

        if not os.path.isdir(pred_folder):
            os.mkdir(pred_folder)
        os.chdir(pred_folder)

        high_tiff = (
            f"{self.sample_id}_high_{batch_idx}.tiff"
            if self.sample_id
            else f"high_{batch_idx}.tiff"
        )
        low_tiff = (
            f"{self.sample_id}_low_{batch_idx}.tiff"
            if self.sample_id
            else f"low_{batch_idx}.tiff"
        )
        predicted_tiff = (
            f"{self.sample_id}_predicted_{batch_idx}.tiff"
            if self.sample_id
            else f"predicted_{batch_idx}.tiff"
        )

        # Ensure all HW
        high_res = high_res[0].cpu().numpy()  # BHW -> HW
        low_res = low_res[0].squeeze(0).cpu().numpy()  # BCHW -> HW
        predicted = (
            predicted[0].squeeze(0).cpu().to(torch.float32).numpy()
        )  # BCHW -> HW

        imageio.imwrite(f"{high_tiff}", high_res)
        imageio.imwrite(f"{low_tiff}", low_res)
        imageio.imwrite(f"{predicted_tiff}", predicted)

        if not os.path.isdir("predicted"):
            os.mkdir("predicted")

        if not os.path.isdir("low_res"):
            os.mkdir("low_res")

        if not os.path.isdir("high_res"):
            os.mkdir("high_res")

        # move all predicted_*.tiff files to the predicted folder
        os.system(f"mv {predicted_tiff} ./predicted/")

        # move all low_*.tiff files to the low_res folder
        os.system(f"mv {low_tiff} ./low_res/")

        # move all high_*.tiff files to the high_res folder
        os.system(f"mv {high_tiff} ./high_res/")

        os.chdir("../")  # back up to original folder

    def on_predict_epoch_end(self):
        """
        Defines what occurs at end of prediction epoch

        1a) Move into predicted folder
        1b) Create a single stack of the predicted_*.tiff files

        2a) Move into low_res folder
        2b) Create a single stack of the low_*.tiff files

        3a) Move into high_res folder
        3b) Create a single stack of the high_*.tiff files

        """

        # Change into the predicted_{pred_axis} folder
        pred_axis = getattr(self.trainer.datamodule, "predict_axis", "unknown")
        pred_folder = (
            f"predicted_{pred_axis}" if pred_axis != "unknown" else "predicted"
        )

        os.chdir(pred_folder)

        # define the file formats depending on sample_id

        if self.sample_id:
            pred_stack_name = f"{self.sample_id}_predicted_stack.tiff"
            pred_pattern = re.compile(rf"{self.sample_id}_predicted_(\d+)\.tiff")

            low_stack_name = f"{self.sample_id}_low_res_stack.tiff"
            low_pattern = re.compile(rf"{self.sample_id}_low_(\d+)\.tiff")

            high_stack_name = f"{self.sample_id}_high_res_stack.tiff"
            high_pattern = re.compile(rf"{self.sample_id}_high_(\d+)\.tiff")

        else:
            pred_stack_name = "predicted_stack.tiff"
            pred_pattern = re.compile(r"predicted_(\d+)\.tiff")

            low_stack_name = "low_res_stack.tiff"
            low_pattern = re.compile(r"low_(\d+)\.tiff")

            high_stack_name = "high_res_stack.tiff"
            high_pattern = re.compile(r"high_(\d+)\.tiff")

        # predicted folder first
        os.chdir("predicted")
        pred_path = os.getcwd()

        if self.global_rank == 0:
            # Get all predicted_*.tiff files and sort them numerically
            tiff_files = sorted(
                [f for f in os.listdir(pred_path) if re.match(pred_pattern, f)],
                key=lambda x: int(re.search(pred_pattern, x).group(1)),
            )

            # Output path for the stack
            output_path = os.path.join(os.pardir, pred_stack_name)

            # Write the stack incrementally (memory-efficient)
            with tifffile.TiffWriter(output_path) as tiff:
                for f in tiff_files:
                    img_path = os.path.join(pred_path, f)
                    img = tifffile.imread(img_path)
                    tiff.write(img)

        os.chdir("../")  # back up to original folder

        # low_res folder next
        os.chdir("low_res")
        low_res_path = os.getcwd()

        if self.global_rank == 0:
            # Get all low_*.tiff files and sort them numerically
            tiff_files = sorted(
                [f for f in os.listdir(low_res_path) if re.match(low_pattern, f)],
                key=lambda x: int(re.search(low_pattern, x).group(1)),
            )

            # Output path for the stack
            output_path = os.path.join(os.pardir, low_stack_name)

            # Write the stack incrementally (memory-efficient)
            with tifffile.TiffWriter(output_path) as tiff:
                for f in tiff_files:
                    img_path = os.path.join(low_res_path, f)
                    img = tifffile.imread(img_path)
                    tiff.write(img)
        os.chdir("../")

        # high_res folder last
        os.chdir("high_res")
        high_res_path = os.getcwd()

        if self.global_rank == 0:
            # Get all high_*.tiff files and sort them numerically
            tiff_files = sorted(
                [f for f in os.listdir(high_res_path) if re.match(high_pattern, f)],
                key=lambda x: int(re.search(high_pattern, x).group(1)),
            )

            # Output path for the stack in folder above
            output_path = os.path.join(os.pardir, high_stack_name)

            # Write the stack incrementally (memory-efficient)
            with tifffile.TiffWriter(output_path) as tiff:
                for f in tiff_files:
                    img_path = os.path.join(high_res_path, f)
                    img = tifffile.imread(img_path)
                    tiff.write(img)
        os.chdir("../")

    def configure_optimizers(self):
        self.automatic_optimization = False
        gen = self.gen
        disc = self.disc

        opt_gen = optim.Adam(
            gen.parameters(), lr=self.learning_rate, betas=(0.9, 0.999)
        )
        opt_disc = optim.Adam(
            disc.parameters(), lr=self.learning_rate, betas=(0.9, 0.999)
        )

        if self.lr_scheduler_type == "constant":
            gen_scheduler = optim.lr_scheduler.LambdaLR(
                opt_gen, lr_lambda=lambda _step: 1.0
            )
            disc_scheduler = optim.lr_scheduler.LambdaLR(
                opt_disc, lr_lambda=lambda _step: 1.0
            )

        elif self.lr_scheduler_type == "CAWR":
            gen_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt_gen,
                **self.lr_scheduler,
            )
            disc_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt_disc,
                **self.lr_scheduler,
            )

        return [
            {"optimizer": opt_gen, "lr_scheduler": gen_scheduler},
            {"optimizer": opt_disc, "lr_scheduler": disc_scheduler},
        ]
