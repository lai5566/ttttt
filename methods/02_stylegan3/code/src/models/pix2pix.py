# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/models/pix2pix.py
# Pix2Pix backbone for paired image-to-image translation

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.misc as misc
from models.image_translation_networks import define_generator, define_discriminator


class Generator(nn.Module):
    """Pix2Pix Generator adapted to StudioGAN interface

    Uses U-Net architecture for paired image translation.
    Takes input image and generates corresponding output image.

    Args:
        z_dim: Not used in Pix2Pix (uses input images)
        g_shared_dim: Not used
        img_size: Output image size (should match input size)
        g_conv_dim: Base number of filters (ngf)
        apply_attn: Not used in Pix2Pix
        attn_g_loc: Not used
        g_cond_mtd: Not used (conditioning is via input image)
        num_classes: Not used
        g_init: Initialization method
        g_depth: Not used (U-Net depth determined by img_size)
        mixed_precision: Use AMP
        MODULES: Not used
        MODEL: Full config object
    """
    def __init__(self, z_dim, g_shared_dim, img_size, g_conv_dim, apply_attn,
                 attn_g_loc, g_cond_mtd, num_classes, g_init, g_depth,
                 mixed_precision, MODULES, MODEL):
        super(Generator, self).__init__()

        self.img_size = img_size
        self.mixed_precision = mixed_precision
        self.g_conv_dim = g_conv_dim

        # Determine U-Net depth based on image size
        if img_size == 256:
            gen_type = 'unet_256'
        elif img_size == 128:
            gen_type = 'unet_128'
        else:
            # For other sizes, use closest match
            gen_type = 'unet_256' if img_size >= 192 else 'unet_128'

        # Allow override via config
        if hasattr(MODEL, 'pix2pix_gen_type'):
            gen_type = MODEL.pix2pix_gen_type

        norm_type = MODEL.pix2pix_norm if hasattr(MODEL, 'pix2pix_norm') else 'batch'
        use_dropout = MODEL.pix2pix_use_dropout if hasattr(MODEL, 'pix2pix_use_dropout') else True
        init_type = 'normal' if g_init == 'N02' else g_init.lower()

        input_nc = MODEL.pix2pix_input_nc if hasattr(MODEL, 'pix2pix_input_nc') else 3
        output_nc = MODEL.pix2pix_output_nc if hasattr(MODEL, 'pix2pix_output_nc') else 3

        # Create U-Net generator
        self.netG = define_generator(
            gen_type=gen_type,
            input_nc=input_nc,
            output_nc=output_nc,
            ngf=g_conv_dim,
            norm=norm_type,
            use_dropout=use_dropout,
            init_type=init_type,
            init_gain=0.02
        )

    def forward(self, z=None, label=None, eval=False, real_images=None):
        """Forward pass - generates output image from input image

        Args:
            z: Not used (Pix2Pix uses real images as input)
            label: Not used (conditioning is through input image)
            eval: Evaluation mode flag
            real_images: Input images [batch, input_nc, H, W]

        Returns:
            Generated output images [batch, output_nc, H, W]
        """
        with torch.cuda.amp.autocast() if self.mixed_precision and not eval else misc.dummy_context_mgr():
            if real_images is None:
                raise ValueError("Pix2Pix requires real_images as input")

            return self.netG(real_images)


class Discriminator(nn.Module):
    """Pix2Pix Discriminator adapted to StudioGAN interface

    Uses PatchGAN discriminator that classifies image patches.
    Takes CONCATENATED [input, output] to condition on input.

    Args:
        img_size: Input image size
        d_conv_dim: Base number of filters (ndf)
        apply_d_sn: Not used in Pix2Pix
        apply_attn: Not used
        attn_d_loc: Not used
        d_cond_mtd: Not used (conditioning via concatenation)
        aux_cls_type: Not used
        d_embed_dim: Not used
        normalize_d_embed: Not used
        num_classes: Not used
        d_init: Initialization method
        d_depth: Number of discriminator layers
        mixed_precision: Use AMP
        MODULES: Not used
        MODEL: Full config object
    """
    def __init__(self, img_size, d_conv_dim, apply_d_sn, apply_attn,
                 attn_d_loc, d_cond_mtd, aux_cls_type, d_embed_dim,
                 normalize_d_embed, num_classes, d_init, d_depth,
                 mixed_precision, MODULES, MODEL):
        super(Discriminator, self).__init__()

        self.mixed_precision = mixed_precision
        self.d_conv_dim = d_conv_dim

        # Get Pix2Pix-specific config
        n_layers = 3 if d_depth == 'N/A' else int(d_depth)
        disc_type = MODEL.pix2pix_disc_type if hasattr(MODEL, 'pix2pix_disc_type') else 'basic'
        norm_type = MODEL.pix2pix_norm if hasattr(MODEL, 'pix2pix_norm') else 'batch'
        init_type = 'normal' if d_init == 'N02' else d_init.lower()

        input_nc = MODEL.pix2pix_input_nc if hasattr(MODEL, 'pix2pix_input_nc') else 3
        output_nc = MODEL.pix2pix_output_nc if hasattr(MODEL, 'pix2pix_output_nc') else 3

        # Pix2Pix discriminator takes concatenated [input, output]
        # So input channels = input_nc + output_nc
        self.netD = define_discriminator(
            disc_type=disc_type,
            input_nc=input_nc + output_nc,  # Concatenated input
            ndf=d_conv_dim,
            n_layers_D=n_layers,
            norm=norm_type,
            init_type=init_type,
            init_gain=0.02
        )

        self.input_nc = input_nc
        self.output_nc = output_nc

    def forward(self, x, label=None, eval=False, adc_fake=False, input_images=None):
        """Forward pass - discriminate real/fake conditioned on input

        Args:
            x: Target images (real or fake) [batch, output_nc, H, W]
            label: Not used
            eval: Evaluation mode flag
            adc_fake: Not used
            input_images: Conditional input images [batch, input_nc, H, W]

        Returns:
            Dictionary with discriminator outputs (StudioGAN format)
        """
        with torch.cuda.amp.autocast() if self.mixed_precision and not eval else misc.dummy_context_mgr():
            if input_images is None:
                raise ValueError("Pix2Pix discriminator requires input_images for conditioning")

            # Concatenate input and output images
            combined = torch.cat([input_images, x], dim=1)
            adv_output = self.netD(combined)

            return {
                "h": None,
                "adv_output": adv_output,  # Patch-based output
                "embed": None,
                "proxy": None,
                "cls_output": None,
                "label": label,
                "mi_embed": None,
                "mi_proxy": None,
                "mi_cls_output": None,
                "info_discrete_c_logits": None,
                "info_conti_mu": None,
                "info_conti_var": None,
            }
