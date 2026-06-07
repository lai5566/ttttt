# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/models/cyclegan.py
# CycleGAN backbone for unpaired image-to-image translation

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.misc as misc
from models.image_translation_networks import define_generator, define_discriminator


class Generator(nn.Module):
    """CycleGAN Generator adapted to StudioGAN interface

    This is a wrapper that manages TWO generators:
    - Gen_A2B: Translates from domain A to domain B
    - Gen_B2A: Translates from domain B to domain A

    Args:
        z_dim: Not used in CycleGAN (kept for interface compatibility)
        g_shared_dim: Not used (kept for interface compatibility)
        img_size: Output image size
        g_conv_dim: Base number of filters (ngf)
        apply_attn: Not used in CycleGAN
        attn_g_loc: Not used
        g_cond_mtd: Not used (CycleGAN is unconditional)
        num_classes: Not used
        g_init: Initialization method
        g_depth: Number of ResNet blocks (6 or 9)
        mixed_precision: Use AMP
        MODULES: Not used (we use standalone networks)
        MODEL: Full config object
    """
    def __init__(self, z_dim, g_shared_dim, img_size, g_conv_dim, apply_attn,
                 attn_g_loc, g_cond_mtd, num_classes, g_init, g_depth,
                 mixed_precision, MODULES, MODEL):
        super(Generator, self).__init__()

        self.img_size = img_size
        self.mixed_precision = mixed_precision
        self.g_conv_dim = g_conv_dim

        # Get CycleGAN-specific config
        n_blocks = 9 if g_depth == 'N/A' else int(g_depth)
        if img_size <= 128:
            n_blocks = 6  # Use 6 blocks for smaller images

        gen_type = MODEL.cyclegan_gen_type if hasattr(MODEL, 'cyclegan_gen_type') else 'resnet_9blocks'
        if n_blocks == 6:
            gen_type = 'resnet_6blocks'

        norm_type = MODEL.cyclegan_norm if hasattr(MODEL, 'cyclegan_norm') else 'instance'
        use_dropout = MODEL.cyclegan_use_dropout if hasattr(MODEL, 'cyclegan_use_dropout') else False
        init_type = 'normal' if g_init == 'N02' else g_init.lower()

        # Create two generators for bidirectional translation
        self.netG_A2B = define_generator(
            gen_type=gen_type,
            input_nc=3,
            output_nc=3,
            ngf=g_conv_dim,
            norm=norm_type,
            use_dropout=use_dropout,
            init_type=init_type,
            init_gain=0.02
        )

        self.netG_B2A = define_generator(
            gen_type=gen_type,
            input_nc=3,
            output_nc=3,
            ngf=g_conv_dim,
            norm=norm_type,
            use_dropout=use_dropout,
            init_type=init_type,
            init_gain=0.02
        )

    def forward(self, z=None, label=None, eval=False, domain='A', real_images=None):
        """Forward pass - generates images in target domain

        Args:
            z: Not used (CycleGAN uses real images as input)
            label: Not used (unconditional)
            eval: Evaluation mode flag
            domain: 'A' or 'B' - which domain to translate FROM
            real_images: Real images from source domain [batch, 3, H, W]

        Returns:
            Generated images in target domain
        """
        with torch.cuda.amp.autocast() if self.mixed_precision and not eval else misc.dummy_context_mgr():
            if real_images is None:
                raise ValueError("CycleGAN requires real_images as input")

            if domain == 'A':
                # Translate A -> B
                return self.netG_A2B(real_images)
            else:
                # Translate B -> A
                return self.netG_B2A(real_images)

    def translate_A2B(self, real_A):
        """Translate from domain A to domain B"""
        return self.netG_A2B(real_A)

    def translate_B2A(self, real_B):
        """Translate from domain B to domain A"""
        return self.netG_B2A(real_B)


class Discriminator(nn.Module):
    """CycleGAN Discriminator adapted to StudioGAN interface

    This is a wrapper that manages TWO discriminators:
    - Dis_A: Distinguishes real/fake in domain A
    - Dis_B: Distinguishes real/fake in domain B

    Args:
        img_size: Input image size
        d_conv_dim: Base number of filters (ndf)
        apply_d_sn: Not used in CycleGAN (uses instance norm instead)
        apply_attn: Not used
        attn_d_loc: Not used
        d_cond_mtd: Not used (unconditional)
        aux_cls_type: Not used
        d_embed_dim: Not used
        normalize_d_embed: Not used
        num_classes: Not used
        d_init: Initialization method
        d_depth: Number of discriminator layers (3 by default)
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

        # Get CycleGAN-specific config
        n_layers = 3 if d_depth == 'N/A' else int(d_depth)
        disc_type = MODEL.cyclegan_disc_type if hasattr(MODEL, 'cyclegan_disc_type') else 'basic'
        norm_type = MODEL.cyclegan_norm if hasattr(MODEL, 'cyclegan_norm') else 'instance'
        init_type = 'normal' if d_init == 'N02' else d_init.lower()

        # Create two discriminators for both domains
        self.netD_A = define_discriminator(
            disc_type=disc_type,
            input_nc=3,
            ndf=d_conv_dim,
            n_layers_D=n_layers,
            norm=norm_type,
            init_type=init_type,
            init_gain=0.02
        )

        self.netD_B = define_discriminator(
            disc_type=disc_type,
            input_nc=3,
            ndf=d_conv_dim,
            n_layers_D=n_layers,
            norm=norm_type,
            init_type=init_type,
            init_gain=0.02
        )

    def forward(self, x, label=None, eval=False, adc_fake=False, domain='A'):
        """Forward pass - discriminate real/fake for given domain

        Args:
            x: Input images [batch, 3, H, W]
            label: Not used (unconditional)
            eval: Evaluation mode flag
            adc_fake: Not used
            domain: 'A' or 'B' - which domain discriminator to use

        Returns:
            Dictionary with discriminator outputs (StudioGAN format)
        """
        with torch.cuda.amp.autocast() if self.mixed_precision and not eval else misc.dummy_context_mgr():
            if domain == 'A':
                adv_output = self.netD_A(x)
            else:
                adv_output = self.netD_B(x)

            # PatchGAN outputs a map, we need to average for single scalar
            # But we keep the map for loss calculation

            return {
                "h": None,  # No intermediate features
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

    def discriminate_A(self, images):
        """Discriminate images in domain A"""
        return self.netD_A(images)

    def discriminate_B(self, images):
        """Discriminate images in domain B"""
        return self.netD_B(images)
