# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/utils/loss.py

from torch.nn import DataParallel
from torch import autograd
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np

from utils.style_ops import conv2d_gradfix
import utils.ops as ops


class GatherLayer(torch.autograd.Function):
    """
    This file is copied from
    https://github.com/open-mmlab/OpenSelfSup/blob/master/openselfsup/models/utils/gather_layer.py
    Gather tensors from all process, supporting backward propagation
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, cls_output, label, **_):
        return self.ce_loss(cls_output, label).mean()


class ConditionalContrastiveLoss(torch.nn.Module):
    def __init__(self, num_classes, temperature, master_rank, DDP):
        super(ConditionalContrastiveLoss, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.master_rank = master_rank
        self.DDP = DDP
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _make_neg_removal_mask(self, labels):
        labels = labels.detach().cpu().numpy()
        n_samples = labels.shape[0]
        mask_multi, target = np.zeros([self.num_classes, n_samples]), 1.0
        for c in range(self.num_classes):
            c_indices = np.where(labels == c)
            mask_multi[c, c_indices] = target
        return torch.tensor(mask_multi).type(torch.long).to(self.master_rank)

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def _remove_diag(self, M):
        h, w = M.shape
        assert h == w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.bool).to(self.master_rank)
        return M[mask].view(h, -1)

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, embed, proxy, label, **_):
        if self.DDP:
            embed = torch.cat(GatherLayer.apply(embed), dim=0)
            proxy = torch.cat(GatherLayer.apply(proxy), dim=0)
            label = torch.cat(GatherLayer.apply(label), dim=0)

        sim_matrix = self.calculate_similarity_matrix(embed, embed)
        sim_matrix = torch.exp(self._remove_diag(sim_matrix) / self.temperature)
        neg_removal_mask = self._remove_diag(self._make_neg_removal_mask(label)[label])
        sim_pos_only = neg_removal_mask * sim_matrix

        emb2proxy = torch.exp(self.cosine_similarity(embed, proxy) / self.temperature)

        numerator = emb2proxy + sim_pos_only.sum(dim=1)
        denomerator = torch.cat([torch.unsqueeze(emb2proxy, dim=1), sim_matrix], dim=1).sum(dim=1)
        return -torch.log(numerator / denomerator).mean()


class Data2DataCrossEntropyLoss(torch.nn.Module):
    def __init__(self, num_classes, temperature, m_p, master_rank, DDP):
        super(Data2DataCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.m_p = m_p
        self.master_rank = master_rank
        self.DDP = DDP
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def make_index_matrix(self, labels):
        labels = labels.detach().cpu().numpy()
        num_samples = labels.shape[0]
        mask_multi, target = np.ones([self.num_classes, num_samples]), 0.0

        for c in range(self.num_classes):
            c_indices = np.where(labels==c)
            mask_multi[c, c_indices] = target
        return torch.tensor(mask_multi).type(torch.long).to(self.master_rank)

    def remove_diag(self, M):
        h, w = M.shape
        assert h==w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.bool).to(self.master_rank)
        return M[mask].view(h, -1)

    def forward(self, embed, proxy, label, **_):
        # If train a GAN throuh DDP, gather all data on the master rank
        if self.DDP:
            embed = torch.cat(GatherLayer.apply(embed), dim=0)
            proxy = torch.cat(GatherLayer.apply(proxy), dim=0)
            label = torch.cat(GatherLayer.apply(label), dim=0)

        # calculate similarities between sample embeddings
        sim_matrix = self.calculate_similarity_matrix(embed, embed) + self.m_p - 1
        # remove diagonal terms
        sim_matrix = self.remove_diag(sim_matrix/self.temperature)
        # for numerical stability
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = F.relu(sim_matrix) - sim_max.detach()

        # calculate similarities between sample embeddings and the corresponding proxies
        smp2proxy = self.cosine_similarity(embed, proxy)
        # make false negative removal
        removal_fn = self.remove_diag(self.make_index_matrix(label)[label])
        # apply the negative removal to the similarity matrix
        improved_sim_matrix = removal_fn*torch.exp(sim_matrix)

        # compute positive attraction term
        pos_attr = F.relu((self.m_p - smp2proxy)/self.temperature)
        # compute negative repulsion term
        neg_repul = torch.log(torch.exp(-pos_attr) + improved_sim_matrix.sum(dim=1))
        # compute data to data cross-entropy criterion
        criterion = pos_attr + neg_repul
        return criterion.mean()


class PathLengthRegularizer:
    def __init__(self, device, pl_decay=0.01, pl_weight=2, pl_no_weight_grad=False):
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.pl_no_weight_grad = pl_no_weight_grad

    def cal_pl_reg(self, fake_images, ws):
        #ws refers to weight style
        #receives new fake_images of original batch (in original implementation, fakes_images used for calculating g_loss and pl_loss is generated independently)
        pl_noise = torch.randn_like(fake_images) / np.sqrt(fake_images.shape[2] * fake_images.shape[3])
        with conv2d_gradfix.no_weight_gradients(self.pl_no_weight_grad):
            pl_grads = torch.autograd.grad(outputs=[(fake_images * pl_noise).sum()], inputs=[ws], create_graph=True, only_inputs=True)[0]
        pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
        pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
        self.pl_mean.copy_(pl_mean.detach())
        pl_penalty = (pl_lengths - pl_mean).square()
        loss_Gpl = (pl_penalty * self.pl_weight).mean(0)
        return loss_Gpl


def enable_allreduce(dict_):
    loss = 0
    for key, value in dict_.items():
        if value is not None and key != "label":
            loss += value.mean()*0
    return loss


def d_vanilla(d_logit_real, d_logit_fake, DDP):
    d_loss = torch.mean(F.softplus(-d_logit_real)) + torch.mean(F.softplus(d_logit_fake))
    return d_loss


def g_vanilla(d_logit_fake, DDP):
    return torch.mean(F.softplus(-d_logit_fake))


def d_logistic(d_logit_real, d_logit_fake, DDP):
    d_loss = F.softplus(-d_logit_real) + F.softplus(d_logit_fake)
    return d_loss.mean()


def g_logistic(d_logit_fake, DDP):
    # basically same as g_vanilla.
    return F.softplus(-d_logit_fake).mean()


def d_ls(d_logit_real, d_logit_fake, DDP):
    d_loss = 0.5 * (d_logit_real - torch.ones_like(d_logit_real))**2 + 0.5 * (d_logit_fake)**2
    return d_loss.mean()


def g_ls(d_logit_fake, DDP):
    gen_loss = 0.5 * (d_logit_fake - torch.ones_like(d_logit_fake))**2
    return gen_loss.mean()


def d_hinge(d_logit_real, d_logit_fake, DDP):
    return torch.mean(F.relu(1. - d_logit_real)) + torch.mean(F.relu(1. + d_logit_fake))


def g_hinge(d_logit_fake, DDP):
    return -torch.mean(d_logit_fake)


def d_wasserstein(d_logit_real, d_logit_fake, DDP):
    return torch.mean(d_logit_fake - d_logit_real)


def g_wasserstein(d_logit_fake, DDP):
    return -torch.mean(d_logit_fake)


def crammer_singer_loss(adv_output, label, DDP, **_):
    # https://github.com/ilyakava/BigGAN-PyTorch/blob/master/train_fns.py
    # crammer singer criterion
    num_real_classes = adv_output.shape[1] - 1
    mask = torch.ones_like(adv_output).to(adv_output.device)
    mask.scatter_(1, label.unsqueeze(-1), 0)
    wrongs = torch.masked_select(adv_output, mask.bool()).reshape(adv_output.shape[0], num_real_classes)
    max_wrong, _ = wrongs.max(1)
    max_wrong = max_wrong.unsqueeze(-1)
    target = adv_output.gather(1, label.unsqueeze(-1))
    return torch.mean(F.relu(1 + max_wrong - target))


def feature_matching_loss(real_embed, fake_embed):
    # https://github.com/ilyakava/BigGAN-PyTorch/blob/master/train_fns.py
    # feature matching criterion
    fm_loss = torch.mean(torch.abs(torch.mean(fake_embed, 0) - torch.mean(real_embed, 0)))
    return fm_loss


def lecam_reg(d_logit_real, d_logit_fake, ema):
    reg = torch.mean(F.relu(d_logit_real - ema.D_fake).pow(2)) + \
          torch.mean(F.relu(ema.D_real - d_logit_fake).pow(2))
    return reg


def cal_deriv(inputs, outputs, device):
    grads = autograd.grad(outputs=outputs,
                          inputs=inputs,
                          grad_outputs=torch.ones(outputs.size()).to(device),
                          create_graph=True,
                          retain_graph=True,
                          only_inputs=True)[0]
    return grads


def latent_optimise(zs, fake_labels, generator, discriminator, batch_size, lo_rate, lo_steps, lo_alpha, lo_beta, eval,
                    cal_trsp_cost, device):
    for step in range(lo_steps - 1):
        drop_mask = (torch.FloatTensor(batch_size, 1).uniform_() > 1 - lo_rate).to(device)

        zs = autograd.Variable(zs, requires_grad=True)
        fake_images = generator(zs, fake_labels, eval=eval)
        fake_dict = discriminator(fake_images, fake_labels, eval=eval)
        z_grads = cal_deriv(inputs=zs, outputs=fake_dict["adv_output"], device=device)
        z_grads_norm = torch.unsqueeze((z_grads.norm(2, dim=1)**2), dim=1)
        delta_z = lo_alpha * z_grads / (lo_beta + z_grads_norm)
        zs = torch.clamp(zs + drop_mask * delta_z, -1.0, 1.0)

        if cal_trsp_cost:
            if step == 0:
                trsf_cost = (delta_z.norm(2, dim=1)**2).mean()
            else:
                trsf_cost += (delta_z.norm(2, dim=1)**2).mean()
        else:
            trsf_cost = None
        return zs, trsf_cost


def cal_grad_penalty(real_images, real_labels, fake_images, discriminator, device):
    batch_size, c, h, w = real_images.shape
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, real_images.nelement() // batch_size).contiguous().view(batch_size, c, h, w)
    alpha = alpha.to(device)

    real_images = real_images.to(device)
    interpolates = alpha * real_images + ((1 - alpha) * fake_images)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    fake_dict = discriminator(interpolates, real_labels, eval=False)
    grads = cal_deriv(inputs=interpolates, outputs=fake_dict["adv_output"], device=device)
    grads = grads.view(grads.size(0), -1)

    grad_penalty = ((grads.norm(2, dim=1) - 1)**2).mean() + interpolates[:,0,0,0].mean()*0
    return grad_penalty


def cal_dra_penalty(real_images, real_labels, discriminator, device):
    batch_size, c, h, w = real_images.shape
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.to(device)

    real_images = real_images.to(device)
    differences = 0.5 * real_images.std() * torch.rand(real_images.size()).to(device)
    interpolates = real_images + (alpha * differences)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    fake_dict = discriminator(interpolates, real_labels, eval=False)
    grads = cal_deriv(inputs=interpolates, outputs=fake_dict["adv_output"], device=device)
    grads = grads.view(grads.size(0), -1)

    grad_penalty = ((grads.norm(2, dim=1) - 1)**2).mean() + interpolates[:,0,0,0].mean()*0
    return grad_penalty


def cal_maxgrad_penalty(real_images, real_labels, fake_images, discriminator, device):
    batch_size, c, h, w = real_images.shape
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, real_images.nelement() // batch_size).contiguous().view(batch_size, c, h, w)
    alpha = alpha.to(device)

    real_images = real_images.to(device)
    interpolates = alpha * real_images + ((1 - alpha) * fake_images)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    fake_dict = discriminator(interpolates, real_labels, eval=False)
    grads = cal_deriv(inputs=interpolates, outputs=fake_dict["adv_output"], device=device)
    grads = grads.view(grads.size(0), -1)

    maxgrad_penalty = torch.max(grads.norm(2, dim=1)**2) + interpolates[:,0,0,0].mean()*0
    return maxgrad_penalty


def cal_r1_reg(adv_output, images, device):
    batch_size = images.size(0)
    grad_dout = cal_deriv(inputs=images, outputs=adv_output.sum(), device=device)
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == images.size())
    r1_reg = 0.5 * grad_dout2.contiguous().view(batch_size, -1).sum(1).mean(0) + images[:,0,0,0].mean()*0
    return r1_reg


def adjust_k(current_k, topk_gamma, inf_k):
    current_k = max(current_k * topk_gamma, inf_k)
    return current_k


def normal_nll_loss(x, mu, var):
    # https://github.com/Natsu6767/InfoGAN-PyTorch/blob/master/utils.py
    # Calculate the negative log likelihood of normal distribution.
    # Needs to be minimized in InfoGAN. (Treats Q(c]x) as a factored Gaussian)
    logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
    nll = -(logli.sum(1).mean())
    return nll


def stylegan_cal_r1_reg(adv_output, images):
    with conv2d_gradfix.no_weight_gradients():
        r1_grads = torch.autograd.grad(outputs=[adv_output.sum()], inputs=[images], create_graph=True, only_inputs=True)[0]
    r1_penalty = r1_grads.square().sum([1,2,3]) / 2
    return r1_penalty.mean()


# ============================================================================
# Image-to-Image Translation Loss Functions (CycleGAN, Pix2Pix)
# ============================================================================

def cycle_consistency_loss(rec_A, real_A, rec_B, real_B, lambda_A, lambda_B):
    """Cycle consistency loss for CycleGAN

    Ensures that:
    - G_B(G_A(A)) ≈ A (forward cycle consistency)
    - G_A(G_B(B)) ≈ B (backward cycle consistency)

    Args:
        rec_A: Reconstructed A images (G_B(G_A(A)))
        real_A: Real A domain images
        rec_B: Reconstructed B images (G_A(G_B(B)))
        real_B: Real B domain images
        lambda_A: Weight for A cycle loss
        lambda_B: Weight for B cycle loss

    Returns:
        Weighted cycle consistency loss
    """
    loss_cycle_A = torch.nn.functional.l1_loss(rec_A, real_A) * lambda_A
    loss_cycle_B = torch.nn.functional.l1_loss(rec_B, real_B) * lambda_B
    return loss_cycle_A + loss_cycle_B


def identity_loss(idt_A, real_B, idt_B, real_A, lambda_A, lambda_B, lambda_identity):
    """Identity mapping loss for CycleGAN

    Encourages generators to preserve color composition when applied to target domain:
    - G_A(B) ≈ B (generator A should not change B domain images)
    - G_B(A) ≈ A (generator B should not change A domain images)

    This is particularly useful for tasks like photo generation from paintings where
    you want to preserve color composition.

    Args:
        idt_A: Identity mapping G_A(B)
        real_B: Real B domain images
        idt_B: Identity mapping G_B(A)
        real_A: Real A domain images
        lambda_A: Weight for A cycle loss
        lambda_B: Weight for B cycle loss
        lambda_identity: Overall identity loss weight

    Returns:
        Weighted identity loss
    """
    loss_idt_A = torch.nn.functional.l1_loss(idt_A, real_B) * lambda_B * lambda_identity
    loss_idt_B = torch.nn.functional.l1_loss(idt_B, real_A) * lambda_A * lambda_identity
    return loss_idt_A + loss_idt_B


def pix2pix_l1_loss(fake_B, real_B, lambda_L1):
    """L1 reconstruction loss for Pix2Pix

    Encourages pixel-level similarity between generated and target images.
    Combined with adversarial loss for high-quality paired translation.

    Args:
        fake_B: Generated B domain images
        real_B: Real B domain images (ground truth)
        lambda_L1: Weight for L1 loss (typically 100.0)

    Returns:
        Weighted L1 loss
    """
    return torch.nn.functional.l1_loss(fake_B, real_B) * lambda_L1


def dual_branch_consistency_loss(ct_images, mask_images, lambda_consistency, consistency_type="l1"):
    """Dual-branch consistency loss for StyleGAN3 Dual

    Encourages coherent generation between CT images and abnormal region masks.
    This loss ensures that the mask highlights regions that correspond to
    abnormalities in the CT image.

    Consistency strategies:
    1. "l1": L1 distance between CT image and mask-weighted CT
    2. "correlation": Negative correlation between CT intensity and mask activation
    3. "edge": Edge consistency between CT and mask boundaries
    4. "combined": Weighted combination of all strategies

    Args:
        ct_images: (B, C, H, W) - Generated CT images
        mask_images: (B, 1, H, W) - Generated abnormal region masks [0, 1]
        lambda_consistency: Weight for consistency loss (typically 1.0)
        consistency_type: Type of consistency metric \in ["l1", "correlation", "edge", "combined"]

    Returns:
        Weighted consistency loss
    """
    if consistency_type == "l1":
        # Simple L1 distance between normalized CT and mask
        # Encourages mask to align with high-intensity regions in CT
        ct_gray = ct_images.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        ct_normalized = (ct_gray - ct_gray.min()) / (ct_gray.max() - ct_gray.min() + 1e-8)
        loss = torch.nn.functional.l1_loss(ct_normalized, mask_images)

    elif consistency_type == "correlation":
        # Negative correlation: mask should highlight abnormal (high intensity) regions
        ct_gray = ct_images.mean(dim=1, keepdim=True)  # (B, 1, H, W)

        # Flatten spatial dimensions
        ct_flat = ct_gray.view(ct_gray.size(0), -1)
        mask_flat = mask_images.view(mask_images.size(0), -1)

        # Compute correlation coefficient
        ct_mean = ct_flat.mean(dim=1, keepdim=True)
        mask_mean = mask_flat.mean(dim=1, keepdim=True)

        ct_centered = ct_flat - ct_mean
        mask_centered = mask_flat - mask_mean

        correlation = (ct_centered * mask_centered).sum(dim=1) / (
            torch.sqrt((ct_centered ** 2).sum(dim=1)) *
            torch.sqrt((mask_centered ** 2).sum(dim=1)) + 1e-8
        )

        # Maximize correlation (minimize negative correlation)
        loss = (1 - correlation).mean()

    elif consistency_type == "edge":
        # Edge consistency: mask boundaries should align with CT edges
        # Use Sobel operator to detect edges
        def sobel_edges(x):
            # Sobel kernels
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                   dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                   dtype=x.dtype, device=x.device).view(1, 1, 3, 3)

            # Compute gradients
            grad_x = torch.nn.functional.conv2d(x, sobel_x, padding=1)
            grad_y = torch.nn.functional.conv2d(x, sobel_y, padding=1)

            # Edge magnitude
            edges = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
            return edges

        ct_gray = ct_images.mean(dim=1, keepdim=True)
        ct_edges = sobel_edges(ct_gray)
        mask_edges = sobel_edges(mask_images)

        # L1 distance between edges
        loss = torch.nn.functional.l1_loss(ct_edges, mask_edges)

    elif consistency_type == "combined":
        # Weighted combination of all consistency metrics
        # L1 consistency (40%)
        ct_gray = ct_images.mean(dim=1, keepdim=True)
        ct_normalized = (ct_gray - ct_gray.min()) / (ct_gray.max() - ct_gray.min() + 1e-8)
        loss_l1 = torch.nn.functional.l1_loss(ct_normalized, mask_images)

        # Correlation consistency (30%)
        ct_flat = ct_gray.view(ct_gray.size(0), -1)
        mask_flat = mask_images.view(mask_images.size(0), -1)
        ct_mean = ct_flat.mean(dim=1, keepdim=True)
        mask_mean = mask_flat.mean(dim=1, keepdim=True)
        ct_centered = ct_flat - ct_mean
        mask_centered = mask_flat - mask_mean
        correlation = (ct_centered * mask_centered).sum(dim=1) / (
            torch.sqrt((ct_centered ** 2).sum(dim=1)) *
            torch.sqrt((mask_centered ** 2).sum(dim=1)) + 1e-8
        )
        loss_corr = (1 - correlation).mean()

        # Edge consistency (30%)
        def sobel_edges(x):
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                   dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                   dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
            grad_x = torch.nn.functional.conv2d(x, sobel_x, padding=1)
            grad_y = torch.nn.functional.conv2d(x, sobel_y, padding=1)
            edges = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
            return edges

        ct_edges = sobel_edges(ct_gray)
        mask_edges = sobel_edges(mask_images)
        loss_edge = torch.nn.functional.l1_loss(ct_edges, mask_edges)

        # Weighted combination
        loss = 0.4 * loss_l1 + 0.3 * loss_corr + 0.3 * loss_edge

    else:
        raise ValueError(f"Unknown consistency_type: {consistency_type}. "
                        f"Choose from ['l1', 'correlation', 'edge', 'combined']")

    return loss * lambda_consistency


class LSGANLoss(nn.Module):
    """Least Squares GAN loss

    Used in CycleGAN by default. More stable than vanilla BCE loss.

    For discriminator:
        L_D = 0.5 * E[(D(real) - 1)^2] + 0.5 * E[D(fake)^2]

    For generator:
        L_G = 0.5 * E[(D(fake) - 1)^2]

    Reference: "Least Squares Generative Adversarial Networks"
    https://arxiv.org/abs/1611.04076
    """
    def __init__(self):
        super(LSGANLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def __call__(self, prediction, target_is_real):
        """Calculate LSGAN loss

        Args:
            prediction: Discriminator output (can be patch-based)
            target_is_real: If True, use label 1.0; if False, use label 0.0

        Returns:
            MSE loss
        """
        if target_is_real:
            target_tensor = torch.ones_like(prediction)
        else:
            target_tensor = torch.zeros_like(prediction)
        return self.mse_loss(prediction, target_tensor)


def patch_gan_discriminator_loss(real_pred, fake_pred, gan_loss_fn):
    """PatchGAN discriminator loss

    Works with both LSGAN and vanilla GAN objectives.
    Averages loss over all patches.

    Args:
        real_pred: Discriminator prediction on real images
        fake_pred: Discriminator prediction on fake images
        gan_loss_fn: Loss function (e.g., LSGANLoss or BCEWithLogitsLoss)

    Returns:
        Combined discriminator loss
    """
    loss_real = gan_loss_fn(real_pred, target_is_real=True)
    loss_fake = gan_loss_fn(fake_pred, target_is_real=False)
    loss_D = (loss_real + loss_fake) * 0.5
    return loss_D


def patch_gan_generator_loss(fake_pred, gan_loss_fn):
    """PatchGAN generator loss

    Generator tries to fool discriminator by making fake_pred close to 1.

    Args:
        fake_pred: Discriminator prediction on fake images
        gan_loss_fn: Loss function (e.g., LSGANLoss or BCEWithLogitsLoss)

    Returns:
        Generator adversarial loss
    """
    return gan_loss_fn(fake_pred, target_is_real=True)
