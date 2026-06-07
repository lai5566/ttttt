"""
MLA-GAN 多卡（DDP）輔助工具。

設計原則
────────
1. **單卡完全相容**：未用 torchrun 啟動（環境無 WORLD_SIZE 或 =1）時，所有函式
   退化為單卡 no-op，訓練行為與原本逐位元一致。
2. **多卡啟用**：用 `torchrun --nproc_per_node=N ...` 啟動時，自動讀取 env
   (RANK/LOCAL_RANK/WORLD_SIZE) 建立 NCCL process group。
3. **GAN + R1 的相容性**：MLA-GAN 的 D-loss 含 R1 正則（create_graph=True，
   屬 double backward），與 PyTorch DDP 的 reducer 有已知不相容。因此本訓練採
   **混合策略**：
     - Generator：包標準 DDP（G-step 純一階，安全且可重疊通訊）。
     - Discriminator：**不包 DDP**，改用 `average_gradients()` 在每次
       backward 後手動 all-reduce 梯度，完全閃開 double-backward 問題。

啟動範例
────────
    # 單卡（與原本一致）
    python train_ir5_v3.py --epochs 400 --batch-size 8 --seed 7

    # 4 卡（全域 batch 仍為 8，平均切到各卡 → 每卡 2 張）
    torchrun --standalone --nproc_per_node=4 \\
        train_ir5_v3.py --epochs 400 --batch-size 8 --seed 7
"""

import os

import torch
import torch.distributed as dist


class DDPContext:
    """封裝分散式狀態，單卡時為安全預設值。"""

    def __init__(self):
        self.distributed = False
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_ddp() -> DDPContext:
    """從 torchrun 注入的環境變數初始化 process group；單卡則回傳 no-op context。"""
    ctx = DDPContext()
    world_size = int(os.environ.get('WORLD_SIZE', '1'))

    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError('WORLD_SIZE>1 但偵測不到 CUDA，DDP 需要 GPU')
        ctx.distributed = True
        ctx.rank = int(os.environ['RANK'])
        ctx.local_rank = int(os.environ.get('LOCAL_RANK', os.environ['RANK']))
        ctx.world_size = world_size
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(ctx.local_rank)
        ctx.device = f'cuda:{ctx.local_rank}'
        if ctx.is_main:
            print(f"[DDP] 分散式訓練啟用：world_size={world_size}, backend=nccl")
        dist.barrier()
    elif torch.cuda.is_available():
        ctx.device = 'cuda'

    return ctx


def cleanup_ddp(ctx: DDPContext):
    """訓練結束時銷毀 process group。"""
    if ctx.distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def unwrap_model(m):
    """剝除 torch.compile (`_orig_mod`) 與 DDP (`module`) 外層，取回原始 nn.Module。

    支援的包裝順序：
      torch.compile(DDP(core)) → m._orig_mod=DDP, .module=core
      DDP(core)                → m.module=core
      torch.compile(core)      → m._orig_mod=core
      core                     → core
    """
    if hasattr(m, '_orig_mod'):
        m = m._orig_mod
    if hasattr(m, 'module'):
        m = m.module
    return m


def broadcast_module(module, src: int = 0):
    """把 src rank 的參數與 buffer 廣播給所有 rank，確保初始權重一致。

    用於未包 DDP 的模型（本專案的 Discriminator）。包 DDP 的模型由 DDP
    建構時自動廣播，不需呼叫此函式。
    """
    if not dist.is_initialized():
        return
    for p in module.parameters():
        dist.broadcast(p.data, src=src)
    for b in module.buffers():
        dist.broadcast(b.data, src=src)


def average_gradients(module, world_size: int):
    """跨 rank 手動平均 module 各參數的梯度（= DDP reducer 的等效行為）。

    在 `loss.backward()` 之後、`optimizer.step()` 之前呼叫，供未包 DDP 的 G/D
    使用，以閃開 R1 (create_graph=True) 與 G 多次 forward 對 DDP 的限制。

    重要：對 `requires_grad` 的參數**一律** all-reduce（grad 為 None 時補零）。
    這樣每個 rank 走訪、通訊的參數集合與順序完全一致，避免「某 rank 的某參數
    這步剛好沒梯度（None）」造成跨 rank all-reduce 數量不對而 deadlock。
    參數走訪順序由模型註冊順序決定，各 rank 相同。
    """
    if world_size <= 1:
        return
    for p in module.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
        p.grad.data /= world_size


def all_reduce_mean(value: float, world_size: int, device) -> float:
    """跨 rank 平均一個純量（給 ADA 的 r_t 用，讓各 rank 的增強機率 p 一致）。"""
    if world_size <= 1:
        return value
    t = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / world_size).item()


def make_epoch_perm(N: int, epoch: int, device):
    """產生跨 rank 一致的 per-epoch 洗牌。

    用獨立的 CPU generator（種子 = 基礎種子 + epoch），確保所有 rank 得到
    **相同**的排列，這樣才能把同一個全域 batch 一致地切分到各 rank。
    """
    base = torch.initial_seed() % (2 ** 31)
    g = torch.Generator()
    g.manual_seed(int(base) + int(epoch))
    return torch.randperm(N, generator=g).to(device)
