import argparse, os, math, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model_pure_ar import PureAutoregressiveTransformer

class ByteDataset(Dataset):
    def __init__(self, path, block_size):
        data = open(path, "rb").read()  # raw bytes
        self.data = np.frombuffer(data, dtype=np.uint8)
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block_size - 1)

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size].astype(np.int64)
        y = self.data[idx+1:idx+self.block_size+1].astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)

def get_dataloader(path, block_size, batch_size, shuffle=True):
    ds = ByteDataset(path, block_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.txt")
    ap.add_argument("--val", default="data/val.txt")
    ap.add_argument("--out_dir", default="checkpoints/pure_ar_mytholm")
    ap.add_argument("--n_embd", type=int, default=512)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--n_layer", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_seq", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume training from")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = get_dataloader(args.train, args.max_seq, args.batch_size, shuffle=True)
    val_loader = get_dataloader(args.val, args.max_seq, args.batch_size, shuffle=False)

    model = PureAutoregressiveTransformer(
        n_embd=args.n_embd, n_head=args.n_head, n_layer=args.n_layer,
        dropout=args.dropout, max_seq=args.max_seq, vocab_size=256
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(train_loader)*args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device=="cuda"))

    # ---- Resume logic ----
    step = 0
    best_val = float("inf")
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt: opt.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt and ckpt["scheduler"] is not None: sched.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt: scaler.load_state_dict(ckpt["scaler"])
        step = ckpt.get("step", 0)
        start_epoch = ckpt.get("epoch", 1)
        best_val = ckpt.get("best_val", float("inf"))
        print(f"Resumed from {args.resume} | epoch={start_epoch}, step={step}, best_val={best_val:.4f}")

    # ---- Training loop ----
    for epoch in range(start_epoch, args.epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x,y in pbar:
            x = x.to(device); y = y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                logits, loss = model(x, y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt); scaler.update()
            sched.step()

            step += 1
            if step % 100 == 0:
                pbar.set_postfix(loss=float(loss))

            if step % args.eval_every == 0:
                model.eval()
                with torch.no_grad():
                    losses = []
                    for vx, vy in val_loader:
                        vx = vx.to(device); vy = vy.to(device)
                        _, vl = model(vx, vy)
                        losses.append(vl.item())
                val_loss = float(np.mean(losses))
                ppl = math.exp(val_loss)
                print(f"\n[eval] step {step} | val_loss={val_loss:.4f} | ppl={ppl:.2f}")
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save({
                        "model": model.state_dict(),
                        "config": {
                            "n_embd": args.n_embd, "n_head": args.n_head, "n_layer": args.n_layer,
                            "dropout": args.dropout, "max_seq": args.max_seq
                        },
                        "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict(),
                        "scaler": scaler.state_dict(),
                        "epoch": epoch,
                        "step": step,
                        "best_val": best_val
                    }, os.path.join(args.out_dir, "best.pt"))
                model.train()

        # save epoch checkpoint
        torch.save({
            "model": model.state_dict(),
            "config": {
                "n_embd": args.n_embd, "n_head": args.n_head, "n_layer": args.n_layer,
                "dropout": args.dropout, "max_seq": args.max_seq
            },
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "step": step,
            "best_val": best_val
        }, os.path.join(args.out_dir, f"epoch{epoch}.pt"))

    # final save
    torch.save({
        "model": model.state_dict(),
        "config": {
            "n_embd": args.n_embd, "n_head": args.n_head, "n_layer": args.n_layer,
            "dropout": args.dropout, "max_seq": args.max_seq
        },
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": args.epochs,
        "step": step,
        "best_val": best_val
    }, os.path.join(args.out_dir, "final.pt"))
    print("Training complete. Checkpoints in", args.out_dir)

if __name__ == "__main__":
    main()
