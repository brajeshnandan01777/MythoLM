
import argparse, torch
from model_pure_ar import PureAutoregressiveTransformer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/pure_ar_mytholm/best.pt")
    ap.add_argument("--prompt", default="In the age before mortals, when gods still walked among the stars,")
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["config"]
    model = PureAutoregressiveTransformer(
        n_embd=cfg["n_embd"], n_head=cfg["n_head"], n_layer=cfg["n_layer"],
        dropout=cfg["dropout"], max_seq=cfg["max_seq"], vocab_size=256
    ).to(device)
    model.load_state_dict(ckpt["model"])

    # byte-level tokenizer: bytes <-> text
    def encode(text): return torch.tensor([[b for b in text.encode("utf-8")]], dtype=torch.long, device=device)
    def decode(tokens): return bytes([int(t) for t in tokens]).decode("utf-8", errors="ignore")

    x = encode(args.prompt)
    y = model.generate(x, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                       top_k=(args.top_k if args.top_k>0 else None), top_p=args.top_p)
    print("\n" + "="*40 + "\nMythoLM (Pure AR) Generation\n" + "="*40)
    print(decode(y[0].tolist()))

if __name__ == "__main__":
    main()
