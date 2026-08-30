# Pontua um gen parquet com k (nº de acertos do solver em n rollouts) e filtra
# a banda difícil-mas-solucionável (k em [1, n-1]) — teste do headroom pós-O16.
import argparse, json, requests, pandas as pd
from concurrent.futures import ThreadPoolExecutor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--endpoint", required=True)        # sglang openai /v1
    ap.add_argument("--model", required=True)
    ap.add_argument("--exec_url", required=True)        # sql exec server
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--max_tokens", type=int, default=1024)
    args = ap.parse_args()

    import sys, os
    sys.path.insert(0, os.getcwd())
    import importlib.util
    spec = importlib.util.spec_from_file_location("t2sql_prompts", os.path.join(os.getcwd(), "verl", "prompts.py"))
    prompts_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(prompts_mod)
    extract_sql = prompts_mod.extract_sql

    df = pd.read_parquet(args.parquet)
    print(f"gen: {len(df)} linhas")

    def rollouts(row):
        prompt = row["prompt"][0]["content"] if isinstance(row["prompt"], (list,)) else str(row["prompt"])
        try:
            r = requests.post(f"{args.endpoint}/completions", json={
                "model": args.model, "prompt": prompt, "n": args.n,
                "temperature": 1.0, "max_tokens": args.max_tokens}, timeout=300)
            return [extract_sql(c["text"]) for c in r.json()["choices"]]
        except Exception:
            return [None] * args.n

    rows = df.to_dict("records")
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        all_sqls = list(pool.map(rollouts, rows))

    # comparação em lote no exec server
    items, owners = [], []
    for i, (row, sqls) in enumerate(zip(rows, all_sqls)):
        gt = row["reward_model"]["ground_truth"]
        gt = json.loads(gt) if isinstance(gt, str) else gt
        for sql in sqls:
            if sql:
                items.append({"db_id": gt["db_id"], "pred_sql": sql, "gold_sql": gt["gold_sql"]})
                owners.append(i)
    ks = [0] * len(rows)
    B = 2000
    for s in range(0, len(items), B):
        res = requests.post(f"{args.exec_url}/compare", json={"items": items[s:s+B]}, timeout=1200).json()
        for owner, r in zip(owners[s:s+B], res["result"]):
            if r["match"]:
                ks[owner] += 1

    df["k_solver"] = ks
    hist = pd.Series(ks).value_counts().sort_index().to_dict()
    print("k-hist:", hist)
    df.to_parquet(args.out.replace(".parquet", "_scored.parquet"))
    # variante principal: REMOVE so as impossiveis (k=0) — testa poluicao sem confundir com tamanho
    keep = df[df.k_solver >= 1].drop(columns=["k_solver"])
    print(f"sem-k0 (principal): {len(keep)} de {len(df)}")
    keep.to_parquet(args.out)
    # variante banda estrita (analise secundaria)
    band = df[(df.k_solver >= 1) & (df.k_solver <= args.n - 1)].drop(columns=["k_solver"])
    band.to_parquet(args.out.replace(".parquet", "_band.parquet"))
    print(f"banda k[1,{args.n-1}]: {len(band)}")
    print("salvo:", args.out)

if __name__ == "__main__":
    main()
