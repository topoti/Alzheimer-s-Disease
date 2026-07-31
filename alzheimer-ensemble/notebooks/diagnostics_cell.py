# =============================================================================
# SECTION 9 - ONE-SHOT DIAGNOSTIC BATTERY
# Paste as ONE new cell at the end of train_kaggle.ipynb and run it.
# Requires cells 0-16 to have already run (CFG, central, F, build_model, extract,
# feature_extractor, OasisDS, set_trainable, param_groups, freeze_frozen_bn,
# class_weights, series_id, set_seed, device, AMP, CLASS_NAMES all in scope).
# Everything is printed AND written to /kaggle/working/diagnostics.log
# =============================================================================
import sys, time, copy, traceback, json
import numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (f1_score, balanced_accuracy_score, roc_auc_score,
                             confusion_matrix, classification_report)

RUN = dict(audit=True, geometry=True, leakage=True, permutation=True,
           probe_variants=True, aggregation=True, subtasks=True,
           slice_window=True, train_ceiling=True, backbones=True, finetune=True)

FT_FOLD, FT_S1, FT_S2, FT_PATIENCE = 0, 2, 8, 4
ALT_BACKBONES, ALT_VIEWS = ["resnet50", "efficientnet_b0"], 2
PROBE_EPOCHS, PROBE_PATIENCE, FIXED_EPOCH = 30, 8, 10
ALL_FOLDS = list(range(CFG["n_folds"]))
NV = CFG["probe_views"]

LOGPATH = Path("/kaggle/working/diagnostics.log")
_real_stdout = sys.stdout
class _Tee:
    def __init__(self, p, o): self.f = open(p, "w", encoding="utf-8"); self.o = o
    def write(self, s): self.f.write(s); self.o.write(s)
    def flush(self): self.f.flush(); self.o.flush()

T0 = time.time()
ROWS = []

def banner(t):
    print(f"\n{'='*78}\n### {t}    [t+{time.time()-T0:.0f}s]\n{'='*78}")

def block(name, fn):
    if not RUN.get(name, False):
        print(f"\n--- SKIPPED block: {name}"); return
    t = time.time()
    try:
        fn()
    except Exception:
        print(f"\n!!! BLOCK {name} FAILED - continuing\n{traceback.format_exc()}")
    print(f"--- block {name} took {time.time()-t:.0f}s")

def logrow(**kw): ROWS.append(kw)

def tsel(X, m):
    """Boolean-select rows of a torch tensor or numpy array with a bool mask."""
    m = np.asarray(m)
    return X[torch.as_tensor(m)] if torch.is_tensor(X) else np.asarray(X)[m]

# ---------------------------------------------------------------- scoring ---
def ovr_auc(P, t, nc):
    out = []
    for c in range(nc):
        yb = (t == c).astype(int)
        out.append(roc_auc_score(yb, P[:, c]) if 0 < yb.sum() < len(yb) else np.nan)
    return np.array(out)

def score(P, t, nc=3):
    L = list(range(nc)); pred = P.argmax(1)
    return dict(macro_f1=f1_score(t, pred, average="macro", labels=L, zero_division=0),
                bal_acc=balanced_accuracy_score(t, pred),
                acc=float((pred == t).mean()),
                f1c=f1_score(t, pred, average=None, labels=L, zero_division=0),
                auc=ovr_auc(P, t, nc),
                cm=confusion_matrix(t, pred, labels=L), pred=pred, true=t)

def subj_pack(p, s, t, nc=3, how="mean"):
    cols = [f"c{i}" for i in range(nc)]
    d = pd.DataFrame(p, columns=cols); d["s"] = np.asarray(s); d["y"] = t
    g = d.groupby("s", sort=True)
    if how == "mean":
        P = g[cols].mean().values
    elif how == "median":
        P = g[cols].median().values
    elif how == "geom":
        dl = pd.DataFrame(np.log(np.clip(p, 1e-9, 1.0)), columns=cols)
        dl["s"] = np.asarray(s)
        P = np.exp(dl.groupby("s", sort=True)[cols].mean().values)
        P = P / P.sum(1, keepdims=True)
    elif how == "vote":
        d2 = d.copy(); d2["a"] = p.argmax(1); g2 = d2.groupby("s", sort=True)["a"]
        P = np.stack([g2.apply(lambda v, c=c: float((v == c).mean())).values
                      for c in range(nc)], 1)
    else:
        raise ValueError(how)
    return P, g["y"].first().values, np.array(sorted(set(np.asarray(s))), dtype=object)

def fmt(m):
    return (f"F1={m['macro_f1']:.4f} bal={m['bal_acc']:.4f} acc={m['acc']:.4f} "
            f"f1c=[{','.join(f'{v:.2f}' for v in m['f1c'])}] "
            f"auc=[{','.join('nan' if np.isnan(v) else f'{v:.3f}' for v in m['auc'])}]")

# ------------------------------------------------------------- probe core ---
def fit_probe(tr_X, tr_y, va_X, va_y, va_s, *, nc=3, head="linear", sampler=True,
              class_weight=False, epochs=PROBE_EPOCHS, patience=PROBE_PATIENCE,
              lr=CFG["probe_lr"], seed=0, extra=None, verbose=False):
    """extra=(X,y,s) is scored at the best epoch (used for the train-set ceiling)."""
    torch.manual_seed(seed)
    mu, sd = tr_X.mean(0, keepdim=True), tr_X.std(0, keepdim=True).clamp_min(1e-6)
    trX = ((tr_X - mu) / sd).to(device); vaX = ((va_X - mu) / sd).to(device)
    trY = tr_y.to(device)
    counts = torch.bincount(tr_y, minlength=nc).float()
    cw = counts.sum() / (nc * counts.clamp_min(1.0))
    ds = TensorDataset(trX, trY)
    dl = (DataLoader(ds, 256, sampler=WeightedRandomSampler(cw[tr_y].double(),
                                                            len(tr_y), True))
          if sampler else DataLoader(ds, 256, shuffle=True))
    D = trX.shape[1]
    net = (nn.Sequential(nn.Dropout(CFG["dropout"]), nn.Linear(D, nc)) if head == "linear"
           else nn.Sequential(nn.Dropout(CFG["dropout"]), nn.Linear(D, 256), nn.ReLU(),
                              nn.Dropout(CFG["dropout"]), nn.Linear(256, nc))).to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"],
                               weight=cw.to(device) if class_weight else None)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=CFG["weight_decay"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    hist, best, best_e, wait, best_state = [], -1.0, 1, 0, None
    vy = np.asarray(va_y.numpy() if torch.is_tensor(va_y) else va_y)
    for e in range(epochs):
        net.train(); tot = 0.0
        for xb, yb in dl:
            opt.zero_grad(); l = crit(net(xb), yb); l.backward(); opt.step()
            tot += l.item() * len(xb)
        sch.step(); net.eval()
        with torch.no_grad():
            logits = net(vaX).float().cpu()
        p = torch.softmax(logits, 1).numpy()
        P, ts, _ = subj_pack(p, va_s, vy, nc)
        mi, ms = score(p, vy, nc), score(P, ts, nc)
        hist.append(dict(epoch=e + 1, train_loss=tot / len(ds), img=mi, subj=ms,
                         logits=logits))
        if verbose:
            print(f"      [{e+1:2d}] tr={tot/len(ds):.4f} imgF1={mi['macro_f1']:.4f} "
                  f"subjF1={ms['macro_f1']:.4f}")
        if ms["macro_f1"] > best:
            best, best_e, wait = ms["macro_f1"], e + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: break

    out = dict(n_epochs=len(hist), best_epoch=best_e, best=hist[best_e - 1],
               final=hist[-1], fixed=hist[min(FIXED_EPOCH, len(hist)) - 1],
               state=best_state, mu=mu, sd=sd, nc=nc, va_s=np.asarray(va_s), va_y=vy)
    if extra is not None:
        net.load_state_dict(best_state); net.eval()
        eX, ey, es = extra
        eyn = ey.numpy() if torch.is_tensor(ey) else np.asarray(ey)
        with torch.no_grad():
            ep = torch.softmax(net(((eX - mu) / sd).to(device)).float().cpu(), 1).numpy()
        EP, et, _ = subj_pack(ep, es, eyn, nc)
        out["extra_img"], out["extra_subj"] = score(ep, eyn, nc), score(EP, et, nc)
    del trX, vaX; torch.cuda.empty_cache()
    return out

def fold_tensors(Fd, fold, use_aug=True):
    tr_d = (Fd["fold_det"] != fold).numpy(); va_d = (Fd["fold_det"] == fold).numpy()
    has_aug = use_aug and len(Fd["X_aug"]) > 0
    if has_aug:
        tr_a = (Fd["fold_aug"] != fold).numpy()
        trX = torch.cat([tsel(Fd["X_det"], tr_d), tsel(Fd["X_aug"], tr_a)])
        trY = torch.cat([tsel(Fd["y_det"], tr_d), tsel(Fd["y_aug"], tr_a)])
    else:
        trX, trY = tsel(Fd["X_det"], tr_d), tsel(Fd["y_det"], tr_d)
    return (trX, trY, tsel(Fd["X_det"], va_d), tsel(Fd["y_det"], va_d),
            tsel(Fd["s_det"], va_d), tr_d, va_d)

def sweep(tag, Fd, folds=ALL_FOLDS, use_aug=True, **kw):
    accB, accF, accX, aucs = [], [], [], []
    for f in folds:
        trX, trY, vaX, vaY, vaS, _, _ = fold_tensors(Fd, f, use_aug)
        r = fit_probe(trX, trY, vaX, vaY, vaS, **kw)
        b, fi, fx = r["best"]["subj"], r["final"]["subj"], r["fixed"]["subj"]
        print(f"  [{tag}] fold{f} ep={r['n_epochs']:2d} best@{r['best_epoch']:2d} | "
              f"BEST {fmt(b)} | FINAL F1={fi['macro_f1']:.4f} | "
              f"FIX{FIXED_EPOCH} F1={fx['macro_f1']:.4f} | "
              f"img F1={r['best']['img']['macro_f1']:.4f} "
              f"auc=[{','.join('nan' if np.isnan(v) else f'{v:.3f}' for v in r['best']['img']['auc'])}]")
        accB.append(b["macro_f1"]); accF.append(fi["macro_f1"]); accX.append(fx["macro_f1"])
        aucs.append(b["auc"])
    print(f"  [{tag}] MEAN subjF1 best={np.mean(accB):.4f}+/-{np.std(accB):.4f}  "
          f"final={np.mean(accF):.4f}  fix{FIXED_EPOCH}={np.mean(accX):.4f}  "
          f"subjAUC={np.round(np.nanmean(np.stack(aucs), 0), 3).tolist()}")
    logrow(test=tag, best=round(float(np.mean(accB)), 4), sd=round(float(np.std(accB)), 4),
           final=round(float(np.mean(accF)), 4), fixed=round(float(np.mean(accX)), 4),
           auc=np.round(np.nanmean(np.stack(aucs), 0), 3).tolist())
    return accB

# =============================================================== BLOCK 0 ====
def b_audit():
    banner("0. ENVIRONMENT + SPLIT AUDIT")
    import sklearn, torchvision
    print("torch", torch.__version__, "| sklearn", sklearn.__version__,
          "| torchvision", torchvision.__version__, "| cuda", torch.cuda.is_available(),
          "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    print("\nCFG:", json.dumps({k: str(v) for k, v in CFG.items()}, indent=1))

    dev = central[central.fold != -1].reset_index(drop=True)
    tst = central[central.fold == -1]
    print(f"\ncentral rows={len(central):,} subjects={central.subject_id.nunique()}")
    print(f"dev rows={len(dev):,} subj={dev.subject_id.nunique()} | "
          f"test rows={len(tst):,} subj={tst.subject_id.nunique()}")
    print("dev/test subject overlap:", len(set(dev.subject_id) & set(tst.subject_id)),
          "(MUST be 0)")
    print("subjects in >1 fold  :", int((central.groupby("subject_id")["fold"]
                                         .nunique() > 1).sum()), "(MUST be 0)")
    print("subjects with >1 label:", int((central.groupby("subject_id")["label"]
                                          .nunique() > 1).sum()), "(MUST be 0)")
    print("duplicate filepaths  :", int(central.filepath.duplicated().sum()), "(MUST be 0)")

    sub = central.groupby("subject_id").agg(label=("label", "first"), fold=("fold", "first"))
    ren = {i: n for i, n in enumerate(CLASS_NAMES)}
    print("\nSUBJECT counts per fold x class  <-- the real sample size:")
    print(pd.crosstab(sub.fold, sub.label).rename(columns=ren).to_string())
    print("\nIMAGE counts per fold x class:")
    print(pd.crosstab(central.fold, central.label).rename(columns=ren).to_string())

    ips = central.groupby("subject_id").size()
    print(f"\nimages/subject: min={ips.min()} med={ips.median():.0f} "
          f"mean={ips.mean():.1f} max={ips.max()}")
    ser = pd.Series(series_id(central.filepath.tolist()), index=central.index)
    nser = central.assign(_s=ser).groupby("subject_id")["_s"].nunique()
    print(f"scan-series/subject: min={nser.min()} med={nser.median():.0f} "
          f"mean={nser.mean():.2f} max={nser.max()} | total series={ser.nunique():,}")
    print("\nslice_idx counts:")
    print(central.slice_idx.value_counts().sort_index().to_string())

    print("\nFEATURE-CACHE ALIGNMENT (all MUST be True):")
    print("  len match     :", len(F["X_det"]) == len(dev))
    print("  labels match  :", bool((F["y_det"].numpy() == dev.label.values).all()))
    print("  subjects match:", bool((np.asarray(F["s_det"]) == dev.subject_id.values).all()))
    print("  folds match   :", bool((F["fold_det"].numpy() == dev.fold.values).all()))
    print("  X_det", tuple(F["X_det"].shape), "X_aug", tuple(F["X_aug"].shape),
          "| NaN:", bool(torch.isnan(F["X_det"]).any()),
          f"| norm mean={F['X_det'].norm(dim=1).mean():.3f} "
          f"std={F['X_det'].norm(dim=1).std():.3f}")

# =============================================================== BLOCK 1 ====
def b_geometry():
    banner("1. FEATURE GEOMETRY - how redundant are slices of the same subject?")
    dev = central[central.fold != -1].reset_index(drop=True)
    X = torch.nn.functional.normalize(F["X_det"].float(), dim=1).to(device)
    subj = dev.subject_id.values
    ser = np.array(series_id(dev.filepath.tolist()))
    lab = dev.label.values
    n = len(X); B = 512
    same_s, same_r, same_l, si, so = [], [], [], [], []
    for i in range(0, n, B):
        S = (X[i:i + B] @ X.T).float()
        rows = torch.arange(S.shape[0], device=device)
        j = np.arange(i, min(i + B, n))
        S[rows, torch.as_tensor(j, device=device)] = -2.0
        k = S.argmax(1).cpu().numpy()
        same_s.append(subj[k] == subj[j]); same_r.append(ser[k] == ser[j])
        same_l.append(lab[k] == lab[j])
        msame = torch.as_tensor(subj[None, :] == subj[j][:, None], device=device)
        S[rows, torch.as_tensor(j, device=device)] = float("nan")
        NANS = torch.full_like(S, float("nan"))
        si.append(torch.nanmean(torch.where(msame, S, NANS), 1).cpu())
        so.append(torch.nanmean(torch.where(~msame, S, NANS), 1).cpu())
    same_s, same_r, same_l = map(np.concatenate, (same_s, same_r, same_l))
    si, so = torch.cat(si).numpy(), torch.cat(so).numpy()
    print(f"1-NN over all {n:,} dev images (self excluded):")
    print(f"  nearest neighbour is same SUBJECT     : {same_s.mean():.4f}")
    print(f"  nearest neighbour is same SCAN SERIES : {same_r.mean():.4f}")
    print(f"  nearest neighbour is same LABEL       : {same_l.mean():.4f}")
    print(f"mean cosine: within-subject={np.nanmean(si):.4f} "
          f"between-subject={np.nanmean(so):.4f} gap={np.nanmean(si)-np.nanmean(so):.4f}")
    logrow(test="1nn_same_subject", best=round(float(same_s.mean()), 4))
    logrow(test="1nn_same_label", best=round(float(same_l.mean()), 4))
    del X, S; torch.cuda.empty_cache()

# =============================================================== BLOCK 2 ====
def b_leakage():
    banner("2. LEAKAGE TEST - subject-grouped split vs random image-level split")
    print("Same features, same head, same hyperparameters. ONLY the split changes.")
    n = len(F["X_det"]); rng = np.random.RandomState(0)
    for fold in ALL_FOLDS:
        n_val = int((F["fold_det"] == fold).sum())
        trX, trY, vaX, vaY, vaS, _, _ = fold_tensors(F, fold)
        ra = fit_probe(trX, trY, vaX, vaY, vaS)
        vm = np.zeros(n, bool); vm[rng.permutation(n)[:n_val]] = True; tm = ~vm
        if NV:
            am = np.tile(tm, NV)
            trX2 = torch.cat([tsel(F["X_det"], tm), tsel(F["X_aug"], am)])
            trY2 = torch.cat([tsel(F["y_det"], tm), tsel(F["y_aug"], am)])
        else:
            trX2, trY2 = tsel(F["X_det"], tm), tsel(F["y_det"], tm)
        rb = fit_probe(trX2, trY2, tsel(F["X_det"], vm), tsel(F["y_det"], vm),
                       tsel(F["s_det"], vm))
        print(f"  fold{fold} n_val_img={n_val:,}")
        print(f"    SUBJECT-GROUPED image-level : {fmt(ra['best']['img'])}")
        print(f"    SUBJECT-GROUPED subject-lvl : {fmt(ra['best']['subj'])}")
        print(f"    RANDOM-IMAGE    image-level : {fmt(rb['best']['img'])}   "
              f"<-- what leaky notebooks report")
        logrow(test=f"leak_grouped_img_f{fold}",
               best=round(ra['best']['img']['macro_f1'], 4))
        logrow(test=f"leak_random_img_f{fold}",
               best=round(rb['best']['img']['macro_f1'], 4))

# =============================================================== BLOCK 3 ====
def b_permutation():
    banner("3. LABEL-PERMUTATION TEST - must collapse to chance (macro-F1 ~0.29)")
    dev = central[central.fold != -1].reset_index(drop=True)
    sub = dev.groupby("subject_id")["label"].first()
    shuf = pd.Series(np.random.RandomState(7).permutation(sub.values), index=sub.index)
    ynew = torch.as_tensor(dev.subject_id.map(shuf).values, dtype=torch.long)
    Fp = dict(F); Fp["y_det"] = ynew
    Fp["y_aug"] = ynew.repeat(NV) if NV else F["y_aug"]
    sweep("permuted_labels", Fp)
    print("  If this is well above ~0.29, the split/eval pipeline itself leaks.")

# =============================================================== BLOCK 4 ====
def b_probe_variants():
    banner("4. PROBE VARIANTS on frozen resnet18 features (all 5 folds each)")
    print("BEST=max over epochs (optimistic, current reporting). FINAL=last epoch. "
          f"FIX{FIXED_EPOCH}=epoch {FIXED_EPOCH}. auc=one-vs-rest, threshold-free.\n")
    sweep("v1_linear_sampler",   F, head="linear", sampler=True,  class_weight=False)
    sweep("v2_linear_clsweight", F, head="linear", sampler=False, class_weight=True)
    sweep("v3_linear_plain",     F, head="linear", sampler=False, class_weight=False)
    sweep("v4_mlp_sampler",      F, head="mlp",    sampler=True,  class_weight=False)
    sweep("v5_linear_noaugviews", F, use_aug=False, head="linear", sampler=True)

    print("\n  v6_subject_mean_features (mean of all slice features -> 1 vector/subject):")
    dev = central[central.fold != -1].reset_index(drop=True)
    dfX = pd.DataFrame(F["X_det"].numpy()); dfX["s"] = dev.subject_id.values
    SX = torch.tensor(dfX.groupby("s", sort=True).mean().values, dtype=torch.float32)
    meta = dev.groupby("subject_id", sort=True).agg(label=("label", "first"),
                                                    fold=("fold", "first"))
    SY = torch.tensor(meta.label.values, dtype=torch.long)
    SF, SS = meta.fold.values, meta.index.values
    accs = []
    for f in ALL_FOLDS:
        tr, va = SF != f, SF == f
        r = fit_probe(tsel(SX, tr), tsel(SY, tr), tsel(SX, va), tsel(SY, va),
                      SS[va], epochs=200, patience=40)
        print(f"    fold{f} n_tr={tr.sum()} n_va={va.sum()} BEST {fmt(r['best']['img'])}")
        accs.append(r["best"]["img"]["macro_f1"])
    print(f"    MEAN F1={np.mean(accs):.4f}+/-{np.std(accs):.4f}")
    logrow(test="v6_subject_mean_features", best=round(float(np.mean(accs)), 4))

# =============================================================== BLOCK 5 ====
def b_aggregation():
    banner("5. AGGREGATION + PRIOR CORRECTION + PER-SUBJECT DUMP (v1 probe)")
    dump = []
    for f in ALL_FOLDS:
        trX, trY, vaX, vaY, vaS, _, _ = fold_tensors(F, f)
        r = fit_probe(trX, trY, vaX, vaY, vaS)
        logits = r["best"]["logits"]; p = torch.softmax(logits, 1).numpy()
        vy = vaY.numpy()
        print(f"\n  fold{f} (best epoch {r['best_epoch']} of {r['n_epochs']})")
        for how in ("mean", "median", "geom", "vote"):
            P, t, _ = subj_pack(p, vaS, vy, 3, how)
            print(f"    agg={how:7s} {fmt(score(P, t))}")
        P, t, sids = subj_pack(p, vaS, vy, 3, "mean")
        m = score(P, t)
        print(f"    subject confusion (rows=true, cols=pred):\n"
              f"{np.array2string(m['cm'])}")
        print(f"    subject-level OvR AUC = {np.round(m['auc'], 3).tolist()}   "
              f"<-- KEY: high AUC + low F1 means threshold/prior problem, not signal")
        tp = np.bincount(trY.numpy(), minlength=3) / len(trY)
        vp = np.bincount(vy, minlength=3) / len(vy)
        print(f"    train prior={np.round(tp,3).tolist()} val prior={np.round(vp,3).tolist()}")
        L = logits.numpy()
        for nm, pr in (("TRAIN-prior shift (honest)     ", tp),
                       ("VAL-prior shift (ORACLE bound) ", vp)):
            line = []
            for tau in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
                La = L + tau * np.log(np.clip(pr, 1e-9, 1))
                pa = np.exp(La - La.max(1, keepdims=True)); pa /= pa.sum(1, keepdims=True)
                Pa, ta, _ = subj_pack(pa, vaS, vy, 3, "mean")
                line.append(f"t={tau}:{score(Pa, ta)['macro_f1']:.3f}")
            print(f"    {nm} " + " ".join(line))
        for sid, tt, pp in zip(sids, t, P):
            dump.append(dict(fold=f, subject=sid, true=CLASS_NAMES[tt],
                             pred=CLASS_NAMES[int(pp.argmax())],
                             p_nd=round(float(pp[0]), 3), p_vm=round(float(pp[1]), 3),
                             p_mm=round(float(pp[2]), 3)))
    D = pd.DataFrame(dump)
    D.to_csv("/kaggle/working/probe_subject_predictions.csv", index=False)
    print(f"\n  per-subject dump -> /kaggle/working/probe_subject_predictions.csv "
          f"({len(D)} rows)")
    print("\n  MISCLASSIFIED SUBJECTS (all folds):")
    print(D[D.true != D.pred].to_string(index=False))
    print("\n  overall subject confusion across folds:")
    print(pd.crosstab(D.true, D.pred).to_string())

# =============================================================== BLOCK 6 ====
def b_subtasks():
    banner("6. SUB-TASK DECOMPOSITION - where does the 3-class confusion live?")
    print("  (a) BINARY no_dementia vs any_dementia   (chance macro-F1 ~0.43)")
    Fb = dict(F); Fb["y_det"] = (F["y_det"] > 0).long()
    Fb["y_aug"] = (F["y_aug"] > 0).long() if len(F["X_aug"]) else F["y_aug"]
    a = []
    for f in ALL_FOLDS:
        trX, trY, vaX, vaY, vaS, _, _ = fold_tensors(Fb, f)
        r = fit_probe(trX, trY, vaX, vaY, vaS, nc=2)
        print(f"    fold{f} BEST subj {fmt(r['best']['subj'])} | "
              f"img {fmt(r['best']['img'])}")
        a.append(r["best"]["subj"]["macro_f1"])
    print(f"    MEAN subjF1={np.mean(a):.4f}+/-{np.std(a):.4f}")
    logrow(test="binary_nd_vs_rest", best=round(float(np.mean(a)), 4))

    print("\n  (b) BINARY very_mild vs mild_moderate, dementia subjects only "
          "(chance macro-F1 ~0.40)")
    a = []
    fd = F["fold_det"].numpy(); fa = F["fold_aug"].numpy()
    dm = (F["y_det"] > 0).numpy(); am = (F["y_aug"] > 0).numpy()
    for f in ALL_FOLDS:
        tr, va, ta = dm & (fd != f), dm & (fd == f), am & (fa != f)
        trX = torch.cat([tsel(F["X_det"], tr), tsel(F["X_aug"], ta)])
        trY = torch.cat([tsel(F["y_det"], tr) - 1, tsel(F["y_aug"], ta) - 1])
        r = fit_probe(trX, trY, tsel(F["X_det"], va), tsel(F["y_det"], va) - 1,
                      tsel(F["s_det"], va), nc=2)
        print(f"    fold{f} n_va_img={va.sum()} BEST subj {fmt(r['best']['subj'])}")
        a.append(r["best"]["subj"]["macro_f1"])
    print(f"    MEAN subjF1={np.mean(a):.4f}+/-{np.std(a):.4f}")
    logrow(test="binary_vm_vs_mm", best=round(float(np.mean(a)), 4))

# =============================================================== BLOCK 7 ====
def b_slice_window():
    banner("7. SLICE-WINDOW ABLATION (re-uses the same cached features)")
    sl = central[central.fold != -1].reset_index(drop=True).slice_idx.values
    for lo, hi, name in ((CFG["central_lo"], CFG["central_hi"], "full cached window"),
                         (128, 132, "128-132 (old config)"), (130, 130, "single slice 130"),
                         (124, 127, "lower edge"), (133, 136, "upper edge")):
        keep = (sl >= lo) & (sl <= hi)
        if keep.sum() == 0:
            print(f"  [{lo}-{hi}] no rows, skipped"); continue
        ka = np.tile(keep, NV) if NV else None
        Fs = dict(X_det=tsel(F["X_det"], keep), y_det=tsel(F["y_det"], keep),
                  s_det=tsel(F["s_det"], keep), fold_det=tsel(F["fold_det"], keep),
                  X_aug=tsel(F["X_aug"], ka) if NV else F["X_aug"],
                  y_aug=tsel(F["y_aug"], ka) if NV else F["y_aug"],
                  s_aug=tsel(F["s_aug"], ka) if NV else F["s_aug"],
                  fold_aug=tsel(F["fold_aug"], ka) if NV else F["fold_aug"])
        print(f"  --- {name} [{lo}-{hi}]  ({keep.sum():,} det images)")
        sweep(f"slice_{lo}_{hi}", Fs)

# =============================================================== BLOCK 8 ====
def b_train_ceiling():
    banner("8. TRAIN-SET CEILING - can the probe fit its own training subjects?")
    print("  low TRAIN F1  => features lack the signal (underfit, data/feature problem)")
    print("  high TRAIN F1 + low VAL F1 => generalisation gap (model/sample-size problem)")
    for f in ALL_FOLDS:
        trX, trY, vaX, vaY, vaS, tr_d, _ = fold_tensors(F, f)
        r = fit_probe(trX, trY, vaX, vaY, vaS,
                      extra=(tsel(F["X_det"], tr_d), tsel(F["y_det"], tr_d),
                             tsel(F["s_det"], tr_d)))
        print(f"  fold{f}   VAL subj {fmt(r['best']['subj'])}")
        print(f"         TRAIN subj {fmt(r['extra_subj'])}")
        print(f"         TRAIN img  {fmt(r['extra_img'])}")

# =============================================================== BLOCK 9 ====
def b_backbones():
    banner("9. ALTERNATIVE FROZEN BACKBONES")
    dev = central[central.fold != -1].reset_index(drop=True)
    fold_det = dev["fold"].to_numpy()
    if NV >= ALT_VIEWS:
        k = ALT_VIEWS * len(dev)
        Fc = dict(F); Fc["X_aug"] = F["X_aug"][:k]; Fc["y_aug"] = F["y_aug"][:k]
        Fc["s_aug"] = np.asarray(F["s_aug"])[:k]; Fc["fold_aug"] = F["fold_aug"][:k]
        print(f"  control: resnet18 with only {ALT_VIEWS} aug views (matched budget)")
        sweep(f"bb_resnet18_ctl_v{ALT_VIEWS}", Fc)
    for bk in ALT_BACKBONES:
        t = time.time()
        path = Path(f"/kaggle/working/feats_{bk}_{CFG['central_lo']}-{CFG['central_hi']}"
                    f"_g{CFG['slice_gap']}_v{ALT_VIEWS}_seed{CFG['seed']}.pt")
        if path.exists():
            Fb = torch.load(path, weights_only=False); print(f"  loaded {path.name}")
        else:
            print(f"  extracting {bk}: 1 det + {ALT_VIEWS} aug views over {len(dev):,} imgs")
            m = feature_extractor(bk)
            Xd, yd, sd_ = extract(dev, m, False, 1)
            Xa, ya, sa = extract(dev, m, True, ALT_VIEWS)
            del m; torch.cuda.empty_cache()
            Fb = dict(X_det=Xd, y_det=yd, s_det=sd_, fold_det=torch.as_tensor(fold_det),
                      X_aug=Xa, y_aug=ya, s_aug=sa,
                      fold_aug=torch.as_tensor(np.tile(fold_det, ALT_VIEWS)))
            torch.save(Fb, path)
            print(f"  cached -> {path.name}  ({Xd.shape[1]}-d, {time.time()-t:.0f}s)")
        sweep(f"bb_{bk}", Fb)
        del Fb; torch.cuda.empty_cache()

# ============================================================== BLOCK 10 ====
def all_bn_eval(model):
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm): m.eval()

def bn_snapshot(model):
    return {n: (m.running_mean.detach().clone(), m.running_var.detach().clone())
            for n, m in model.named_modules()
            if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.running_mean is not None}

def bn_drift(model, snap, top=6):
    d = dict(model.named_modules())
    rows = [(n,
             (d[n].running_mean - mu).norm().item() / (mu.norm().item() + 1e-8),
             (d[n].running_var - var).norm().item() / (var.norm().item() + 1e-8))
            for n, (mu, var) in snap.items()]
    rows.sort(key=lambda r: -r[1])
    return rows[:top]

@torch.no_grad()
def eval_loader(model, loader, crit, batchstats=False):
    mdl = copy.deepcopy(model) if batchstats else model
    mdl.eval()
    if batchstats:
        for m in mdl.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm): m.train()
    ps, ys, ss, tot = [], [], [], 0.0
    for x, y, s in loader:
        lo = mdl(x.to(device))
        tot += crit(lo, y.to(device)).item() * x.size(0)
        ps.append(torch.softmax(lo.float(), 1).cpu().numpy())
        ys.append(y.numpy()); ss.extend(s)
    p = np.concatenate(ps); t = np.concatenate(ys)
    P, ts, _ = subj_pack(p, ss, t)
    if batchstats: del mdl; torch.cuda.empty_cache()
    return dict(loss=tot / len(loader.dataset), img=score(p, t), subj=score(P, ts))

def run_epoch_diag(model, loader, crit, opt, scaler, bn_eval_all):
    model.train(); freeze_frozen_bn(model)
    if bn_eval_all: all_bn_eval(model)
    tot = 0.0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        with torch.autocast(device_type="cuda", enabled=AMP):
            loss = crit(model(x), y)
        if scaler is not None:
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()
        tot += loss.item() * x.size(0)
    return tot / len(loader.dataset)

def b_finetune():
    banner(f"10. FINE-TUNE DIAGNOSTICS (fold {FT_FOLD} only, 4 variants)")
    tr = central[(central.fold != FT_FOLD) & (central.fold != -1)]
    va = central[central.fold == FT_FOLD]
    print(f"train {tr.subject_id.nunique()} subj / {len(tr):,} img | "
          f"val {va.subject_id.nunique()} subj / {len(va):,} img")
    va_dl = DataLoader(OasisDS(va, False), CFG["batch_size"], shuffle=False,
                       num_workers=CFG["num_workers"], pin_memory=True)
    VARIANTS = [
        dict(tag="A_baseline_repro", bn_eval_all=False, sampler=True,  lr_bb=CFG["lr_backbone"]),
        dict(tag="B_allBN_eval",     bn_eval_all=True,  sampler=True,  lr_bb=CFG["lr_backbone"]),
        dict(tag="C_BN+clsweight",   bn_eval_all=True,  sampler=False, lr_bb=CFG["lr_backbone"]),
        dict(tag="D_BN+clsw+lowLR",  bn_eval_all=True,  sampler=False, lr_bb=2e-6),
    ]
    for V in VARIANTS:
        t = time.time()
        print(f"\n--- {V['tag']}: bn_eval_all={V['bn_eval_all']} sampler={V['sampler']} "
              f"lr_backbone={V['lr_bb']}")
        set_seed(CFG["seed"])
        ds = OasisDS(tr, True)
        if V["sampler"]:
            sw = class_weights(tr).cpu().numpy()[tr["label"].to_numpy()]
            dl = DataLoader(ds, CFG["batch_size"],
                            sampler=WeightedRandomSampler(sw, len(sw), True),
                            num_workers=CFG["num_workers"], pin_memory=True, drop_last=True)
        else:
            dl = DataLoader(ds, CFG["batch_size"], shuffle=True,
                            num_workers=CFG["num_workers"], pin_memory=True, drop_last=True)
        model = build_model(CFG["backbone"]).to(device)
        crit = nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"],
                                   weight=None if V["sampler"] else class_weights(tr))
        scaler = torch.amp.GradScaler("cuda", enabled=AMP)

        set_trainable(model, CFG["backbone"], "head")
        opt = torch.optim.AdamW(param_groups(model, CFG["backbone"], V["lr_bb"],
                                             CFG["lr_head"]), weight_decay=CFG["weight_decay"])
        for e in range(FT_S1):
            print(f"  [s1 {e+1}] train_loss="
                  f"{run_epoch_diag(model, dl, crit, opt, scaler, V['bn_eval_all']):.4f}")
        m0 = eval_loader(model, va_dl, crit)
        print(f"  [s1 end] val_loss={m0['loss']:.4f} img F1={m0['img']['macro_f1']:.4f} | "
              f"subj {fmt(m0['subj'])}")

        set_trainable(model, CFG["backbone"], CFG["freeze_until"])
        snap = bn_snapshot(model)
        opt = torch.optim.AdamW(param_groups(model, CFG["backbone"], V["lr_bb"],
                                             CFG["lr_head_ft"]),
                                weight_decay=CFG["weight_decay"])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FT_S2)
        best, best_e, wait = m0["subj"]["macro_f1"], "s1", 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        for e in range(FT_S2):
            l = run_epoch_diag(model, dl, crit, opt, scaler, V["bn_eval_all"])
            m = eval_loader(model, va_dl, crit); sch.step()
            print(f"  [s2 {e+1:2d}] tr={l:.4f} val={m['loss']:.4f} gap={m['loss']-l:+.3f} "
                  f"imgF1={m['img']['macro_f1']:.4f} "
                  f"imgAUC=[{','.join(f'{v:.3f}' for v in m['img']['auc'])}] "
                  f"subjF1={m['subj']['macro_f1']:.4f} subjBal={m['subj']['bal_acc']:.4f} "
                  f"subjAUC=[{','.join('nan' if np.isnan(v) else f'{v:.3f}' for v in m['subj']['auc'])}]")
            if m["subj"]["macro_f1"] > best:
                best, best_e, wait = m["subj"]["macro_f1"], f"s2e{e+1}", 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= FT_PATIENCE: print("  early stop"); break
        print(f"  best subjF1={best:.4f} @ {best_e}   ({time.time()-t:.0f}s)")
        print("  BN running-stat drift during stage 2 (relative L2, worst 6):")
        for n, dm_, dv in bn_drift(model, snap):
            print(f"    {n:30s} d_mean={dm_:.4f} d_var={dv:.4f}")
        model.load_state_dict(best_state)
        rr = eval_loader(model, va_dl, crit)
        rb = eval_loader(model, va_dl, crit, batchstats=True)
        print(f"  BEST weights, RUNNING BN stats : {fmt(rr['subj'])}")
        print(f"  BEST weights, BATCH   BN stats : {fmt(rb['subj'])}"
              f"   <-- much higher => BN running stats ARE the problem")
        print(f"  subject confusion (running stats):\n{np.array2string(rr['subj']['cm'])}")
        print(classification_report(rr["subj"]["true"], rr["subj"]["pred"],
                                    target_names=CLASS_NAMES, labels=[0, 1, 2],
                                    zero_division=0))
        logrow(test=f"ft_{V['tag']}", best=round(best, 4),
               auc=np.round(rr["subj"]["auc"], 3).tolist())
        del model; torch.cuda.empty_cache()

# ================================================================== MAIN ====
sys.stdout = _Tee(LOGPATH, _real_stdout)
try:
    for nm, fn in (("audit", b_audit), ("geometry", b_geometry), ("leakage", b_leakage),
                   ("permutation", b_permutation), ("probe_variants", b_probe_variants),
                   ("aggregation", b_aggregation), ("subtasks", b_subtasks),
                   ("slice_window", b_slice_window), ("train_ceiling", b_train_ceiling),
                   ("backbones", b_backbones), ("finetune", b_finetune)):
        block(nm, fn)
    banner("SUMMARY TABLE")
    print(pd.DataFrame(ROWS).to_string(index=False))
    print(f"\nTOTAL RUNTIME {time.time()-T0:.0f}s")
finally:
    sys.stdout.flush(); sys.stdout = _real_stdout
print("diagnostics done ->", LOGPATH)
