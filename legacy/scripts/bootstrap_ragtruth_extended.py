import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

SEED=42; N_BOOT=5000
S2_MIN=-11.430; S2_MAX=10.641
FUSION_THRESHOLD=0.40; MC_HALL_THRESHOLD=0.80; ESC_PCT=20
OUT=Path('/workspace/bootstrap_ragtruth_extended_results.json')

def load(p):
    with open(p) as f: return json.load(f)

def norm_s2(x): return float(np.clip((float(x)-S2_MIN)/(S2_MAX-S2_MIN),0,1))

def ece(scores,labels,n_bins=10):
    scores=np.asarray(scores,float); labels=np.asarray(labels,int); bins=np.linspace(0,1,n_bins+1); total=0.0
    for i in range(n_bins):
        m=(scores>=bins[i]) & ((scores<=bins[i+1]) if i==n_bins-1 else (scores<bins[i+1]))
        if m.any(): total += m.sum()*abs(labels[m].mean()-scores[m].mean())
    return float(total/len(labels))

def metrics(y,s,p):
    y=np.asarray(y,int); s=np.asarray(s,float); p=np.asarray(p,int)
    return {'f1':float(f1_score(y,p,zero_division=0)), 'auroc':float(roc_auc_score(y,s)),
            'auprc':float(average_precision_score(y,s)), 'ece':ece(s,y)}

def summary(v):
    a=np.asarray(v,float)
    return {'mean':float(a.mean()),'lower_95':float(np.percentile(a,2.5)),'upper_95':float(np.percentile(a,97.5))}

rel_tr={r['idx']:r for r in load('/workspace/relevance_results_train_v2.json')}
rel_te={r['idx']:r for r in load('/workspace/relevance_results_test_v2.json')}
s4_tr={r['idx']:r for r in load('/workspace/signal4_results_train_oof.json')}
s4_te={r['idx']:r for r in load('/workspace/signal4_results_test.json')}
mc_te={r['idx']:r for r in load('/workspace/minicheck_results_test_7b.json')}

def extract(rel,s4):
    X=[]; y=[]; cats=[]; idxs=[]
    for idx in sorted(rel.keys() & s4.keys()):
        r2,r4=rel[idx],s4[idx]
        if r2['raw_min_relevance'] is None or r4['signal4_score'] is None: continue
        X.append([norm_s2(r2['raw_min_relevance']),float(r4['signal4_score'])])
        y.append(int(r4['ground_truth_hallucination'])); cats.append([r4['task_type'],r4['model']]); idxs.append(int(idx))
    return np.asarray(X,float),np.asarray(y,int),cats,idxs

Xtr,ytr,ctr,_=extract(rel_tr,s4_tr); Xte,yte,cte,test_idxs=extract(rel_te,s4_te)
assert len(ytr)==15090 and len(yte)==2700
try: ohe=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
except TypeError: ohe=OneHotEncoder(handle_unknown='ignore',sparse=False)
ohe.fit(ctr)
Xtr=np.hstack([Xtr,ohe.transform(ctr)]); Xte=np.hstack([Xte,ohe.transform(cte)])
clf=LogisticRegression(max_iter=1000,random_state=42).fit(Xtr,ytr)
fs=clf.predict_proba(Xte)[:,1]; fp=(fs>=FUSION_THRESHOLD).astype(int)
ms=[]
for idx in test_idxs:
    r=mc_te[idx]
    if r['minicheck_score'] is None: raise RuntimeError(f'Missing MiniCheck score at idx={idx}')
    ms.append(1.0-float(r['minicheck_score']))
ms=np.asarray(ms,float); mp=(ms>=MC_HALL_THRESHOLD).astype(int)

conf=np.abs(fs-0.5); nesc=int(len(yte)*ESC_PCT/100); esc=np.argsort(conf)[:nesc]
mask=np.zeros(len(yte),bool); mask[esc]=True
cs=fs.copy(); cs[mask]=ms[mask]
cp=fp.copy(); cp[mask]=mp[mask]

point={'fusion':metrics(yte,fs,fp),'minicheck_7b':metrics(yte,ms,mp),'cascade_20pct':metrics(yte,cs,cp)}
checks=[(point['fusion']['f1'],.7262,'fusion F1'),(point['fusion']['auroc'],.8749,'fusion AUROC'),
        (point['minicheck_7b']['f1'],.7260,'MC F1'),(point['minicheck_7b']['auroc'],.8754,'MC AUROC'),
        (point['cascade_20pct']['f1'],.7656,'cascade F1'),(point['cascade_20pct']['auroc'],.8750,'cascade AUROC')]
for a,e,n in checks:
    if abs(a-e)>.0002: raise RuntimeError(f'{n} mismatch: {a:.6f} vs ~{e:.4f}')

print('='*90); print('POINT ESTIMATES'); print('='*90)
for n,m in point.items(): print(f"{n:<16} F1={m['f1']:.4f} AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f} ECE={m['ece']:.4f}")

systems={'fusion':(fs,fp),'minicheck_7b':(ms,mp),'cascade_20pct':(cs,cp)}
comparisons={'fusion_minus_minicheck':('fusion','minicheck_7b'),
             'cascade20_minus_fusion':('cascade_20pct','fusion'),
             'cascade20_minus_minicheck':('cascade_20pct','minicheck_7b')}
metric_names=['f1','auroc','auprc','ece']
store={n:{k:[] for k in metric_names} for n in list(systems)+list(comparisons)}
rng=np.random.default_rng(SEED); pos=np.where(yte==1)[0]; neg=np.where(yte==0)[0]

for _ in range(N_BOOT):
    idx=np.concatenate([rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)])
    rng.shuffle(idx); yb=yte[idx]; bm={}
    for n,(s,p) in systems.items():
        m=metrics(yb,s[idx],p[idx]); bm[n]=m
        for k in metric_names: store[n][k].append(m[k])
    for cn,(l,r) in comparisons.items():
        for k in metric_names: store[cn][k].append(bm[l][k]-bm[r][k])

boot={n:{k:summary(v) for k,v in d.items()} for n,d in store.items()}
print('\n'+'='*90); print(f'95% STRATIFIED PAIRED BOOTSTRAP CIs ({N_BOOT} resamples)'); print('='*90)
for n in systems:
    print('\n'+n.upper())
    for k in metric_names:
        b=boot[n][k]; print(f"  {k.upper():<6} {point[n][k]:.4f} [{b['lower_95']:.4f}, {b['upper_95']:.4f}]")
for cn,(l,r) in comparisons.items():
    print(f'\nPAIRED DELTA: {l} - {r}')
    for k in metric_names:
        b=boot[cn][k]; d=point[l][k]-point[r][k]
        print(f"  {k.upper():<6} {d:+.4f} [{b['lower_95']:+.4f}, {b['upper_95']:+.4f}]")

out={'protocol':{'dataset':'RAGTruth test','n':int(len(yte)),'n_bootstrap':N_BOOT,
     'bootstrap':'stratified nonparametric paired bootstrap','strata':'ground-truth hallucination label',
     'ci':'95% percentile','fusion_threshold':FUSION_THRESHOLD,'minicheck_hallucination_threshold':MC_HALL_THRESHOLD,
     'selected_cascade_escalation_pct':ESC_PCT,'selected_cascade_n_escalated':int(nesc),
     'cascade_selection_rule':'lowest absolute fusion confidence |p-0.5| first',
     'cascade_bootstrap_treatment':'20% cascade vector reconstructed once on original test set; same paired bootstrap indices applied to all systems.',
     'important_note':'CIs quantify test-sample uncertainty for fixed trained models and fixed 20% operating point; they do not capture training uncertainty or uncertainty from post-hoc selection of 20% after inspecting the test cascade curve.'},
     'point_estimates':point,'bootstrap':boot}
with open(OUT,'w') as f: json.dump(out,f,indent=2)
print(f'\nSaved: {OUT}')
