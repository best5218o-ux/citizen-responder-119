import pandas as pd, numpy as np, json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
F='ems.csv'  # 소방안전 빅데이터 플랫폼 「소방청_전국 구급 현황」 2025년 원자료 표본(1월 초)
d=pd.read_csv(F,dtype=str,encoding='utf-8-sig')
def ts(y,t):
    return pd.to_datetime(d[y].str.strip()+d[t].str.strip().str.zfill(6),format='%Y%m%d%H%M%S',errors='coerce')
d['disp']=ts('dspt_ymd','dspt_tm'); d['arr']=ts('grnds_arvl_ymd','grnds_arvl_tm')
d['m']=(d.arr-d.disp).dt.total_seconds()/60
d['km']=pd.to_numeric(d.grnds_dstnc,errors='coerce')
d=d[(d.m>0)&(d.m<120)&(d.km>=0)&(d.km<100)].copy()
d['hr']=pd.to_numeric(d.dspt_hr,errors='coerce')
d['tb']=pd.cut(d.hr,[-1,5,11,17,23],labels=['심야(0-5시)','오전(6-11시)','오후(12-17시)','저녁(18-23시)']).astype(str)
d['sido']=d.ctpv_nm.fillna('미상').str.strip(); d['ur']=d.cty_frmvl_se_nm.fillna('미상').str.strip()
d['ca']=d.hrtarst_nm.fillna('').str.strip()!=''
d['y']=(d.m>6).astype(int)
print('n',len(d),'cardiac',d.ca.sum(),'y rate',d.y.mean().round(3))
print(d.hrtarst_nm.value_counts().head())
X=pd.get_dummies(d[['sido','ur','tb']],dtype=float)
X['km']=d.km; X['logkm']=np.log1p(d.km)
Xtr,Xte,ytr,yte,ctr,cte=train_test_split(X,d.y,d.ca,test_size=0.3,random_state=7,stratify=d.y)
lr=LogisticRegression(max_iter=3000,C=1.0).fit(Xtr,ytr)
p=lr.predict_proba(Xte)[:,1]
print('AUC',round(roc_auc_score(yte,p),3),'Brier',round(brier_score_loss(yte,p),3))
if cte.sum()>20: print('AUC cardiac',round(roc_auc_score(yte[cte],p[cte.values]),3),'n',int(cte.sum()))
# baseline distance only
lr0=LogisticRegression(max_iter=2000).fit(Xtr[['km','logkm']],ytr); print('AUC km-only',round(roc_auc_score(yte,lr0.predict_proba(Xte[['km','logkm']])[:,1]),3))
# 6min threshold policy at p>=0.5: share flagged & recall
for th in [0.4,0.5,0.6]:
    f=p>=th; print(th,'flag',round(f.mean(),3),'recall',(f&(yte==1)).sum()/ (yte==1).sum(),'precision',(f&(yte==1)).sum()/max(f.sum(),1))
lr=LogisticRegression(max_iter=3000).fit(X,d.y)
model={'intercept':float(lr.intercept_[0]),'coef':{c:float(v) for c,v in zip(X.columns,lr.coef_[0])},
 'n':int(len(d)),'auc':float(roc_auc_score(yte,p)),'sidos':sorted(d.sido.unique().tolist())}
# distance median by sido/ur for default
model['km_med']={f'{s}|{u}':float(g.km.median()) for (s,u),g in d.groupby(['sido','ur'])}
json.dump(model,open('model.json','w'),ensure_ascii=False,indent=0)
print(len(model['coef']))
