import streamlit as st
import pandas as pd, numpy as np, requests
from datetime import date,timedelta

st.set_page_config(page_title='家銘台股雷達 V5.1',page_icon='📡',layout='wide')
st.title('📡 家銘台股雷達 V5.1｜品質 × 進場雙分數')
st.caption('先找好股票，再判斷現在是不是好買點｜技術＋籌碼＋基本面＋市場環境＋風險否決器')
API='https://api.finmindtrade.com/api/v4/data'
DEFAULT='''2330\n2317\n2454\n2303\n2382\n3231\n2379\n3034\n3711\n2409\n3481\n2344\n2603\n2609\n2002\n1301\n1303\n2881\n2882\n2886'''

def call(ds,sid,start,end,token):
    q={'dataset':ds,'data_id':sid,'start_date':str(start),'end_date':str(end)}
    h={'Authorization':f'Bearer {token}'} if token else {}
    r=requests.get(API,params=q,headers=h,timeout=30); r.raise_for_status()
    js=r.json()
    if js.get('status') not in (None,200): raise RuntimeError(js.get('msg','FinMind API error'))
    return pd.DataFrame(js.get('data',[]))

@st.cache_data(ttl=3600,show_spinner=False)
def get(ds,sid,start,end,token):
    try:return call(ds,sid,start,end,token)
    except:return pd.DataFrame()

def price(sid,start,end,token):
    d=get('TaiwanStockPrice',sid,start,end,token)
    if d.empty:return d
    d['date']=pd.to_datetime(d['date']); d=d.sort_values('date')
    d=d.rename(columns={'close':'Close','Trading_Volume':'Volume'})
    d['Close']=pd.to_numeric(d['Close'],errors='coerce'); d['Volume']=pd.to_numeric(d['Volume'],errors='coerce')
    return d.dropna(subset=['Close']).reset_index(drop=True)

def feat(d):
    x=d.copy()
    for n in [5,10,20,60,120,200]:x[f'MA{n}']=x.Close.rolling(n).mean()
    x['VR']=x.Volume/x.Volume.rolling(20).mean(); x['R20']=x.Close.pct_change(20); x['R60']=x.Close.pct_change(60)
    x['HH20']=x.Close.rolling(20).max(); x['LL20']=x.Close.rolling(20).min()
    delta=x.Close.diff(); g=delta.clip(lower=0).rolling(14).mean(); l=(-delta.clip(upper=0)).rolling(14).mean()
    x['RSI']=100-100/(1+g/l.replace(0,np.nan))
    e12=x.Close.ewm(span=12,adjust=False).mean(); e26=x.Close.ewm(span=26,adjust=False).mean()
    x['MACD']=e12-e26; x['SIG']=x.MACD.ewm(span=9,adjust=False).mean()
    return x

def tech_quality(x):
    r=x.iloc[-1]; s=0; w=[]
    checks=[(r.Close>r.MA20,4,'站上20MA'),(r.MA5>r.MA10>r.MA20,5,'短中期多頭排列'),(r.MA20>r.MA60,4,'20MA>60MA'),
            (r.MA60>r.MA120,3,'中期趨勢偏多'),(r.MA20>x.MA20.iloc[-6],3,'20MA上彎'),(r.R20>.03,3,'20日動能正向'),
            (r.R60>.08,2,'60日動能強'),(r.Close>=r.HH20*.98,2,'接近20日高'),(r.VR>=1.2,2,'量能放大'),(r.MACD>r.SIG,2,'MACD偏多')]
    for ok,p,msg in checks:
        if pd.notna(ok) and bool(ok):s+=p;w.append(msg)
    return min(s,30),w

def chip_score(d):
    if d.empty:return 15,['法人資料不足：中性暫估'],False
    z=d.copy(); buy=[c for c in z if 'buy' in c.lower()]; sell=[c for c in z if 'sell' in c.lower()]
    if not buy or not sell:return 15,['法人欄位不足：中性暫估'],False
    for c in buy+sell:z[c]=pd.to_numeric(z[c],errors='coerce').fillna(0)
    z['net']=z[buy].sum(axis=1)-z[sell].sum(axis=1)
    if 'date' in z:z['date']=pd.to_datetime(z['date']); daily=z.groupby('date')['net'].sum().sort_index()
    else:daily=z['net']
    s=0;w=[]
    for ok,p,msg in [(daily.tail(5).sum()>0,10,'近5日法人淨買'),(daily.tail(10).sum()>0,7,'近10日法人淨買'),
                     ((daily.tail(5)>0).sum()>=3,7,'近5日至少3日淨買'),(daily.tail(20).sum()>0,4,'近20日法人偏多')]:
        if ok:s+=p;w.append(msg)
    # 投信/外資名稱若 API 有提供，額外顯示，不重複灌分
    if 'name' in z.columns:
        names='、'.join(map(str,z['name'].dropna().astype(str).unique()[:5]));
        if names:w.append('法人來源：'+names)
    return min(s,28),w,True

def margin_score(d):
    if d.empty:return 1,['融資券資料不足'],False
    z=d.copy(); cols={c.lower():c for c in z.columns}
    # 不假設固定 schema；只在找到 balance 類欄位時判斷最近變化
    candidates=[c for c in z.columns if 'margin' in c.lower() and ('balance' in c.lower() or 'today' in c.lower())]
    if not candidates:return 1,['融資券欄位不足'],False
    c=candidates[0]; z[c]=pd.to_numeric(z[c],errors='coerce'); z=z.dropna(subset=[c])
    if len(z)<6:return 1,['融資券歷史不足'],False
    ch=z[c].iloc[-1]/z[c].iloc[-6]-1 if z[c].iloc[-6] else np.nan
    if pd.notna(ch) and ch<-.03:return 2,[f'近5期融資下降 {ch:.1%}'],True
    if pd.notna(ch) and ch>.08:return 0,[f'近5期融資增加 {ch:.1%}'],True
    return 1,[f'近5期融資變化 {ch:.1%}' if pd.notna(ch) else '融資變化中性'],True

def revenue_score(d):
    if d.empty:return 7,['月營收資料不足'],False
    z=d.copy(); cols=[c for c in z if 'revenue' in c.lower() and c.lower() not in ('revenue_month','revenue_year')]
    if not cols:return 7,['營收欄位不足'],False
    c=cols[0]; z[c]=pd.to_numeric(z[c],errors='coerce'); z=z.dropna(subset=[c])
    if 'date' in z:z['date']=pd.to_datetime(z['date']);z=z.sort_values('date')
    if len(z)<13:return 7,['營收歷史不足'],False
    yoy=z[c].iloc[-1]/z[c].iloc[-13]-1 if z[c].iloc[-13] else np.nan
    mom=z[c].iloc[-1]/z[c].iloc[-2]-1 if z[c].iloc[-2] else np.nan
    s=3;w=[]
    if pd.notna(yoy):
        w.append(f'營收YoY {yoy:.1%}'); s+=7 if yoy>.20 else 5 if yoy>.10 else 3 if yoy>0 else 0
    if pd.notna(mom):
        w.append(f'營收MoM {mom:.1%}'); s+=2 if mom>0 else 0
    return min(s,12),w,True

def financial_score(d):
    # FinMind 財報資料格式可能因版本不同；採欄位/指標文字搜尋，找不到就不硬算
    if d.empty:return 6,['財報資料不足'],False
    z=d.copy(); textcol=next((c for c in z.columns if c.lower() in ('type','name','item','account')),None)
    valcol=next((c for c in z.columns if c.lower() in ('value','amount')),None)
    if not textcol or not valcol:return 6,['財報欄位不足'],False
    z[valcol]=pd.to_numeric(z[valcol],errors='coerce'); z=z.dropna(subset=[valcol])
    def latest(keys):
        q=z[z[textcol].astype(str).str.lower().apply(lambda s:any(k in s for k in keys))]
        if q.empty:return np.nan
        if 'date' in q:q=q.sort_values('date')
        return q[valcol].iloc[-1]
    eps=latest(['eps','earningspershare','basic earnings per share'])
    roe=latest(['roe','returnonequity','return on equity'])
    gp=latest(['grossprofit','gross profit']); revv=latest(['revenue','operatingrevenue','operating revenue'])
    op=latest(['operatingincome','operating income','operatingprofit'])
    gm=gp/revv if pd.notna(gp) and pd.notna(revv) and revv else np.nan
    om=op/revv if pd.notna(op) and pd.notna(revv) and revv else np.nan
    s=0;w=[];found=0
    if pd.notna(eps):found+=1;w.append(f'EPS {eps:.2f}');s+=3 if eps>0 else 0
    if pd.notna(roe):found+=1; roe_pct=roe if abs(roe)>1 else roe*100;w.append(f'ROE {roe_pct:.1f}%');s+=4 if roe_pct>=15 else 2 if roe_pct>=8 else 0
    if pd.notna(gm):found+=1;w.append(f'毛利率 {gm:.1%}');s+=3 if gm>.25 else 2 if gm>.15 else 1 if gm>0 else 0
    if pd.notna(om):found+=1;w.append(f'營益率 {om:.1%}');s+=3 if om>.15 else 2 if om>.08 else 1 if om>0 else 0
    if not found:return 6,['財報指標未辨識：中性暫估'],False
    return min(s,13),w,True

def market_score(d):
    if d.empty or len(d)<200:return 7,['市場資料不足']
    x=feat(d);r=x.iloc[-1];s=0;w=[]
    for ok,p,msg in [(r.Close>r.MA20,4,'市場>20MA'),(r.Close>r.MA60,4,'市場>60MA'),(r.MA20>r.MA60,4,'市場中期多頭'),(r.MA60>r.MA200,3,'市場長期偏多')]:
        if bool(ok):s+=p;w.append(msg)
    return min(s,15),w

def entry_score(x,quality):
    r=x.iloc[-1];g5=r.Close/r.MA5-1;g20=r.Close/r.MA20-1;s=50;reasons=[];veto=[]
    # 趨勢確認
    if r.Close>r.MA20:s+=8;reasons.append('站上20MA')
    if r.MA5>r.MA10>r.MA20:s+=8;reasons.append('5>10>20MA')
    if r.MACD>r.SIG:s+=5;reasons.append('MACD偏多')
    if 50<=r.RSI<=68:s+=7;reasons.append('RSI健康')
    if 1.0<=r.VR<=2.2:s+=5;reasons.append('量能合理')
    if -.01<=g5<=.04:s+=7;reasons.append('靠近5MA')
    if quality>=75:s+=5;reasons.append('品質A級以上')
    # 過熱/破線扣分
    if g5>.08:s-=35;veto.append(f'5MA乖離 {g5:.1%}')
    elif g5>.05:s-=15;veto.append(f'5MA乖離偏高 {g5:.1%}')
    if r.R20>.30:s-=30;veto.append(f'20日漲幅 {r.R20:.1%}')
    elif r.R20>.20:s-=12;veto.append(f'20日漲幅偏大 {r.R20:.1%}')
    if r.RSI>78:s-=30;veto.append(f'RSI {r.RSI:.0f} 過熱')
    elif r.RSI>72:s-=12;veto.append(f'RSI {r.RSI:.0f} 偏熱')
    if r.Close<r.MA20:s-=35;veto.append('跌破20MA')
    if g20<-.05:s-=15;veto.append('明顯低於20MA')
    s=int(np.clip(s,0,100))
    status='🟢 可觀察' if s>=75 and not veto else '🟡 等待買點' if s>=55 else '🔴 暫不進場'
    return s,status,reasons,veto,g5

def grade(s):return 'S' if s>=85 else 'A' if s>=75 else 'B' if s>=65 else 'C'

with st.sidebar:
    token=st.text_input('FinMind Token',type='password')
    days=st.slider('技術資料天數',300,800,500,50)
    st.caption('V5.1：品質分數與進場分數分離。資料缺漏會標示 △，不把缺資料當成強勢訊號。')
stocks=[x.strip() for x in st.text_area('股票池｜一行一個代號',DEFAULT,height=220).splitlines() if x.strip()]

if st.button('🚀 啟動 V5.1 雷達',type='primary',use_container_width=True):
    end=date.today();start=end-timedelta(days=days)
    m=price('0050',start,end,token); ms,mw=market_score(m)
    rows=[];detail={};bar=st.progress(0)
    for i,sid in enumerate(stocks):
        try:
            p=price(sid,start,end,token)
            if len(p)<150:continue
            x=feat(p).dropna().reset_index(drop=True); ts,tw=tech_quality(x)
            ins=get('TaiwanStockInstitutionalInvestorsBuySell',sid,end-timedelta(days=60),end,token); cs,cw,cok=chip_score(ins)
            mar=get('TaiwanStockMarginPurchaseShortSale',sid,end-timedelta(days=60),end,token); mgs,mgw,mgok=margin_score(mar); cs=min(cs+mgs,30)
            rv=get('TaiwanStockMonthRevenue',sid,end-timedelta(days=500),end,token); rs,rw,rok=revenue_score(rv)
            fsd=get('TaiwanStockFinancialStatements',sid,end-timedelta(days=800),end,token); fss,fsw,fsok=financial_score(fsd); fs=min(rs+fss,25)
            quality=int(ts+cs+fs+ms); entry,status,er,flags,g5=entry_score(x,quality); r=x.iloc[-1]
            rows.append([sid,quality,grade(quality),entry,status,ts,cs,fs,ms,r.Close,g5,r.R20,'✓' if cok else '△','✓' if (rok or fsok) else '△'])
            detail[sid]=(tw,cw+mgw,rw+fsw,er,flags)
        except Exception as e:
            detail[sid]=([],[],[f'錯誤：{e}'],[],[])
        bar.progress((i+1)/max(len(stocks),1),text=f'分析 {i+1}/{len(stocks)}｜{sid}')
    bar.empty()
    if not rows:st.error('沒有成功取得資料。請檢查 FinMind Token 或稍後再試。');st.stop()
    out=pd.DataFrame(rows,columns=['代號','品質分','級別','進場分','狀態','技術','籌碼','基本','環境','最新價','5MA乖離','20日漲跌','籌碼資料','基本資料'])
    # 主榜依品質；買點榜依進場分，避免「高品質但過熱」混在可買候選
    quality_board=out.sort_values(['品質分','進場分'],ascending=False)
    show=quality_board.copy();show['5MA乖離']=show['5MA乖離'].map(lambda v:f'{v:.1%}');show['20日漲跌']=show['20日漲跌'].map(lambda v:f'{v:.1%}')
    st.subheader('🏆 股票品質排行榜');st.dataframe(show,use_container_width=True,hide_index=True)
    st.subheader('🎯 現在的進場候選')
    entry_board=out[(out['品質分']>=65)&(out['進場分']>=55)].sort_values(['進場分','品質分'],ascending=False)
    if entry_board.empty:st.info('目前沒有同時通過品質與進場門檻的股票。這是正常結果，不強迫產生買點。')
    for _,r in entry_board.head(10).iterrows():
        tw,cw,fw,er,flags=detail[r.代號]
        with st.expander(f"{r.級別}｜{r.代號}｜品質 {int(r.品質分)}/100｜進場 {int(r.進場分)}/100｜{r.狀態}"):
            a,b,c,d=st.columns(4);a.metric('技術',f'{r.技術}/30');b.metric('籌碼',f'{r.籌碼}/30');c.metric('基本',f'{r.基本}/25');d.metric('環境',f'{r.環境}/15')
            st.write('技術：'+('、'.join(tw) if tw else '無加分'))
            st.write('籌碼：'+('、'.join(cw) if cw else '資料不足'))
            st.write('基本：'+('、'.join(fw) if fw else '資料不足'))
            st.write('進場加分：'+('、'.join(er) if er else '無'))
            if flags:st.warning('風險扣分／否決：'+'、'.join(flags))
    st.subheader('🚫 高品質但目前不宜追價')
    blocked=out[(out['品質分']>=75)&(out['進場分']<55)].sort_values('品質分',ascending=False)
    if blocked.empty:st.caption('目前沒有。')
    else:st.dataframe(blocked[['代號','品質分','級別','進場分','狀態','最新價']],use_container_width=True,hide_index=True)

st.caption('V5.1｜研究用途，不構成投資建議。FinMind 不同權限/版本可能缺少部分資料；△ 代表模組資料不足，系統採保守中性處理。')
