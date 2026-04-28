import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import joblib
import os
import urllib.request
import datetime
import json
from groq import Groq

# ฟังก์ชันสำหรับโหลดโมเดล (ปรับปรุงใหม่)
@st.cache_resource
def load_all():
    # กำหนดชื่อไฟล์และที่อยู่ให้ชัดเจน
    file_name = "ensemble_short_models.pkl"
    # ใช้ os.getcwd() เพื่อหาโฟลเดอร์ที่แอปกำลังรันอยู่จริงๆ
    save_path = os.path.join(os.getcwd(), file_name)
    
    url = f"https://drive.google.com/uc?export=download&id=1wTbk28p5NzW9l40-XWWWZZNlDG-pYsiW"
    
    # 1. ดาวน์โหลดถ้ายังไม่มีไฟล์
    if not os.path.exists(save_path):
        try:
            with st.spinner('กำลังดึงข้อมูลโมเดล...'):
                urllib.request.urlretrieve(url, save_path)
        except Exception as e:
            st.error(f"Download Error: {e}")
            return None

    # 2. โหลดไฟล์จากพาธที่ระบุไว้ชัดเจน
    try:
        # ตรวจสอบอีกทีว่าไฟล์มาจริงไหมก่อนโหลด
        if os.path.exists(save_path):
            model = joblib.load(save_path)
            return model
        else:
            st.error("ไฟล์หายไประหว่างทาง!")
            return None
    except Exception as e:
        st.error(f"Load Error: {e}")
        return None

md = load_all()
    

st.set_page_config(
    page_title="SME Early Warning System",
    page_icon="🚨",
    layout="wide"
)

font_path = "Sarabun-Regular.ttf"
if not os.path.exists(font_path):
    urllib.request.urlretrieve(
        "https://github.com/google/fonts/raw/main/ofl/sarabun/Sarabun-Regular.ttf",
        font_path)
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Sarabun'

@st.cache_resource
def load_all():
    return {
        'g1_short':    joblib.load("ensemble_short_models.pkl"),
        'g1_long':     joblib.load("ensemble_long_models.pkl"),
        'g1_sc_s':     joblib.load("scaler_g1_short.pkl"),
        'g1_sc_l':     joblib.load("scaler_g1_long.pkl"),
        'g1_f_s':      joblib.load("features_g1_short.pkl"),
        'g1_f_l':      joblib.load("features_g1_long.pkl"),
        'g1_le':       joblib.load("label_encoder_g1.pkl"),
        'g1_w_s':      joblib.load("weights_short.pkl"),
        'g1_w_l':      joblib.load("weights_long.pkl"),
        'g1_avg':      joblib.load("avg_by_province_month.pkl"),
        'g2_short':    joblib.load("ensemble_short_models_g2.pkl"),
        'g2_long':     joblib.load("ensemble_long_models_g2.pkl"),
        'g2_sc_s':     joblib.load("scaler_g2_short.pkl"),
        'g2_sc_l':     joblib.load("scaler_g2_long.pkl"),
        'g2_f_s':      joblib.load("features_g2_short.pkl"),
        'g2_f_l':      joblib.load("features_g2_long.pkl"),
        'g2_le':       joblib.load("label_encoder_g2.pkl"),
        'g2_w_s':      joblib.load("weights_short_g2.pkl"),
        'g2_w_l':      joblib.load("weights_long_g2.pkl"),
        'g2_avg':      joblib.load("avg_revenue_by_province_month.pkl"),
        'g3_clf':      joblib.load("model_g3_classifier.pkl"),
        'g3_sc':       joblib.load("scaler_g3.pkl"),
        'g3_robust':   joblib.load("robust_scaler_g3.pkl"),
        'g3_le_prov':  joblib.load("label_encoder_g3_province.pkl"),
        'g3_le_sea':   joblib.load("label_encoder_g3_season.pkl"),
        'g3_feats':    joblib.load("features_g3.pkl"),
        'g3_avg_t':    joblib.load("avg_tourist_g3.pkl"),
        'g3_avg_r':    joblib.load("avg_revenue_g3.pkl"),
        'metrics':     joblib.load("model_metrics.pkl"),
    }

md = load_all()

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
groq_client  = Groq(api_key=GROQ_API_KEY)

months_th   = ["ม.ค.","ก.พ.","มี.ค.","เม.ย.","พ.ค.","มิ.ย.",
               "ก.ค.","ส.ค.","ก.ย.","ต.ค.","พ.ย.","ธ.ค."]
months_full = ["มกราคม","กุมภาพันธ์","มีนาคม","เมษายน",
               "พฤษภาคม","มิถุนายน","กรกฎาคม","สิงหาคม",
               "กันยายน","ตุลาคม","พฤศจิกายน","ธันวาคม"]
biz_types   = ["ที่พัก/โรงแรม","ร้านอาหาร/คาเฟ่",
               "ทัวร์/นำเที่ยว","รถเช่า",
               "ของที่ระลึก/ของฝาก","สปา/นวด"]
tier1_provs = ['กรุงเทพมหานคร','ภูเก็ต','เชียงใหม่',
               'ชลบุรี','กาญจนบุรี']
horizon_options = {
    "3 เดือนข้างหน้า":  3,
    "6 เดือนข้างหน้า":  6,
    "1 ปีข้างหน้า":    12,
    "2 ปีข้างหน้า":    24,
    "3 ปีข้างหน้า":    36,
}
season_config = {
    'Golden Opportunity': {
        'emoji':'🌟','color':'#fef9c3','border':'#eab308',
        'label':'โอกาสทอง','risk_base':15,
    },
    'Normal': {
        'emoji':'✅','color':'#f0fdf4','border':'#22c55e',
        'label':'ปกติ','risk_base':30,
    },
    'Mixed': {
        'emoji':'⚖️','color':'#eff6ff','border':'#3b82f6',
        'label':'ผสมผสาน','risk_base':50,
    },
    'Survival': {
        'emoji':'⚠️','color':'#fff7ed','border':'#f97316',
        'label':'อยู่รอด','risk_base':70,
    },
    'Critical Risk': {
        'emoji':'🚨','color':'#fef2f2','border':'#ef4444',
        'label':'เสี่ยงสูง','risk_base':88,
    },
}
biz_kpi = {
    "ที่พัก/โรงแรม": {
        "kpi": ["Occupancy Rate","ราคาห้องเฉลี่ย",
                "รายได้/ห้อง/วัน","OTA Ranking"],
        "low_action":  "ลด Rate ใน OTA 15% และเพิ่ม Free Breakfast ดึงลูกค้า",
        "high_action": "ปิด OTA บางส่วน เน้น Direct Booking Margin สูงกว่า",
        "cost_focus":  "ค่าไฟ ค่าแม่บ้าน ค่า OTA Commission 15-20%",
    },
    "ร้านอาหาร/คาเฟ่": {
        "kpi": ["ลูกค้าต่อวัน","ต้นทุนต่อจาน",
                "รอบโต๊ะ/วัน","ของเสียหาย %"],
        "low_action":  "ลดเมนูเหลือ Top 10 ที่ขายดีสุด ลด Food Waste 20-30%",
        "high_action": "เพิ่มรอบ Turnover จาก 2→3 รอบ/วัน เพิ่มรายได้ 50%",
        "cost_focus":  "วัตถุดิบ 30-35% ค่าแรง ค่าเช่า",
    },
    "ทัวร์/นำเที่ยว": {
        "kpi": ["อัตราจอง","อัตรายกเลิก",
                "ต้นทุน/คน","คะแนนรีวิว"],
        "low_action":  "จับมือ Klook/GetYourGuide เพิ่มช่องทางขาย",
        "high_action": "เพิ่ม Premium Tour ราคาสูงขึ้น 30% Margin ดีกว่า",
        "cost_focus":  "ค่าเชื้อเพลิง ค่าไกด์ ค่าประกัน",
    },
    "รถเช่า": {
        "kpi": ["อัตราการใช้งาน %","รายได้/คัน/วัน",
                "ค่าซ่อม","วันว่าง"],
        "low_action":  "ลดราคา 20% แพ็คเกจรายสัปดาห์ ลด Idle Days",
        "high_action": "เพิ่มรถ Premium SUV Margin สูงกว่า Economy 40%",
        "cost_focus":  "ค่าซ่อมบำรุง ค่าประกัน ค่าเสื่อมราคา",
    },
    "ของที่ระลึก/ของฝาก": {
        "kpi": ["อัตราซื้อ","ยอดเฉลี่ย/ครั้ง",
                "หมุนเวียนสต็อก","ลูกค้าซ้ำ %"],
        "low_action":  "จัด Bundle Set 3 ชิ้น ลด 15% เพิ่ม Basket Size",
        "high_action": "เพิ่มสินค้า Exclusive Margin สูง ขายเฉพาะที่ร้าน",
        "cost_focus":  "สต็อกสินค้า หลีกเลี่ยง Overstock ค่าเช่าพื้นที่",
    },
    "สปา/นวด": {
        "kpi": ["อัตราการใช้ห้อง %","รายได้/ชั่วโมง",
                "ลูกค้าซ้ำ %","อัตราไม่มา"],
        "low_action":  "โปรโมชัน Couple Package ลด 20% ดึงลูกค้าใหม่",
        "high_action": "เพิ่ม Add-on Aromatherapy เพิ่มรายได้ต่อ Visit",
        "cost_focus":  "ค่าแรงพนักงาน 40-50% ค่าผลิตภัณฑ์ ค่าเช่า",
    },
}

# ── UI Helpers ────────────────────────────────────
def small_card(col, title, value, caption,
               color="#1e293b", size=18):
    with col:
        st.markdown(
            f"<div style='padding:10px;background:#f8fafc;"
            f"border-radius:8px;border:1px solid #e2e8f0;"
            f"min-height:110px'>"
            f"<p style='margin:0;font-size:11px;color:#666'>"
            f"{title}</p>"
            f"<p style='margin:2px 0;font-size:{size}px;"
            f"font-weight:bold;color:{color}'>{value}</p>"
            f"<p style='margin:0;font-size:11px;color:#94a3b8'>"
            f"{caption}</p>"
            f"</div>",
            unsafe_allow_html=True)

def risk_card(col, title, score, level, color):
    with col:
        st.markdown(
            f"<div style='text-align:center;padding:12px;"
            f"background:#f8fafc;border-radius:10px;"
            f"border:2px solid {color};min-height:110px'>"
            f"<p style='margin:0;font-size:11px;color:#666'>"
            f"{title}</p>"
            f"<p style='margin:2px 0;font-size:22px;"
            f"font-weight:bold;color:{color}'>{score}</p>"
            f"<p style='margin:0;font-size:10px;color:#666'>/100</p>"
            f"<p style='margin:2px 0;font-size:12px;"
            f"font-weight:bold;color:{color}'>ระดับ{level}</p>"
            f"</div>",
            unsafe_allow_html=True)

# ── Helper Functions ──────────────────────────────
def get_tier(province):
    if province in tier1_provs: return 1
    avg = md['g1_avg'][
        md['g1_avg']['province_thai']==province]\
        ['avg_tourist'].mean()
    return 2 if avg >= 200000 else 3

def get_avg_lag_t(province, month):
    prev = month-1 if month>1 else 12
    c = md['g1_avg'][(md['g1_avg']['province_thai']==province)&
                     (md['g1_avg']['month']==month)]
    p = md['g1_avg'][(md['g1_avg']['province_thai']==province)&
                     (md['g1_avg']['month']==prev)]
    if len(c)==0 or len(p)==0: return None, None
    return p['avg_tourist'].values[0], c['avg_tourist'].values[0]

def get_avg_lag_r(province, month):
    prev = month-1 if month>1 else 12
    c = md['g2_avg'][(md['g2_avg']['province_thai']==province)&
                     (md['g2_avg']['month']==month)]
    p = md['g2_avg'][(md['g2_avg']['province_thai']==province)&
                     (md['g2_avg']['month']==prev)]
    if len(c)==0 or len(p)==0: return None, None
    return p['avg_revenue'].values[0], c['avg_revenue'].values[0]

def build_g1_input(penc, month, year, l1, l12, tier, feats):
    is_covid = 1 if year in [2020,2021] else 0
    mv = (l1+l12)/2
    gr = np.clip((l1-l12)/max(l12,1),-1,10)
    row = {
        'province_enc':   penc,'month':month,'year':year,
        'is_covid':       is_covid,'tier':tier,
        'month_sin':      np.sin(2*np.pi*month/12),
        'month_cos':      np.cos(2*np.pi*month/12),
        'lag_1_log':      np.log1p(l1),
        'lag_12_log':     np.log1p(l12),
        'moving_avg_log': np.log1p(mv),
        'growth_rate_12': gr,
    }
    return pd.DataFrame([{f: row[f] for f in feats}])

def pred_ensemble(models, X_sc, weights):
    preds = [np.maximum(np.expm1(m.predict(X_sc)),0)
             for m in models.values()]
    w = [weights['rf'], weights['xgb']]
    return float(sum(p*wi for p,wi in zip(preds,w))[0])

def predict_g1(penc, month, year, l1t, l12t, tier, ma):
    is_s  = ma <= 6
    feats = md['g1_f_s'] if is_s else md['g1_f_l']
    sc    = md['g1_sc_s'] if is_s else md['g1_sc_l']
    mods  = md['g1_short'] if is_s else md['g1_long']
    w     = md['g1_w_s'] if is_s else md['g1_w_l']
    X = build_g1_input(penc, month, year, l1t, l12t, tier, feats)
    return pred_ensemble(mods, sc.transform(X), w)

def predict_g2(penc, month, year, l1r, l12r, tier, ma):
    is_s  = ma <= 6
    feats = md['g2_f_s'] if is_s else md['g2_f_l']
    sc    = md['g2_sc_s'] if is_s else md['g2_sc_l']
    mods  = md['g2_short'] if is_s else md['g2_long']
    w     = md['g2_w_s'] if is_s else md['g2_w_l']
    X = build_g1_input(penc, month, year, l1r, l12r, tier, feats)
    return pred_ensemble(mods, sc.transform(X), w)

def predict_g3(province, month, year,
               l1t, l12t, l1r, l12r, tourist, revenue):
    if province not in md['g3_le_prov'].classes_:
        return 'Normal', 50.0
    penc = md['g3_le_prov'].transform([province])[0]
    tourist_growth = np.clip((l1t-l12t)/max(l12t,1),-1,5)
    revenue_growth = np.clip((l1r-l12r)/max(l12r,1),-1,5)
    volatility_raw = (abs(tourist_growth)+abs(revenue_growth))/2
    robust_cols    = list(md['g3_robust'].feature_names_in_)
    robust_row = {
        'value_tourist_log':  np.log1p(max(tourist,0)),
        'value_revenue_log':  np.log1p(max(revenue,0)),
        'lag1_tourist_log':   np.log1p(max(l1t,0)),
        'lag12_tourist_log':  np.log1p(max(l12t,0)),
        'lag1_revenue_log':   np.log1p(max(l1r,0)),
        'lag12_revenue_log':  np.log1p(max(l12r,0)),
        'tourist_growth_12m': tourist_growth,
        'revenue_growth_12m': revenue_growth,
        'volatility':         volatility_raw,
    }
    X_r    = pd.DataFrame(
        [[robust_row[c] for c in robust_cols]],
        columns=robust_cols)
    X_r_sc = md['g3_robust'].transform(X_r)
    X_r_df = pd.DataFrame(X_r_sc, columns=robust_cols)
    feat_row = {
        'province_enc':       penc,
        'month':              month,
        'year':               year,
        'lag1_tourist_log':   float(X_r_df['lag1_tourist_log'].values[0]),
        'lag12_tourist_log':  float(X_r_df['lag12_tourist_log'].values[0]),
        'lag1_revenue_log':   float(X_r_df['lag1_revenue_log'].values[0]),
        'lag12_revenue_log':  float(X_r_df['lag12_revenue_log'].values[0]),
        'tourist_growth_12m': float(X_r_df['tourist_growth_12m'].values[0]),
        'revenue_growth_12m': float(X_r_df['revenue_growth_12m'].values[0]),
        'volatility':         float(X_r_df['volatility'].values[0]),
    }
    X_final = pd.DataFrame(
        [[feat_row[f] for f in md['g3_feats']]],
        columns=md['g3_feats'])
    pred  = md['g3_clf'].predict(X_final)[0]
    prob  = md['g3_clf'].predict_proba(X_final)[0]
    label = md['g3_le_sea'].inverse_transform([pred])[0]
    conf  = max(prob)*100
    return label, conf

def calc_3d_risk(season_label, tourist, avg_tourist,
                 tourist_trend, revenue_trend,
                 monthly_revenue, monthly_cost,
                 cash_on_hand, survival_months,
                 monthly_profit, cost_ratio):
    base   = season_config.get(
        season_label, season_config['Normal'])['risk_base']
    t_diff = (tourist-avg_tourist)/max(avg_tourist,1)*100
    t_pen  = (20 if t_diff<-30 else 10 if t_diff<-10
              else -10 if t_diff>20 else 0)
    tr_pen = 10 if tourist_trend=="ลดลง" else -5
    tourism_risk = int(np.clip(base+t_pen+tr_pen, 0, 100))

    if survival_months < 1:    cf_risk = 95
    elif survival_months < 3:  cf_risk = 80
    elif survival_months < 6:  cf_risk = 60
    elif survival_months < 12: cf_risk = 40
    else:                      cf_risk = 20
    if monthly_profit < 0:     cf_risk = min(100, cf_risk+15)
    if cost_ratio > 90:        cf_risk = min(100, cf_risk+10)

    trend_score = season_config.get(
        season_label, season_config['Normal'])['risk_base']
    if tourist_trend=="เพิ่มขึ้น" and revenue_trend=="เพิ่มขึ้น":
        trend_score = max(0,   trend_score-20)
    elif tourist_trend=="เพิ่มขึ้น" or revenue_trend=="เพิ่มขึ้น":
        trend_score = max(0,   trend_score-10)
    elif tourist_trend=="ลดลง" and revenue_trend=="ลดลง":
        trend_score = min(100, trend_score+20)
    else:
        trend_score = min(100, trend_score+10)
    trend_risk = int(np.clip(trend_score, 0, 100))

    overall = int(np.clip(
        tourism_risk*0.35 + cf_risk*0.45 + trend_risk*0.20,
        0, 100))

    def lv(s):
        return "สูง" if s>=70 else "ปานกลาง" if s>=40 else "ต่ำ"

    return {
        'tourism_risk':  tourism_risk,
        'cf_risk':       int(cf_risk),
        'trend_risk':    trend_risk,
        'overall':       overall,
        'tourism_level': lv(tourism_risk),
        'cf_level':      lv(cf_risk),
        'trend_level':   lv(trend_risk),
        'overall_level': lv(overall),
    }

def get_groq_strategy(province, month_name, year, biz_type,
                      usp, pain_points, tourist, avg_tourist,
                      revenue, season_label, risks,
                      monthly_profit, survival_months,
                      monthly_cost, monthly_revenue,
                      tourist_trend, revenue_trend,
                      breakeven_customers, customers_per_day):
    kpi_info  = biz_kpi.get(biz_type, biz_kpi["ร้านอาหาร/คาเฟ่"])
    worst_dim = max(
        [('นักท่องเที่ยว', risks['tourism_risk']),
         ('การเงิน',       risks['cf_risk']),
         ('แนวโน้ม',       risks['trend_risk'])],
        key=lambda x: x[1])
    gap = breakeven_customers - customers_per_day
    prompt = f"""
คุณคือที่ปรึกษา SME ท่องเที่ยวไทย วิเคราะห์เชิงลึกและให้คำแนะนำทำได้จริงทันที

=== ข้อมูลจังหวัด ===
จังหวัด: {province} เดือน: {month_name} {year}
นักท่องเที่ยว: {tourist:,.0f} คน (ปกติ {avg_tourist:,.0f} คน)
รายได้จังหวัด: {revenue/1e9:.2f} พันล้านบาท
สถานการณ์: {season_label}
แนวโน้ม: นักท่องเที่ยว{tourist_trend} รายได้{revenue_trend}

=== ความเสี่ยง 3 มิติ ===
นักท่องเที่ยว: {risks['tourism_risk']}/100 ({risks['tourism_level']})
การเงิน: {risks['cf_risk']}/100 ({risks['cf_level']})
แนวโน้มตลาด: {risks['trend_risk']}/100 ({risks['trend_level']})
รวม: {risks['overall']}/100 — จุดอ่อนสุด: ด้าน{worst_dim[0]}

=== การเงินธุรกิจ ===
รายได้: {monthly_revenue:,.0f} บาท/เดือน
ต้นทุน: {monthly_cost:,.0f} บาท ({monthly_cost/max(monthly_revenue,1)*100:.0f}%)
กำไร/ขาดทุน: {monthly_profit:+,.0f} บาท
เงินสำรองรอดได้: {survival_months:.1f} เดือน
จุดคุ้มทุน: {breakeven_customers:.0f} คน/วัน
ลูกค้าปัจจุบัน: {customers_per_day} คน/วัน
{'ยังขาดอีก: '+str(round(gap,0))+' คน/วัน' if gap>0 else 'เกินจุดคุ้มทุน: '+str(round(abs(gap),0))+' คน/วัน'}

=== ธุรกิจ ===
ประเภท: {biz_type}
KPI: {', '.join(kpi_info['kpi'])}
จุดขาย: {usp or 'ไม่ระบุ'}
ปัญหา: {pain_points or 'ไม่ระบุ'}

ตอบ JSON เท่านั้น ห้าม markdown:
{{
  "summary": "<วิเคราะห์สถานการณ์ 2-3 ประโยค>",
  "survival_warning": "<คำเตือนการเงิน ถ้าวิกฤตบอกชัดเจน>",
  "risk_analysis": {{
    "tourism": "<วิเคราะห์ด้านนักท่องเที่ยว>",
    "cashflow": "<วิเคราะห์สถานะเงินสด>",
    "trend": "<แนวโน้มตลาด 3 เดือนข้างหน้า>"
  }},
  "strategic_recommendations": [
    "<กลยุทธ์ที่ 1 ระบุตัวเลขชัดเจน>",
    "<กลยุทธ์ที่ 2>",
    "<กลยุทธ์ที่ 3>"
  ],
  "immediate_actions_7_days": [
    "<วันที่ 1-2: ทำอะไร ที่ไหน ยังไง>",
    "<วันที่ 3-4: ทำอะไร>",
    "<วันที่ 5-7: ทำอะไร>"
  ],
  "cost_cut_tips": [
    "<ลดต้นทุนอย่างไร ตัวเลขชัดเจน>",
    "<วิธีที่ 2>"
  ],
  "if_then_guide": [
    "<ถ้าลูกค้าน้อยกว่าคาด → ทำ...>",
    "<ถ้าลูกค้ามากกว่าคาด → ทำ...>",
    "<ถ้าเงินสำรองเหลือน้อย → ทำ...>"
  ]
}}
"""
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":prompt}],
            max_tokens=1200, temperature=0.3
        )
        raw = resp.choices[0].message.content.strip()
        if '{' in raw and '}' in raw:
            raw = raw[raw.index('{'):raw.rindex('}')+1]
        return json.loads(raw)
    except json.JSONDecodeError:
        return None
    except Exception as e:
        if "rate_limit" in str(e).lower() or "429" in str(e):
            return {"error":"rate_limit"}
        return None

def build_fallback(season_label, risks, monthly_profit,
                   survival_months, biz_type,
                   tourist, avg_tourist, tourist_trend):
    kpi_info = biz_kpi.get(biz_type, biz_kpi["ร้านอาหาร/คาเฟ่"])
    overall  = risks['overall']
    if overall >= 70:
        recs    = [f"ลดต้นทุน {kpi_info['cost_focus']} ลง 20-25% ทันที",
                   kpi_info['low_action'],
                   "หยุดลงทุนใหม่ทุกอย่าง รักษาเงินสดไว้ก่อน"]
        actions = ["วันที่ 1-2: นับต้นทุนทุกรายการ ตัดที่ไม่จำเป็น",
                   "วันที่ 3-4: Flash Sale ลด 25% เน้นลูกค้าท้องถิ่น",
                   "วันที่ 5-7: คุยกับเจ้าของที่ขอลดค่าเช่าชั่วคราว"]
        cuts    = [f"ลด {kpi_info['cost_focus']} ได้ทันที",
                   "ยกเลิก Subscription รายเดือนที่ไม่ได้ใช้"]
        ifthen  = [f"ถ้าลูกค้าน้อยกว่าคาด → {kpi_info['low_action']}",
                   "ถ้าเงินสำรองน้อยกว่า 2 เดือน → ขอสินเชื่อฉุกเฉิน SME ธ.ออมสิน",
                   "ถ้าขาดทุนติด 3 เดือน → พิจารณาปรับ Business Model"]
        warning = f"เงินสำรองรอดได้แค่ {survival_months:.1f} เดือน ต้องลดต้นทุนทันที!"
        summary = "สถานการณ์วิกฤต ด้านการเงินเสี่ยงสูงสุด ต้องแก้ไขทันที"
    elif overall >= 40:
        recs    = ["จัด Package ร่วมธุรกิจใกล้เคียง เพิ่มมูลค่า 15-20%",
                   kpi_info['low_action'] if tourist < avg_tourist
                   else kpi_info['high_action'],
                   "ทบทวนต้นทุนรายเดือน หาจุดลดโดยไม่กระทบบริการ"]
        actions = ["วันที่ 1-2: ติดต่อธุรกิจข้างเคียงทำ Cross-promotion",
                   "วันที่ 3-4: ส่งโปรโมชันให้ลูกค้าเก่าผ่าน LINE",
                   "วันที่ 5-7: ทดสอบ Upsell 3 รายการที่ Margin ดีสุด"]
        cuts    = [f"ทบทวน {kpi_info['cost_focus']} ลดได้ 10-15%",
                   "สั่งวัตถุดิบร่วมกับร้านใกล้เคียงต่อรองราคา"]
        ifthen  = [f"ถ้าลูกค้าน้อยกว่าคาด → {kpi_info['low_action']}",
                   f"ถ้าลูกค้ามากกว่าคาด → {kpi_info['high_action']}",
                   "ถ้าเงินสำรองน้อยกว่า 6 เดือน → สำรองเงินเพิ่ม 10% ของรายได้"]
        warning = ""
        summary = "สถานการณ์ปานกลาง ต้องติดตามใกล้ชิดและทำการตลาดเชิงรุก"
    else:
        recs    = [kpi_info['high_action'],
                   "ตั้งราคาเพิ่ม 10-15% ช่วงนี้นักท่องเที่ยวมากพอ",
                   "สำรองเงิน 20-30% ของกำไรรับมือ Low Season"]
        actions = ["วันที่ 1-2: ปรับราคาและประกาศใน Social Media",
                   "วันที่ 3-4: เพิ่ม Add-on Service เพิ่มรายได้/Visit",
                   "วันที่ 5-7: เปิด Pre-booking เก็บ Deposit ล่วงหน้า"]
        cuts    = ["ไม่ควรลดต้นทุนตอนนี้ ควรลงทุนเพิ่มแทน",
                   f"สำรองเงินสำหรับ {kpi_info['cost_focus']} ช่วง Low Season"]
        ifthen  = [f"ถ้าลูกค้ามากกว่าคาด → {kpi_info['high_action']}",
                   "ถ้ากำไรดีกว่าคาด → สำรองเงินเพิ่มรับ Low Season",
                   "ถ้านักท่องเที่ยวเริ่มลด → ทำ Package Early Bird ทันที"]
        warning = ""
        summary = "สถานการณ์ดี เป็นช่วงโอกาสทอง ควรลงทุนและขยายบริการ"
    return {
        'summary': summary,
        'survival_warning': warning,
        'risk_analysis': {
            'tourism': f"นักท่องเที่ยว{'มาก' if tourist>avg_tourist else 'น้อย'}กว่าปกติ ส่งผลต่อยอดขายโดยตรง",
            'cashflow': f"เงินสำรองรอดได้ {survival_months:.1f} เดือน {'ต้องระวัง' if survival_months<6 else 'พอรับได้'}",
            'trend': f"แนวโน้มตลาด{tourist_trend} ควรวางแผนล่วงหน้า",
        },
        'strategic_recommendations': recs,
        'immediate_actions_7_days':  actions,
        'cost_cut_tips':             cuts,
        'if_then_guide':             ifthen,
    }

# ════════════════════════════════════════════════
# UI Header
# ════════════════════════════════════════════════
st.title("🚨 SME Early Warning System")
st.caption(
    "ระบบเตือนภัยล่วงหน้าสำหรับ SME ท่องเที่ยว · "
    "G1→G2→G3 RF 93% · Risk Score 3 มิติ · Groq LLM")

st.markdown("""
<div style='display:flex;gap:8px;align-items:center;margin:16px 0'>
    <div style='flex:1;padding:10px;background:#f0fdf4;
    border-radius:8px;border:1px solid #22c55e;
    text-align:center;font-size:13px;font-weight:bold'>
    🧭 G1<br><span style='font-size:11px;font-weight:normal'>
    นักท่องเที่ยว</span></div>
    <div style='font-size:18px;color:#94a3b8'>→</div>
    <div style='flex:1;padding:10px;background:#eff6ff;
    border-radius:8px;border:1px solid #3b82f6;
    text-align:center;font-size:13px;font-weight:bold'>
    💰 G2<br><span style='font-size:11px;font-weight:normal'>
    รายได้จังหวัด</span></div>
    <div style='font-size:18px;color:#94a3b8'>→</div>
    <div style='flex:1;padding:10px;background:#fffbeb;
    border-radius:8px;border:1px solid #f59e0b;
    text-align:center;font-size:13px;font-weight:bold'>
    🤖 G3<br><span style='font-size:11px;font-weight:normal'>
    Season Classifier</span></div>
    <div style='font-size:18px;color:#94a3b8'>→</div>
    <div style='flex:1;padding:10px;background:#fff7ed;
    border-radius:8px;border:1px solid #f97316;
    text-align:center;font-size:13px;font-weight:bold'>
    💸 Cashflow<br><span style='font-size:11px;font-weight:normal'>
    3 มิติ</span></div>
    <div style='font-size:18px;color:#94a3b8'>→</div>
    <div style='flex:1;padding:10px;background:#fef2f2;
    border-radius:8px;border:1px solid #ef4444;
    text-align:center;font-size:13px;font-weight:bold'>
    📋 แผน<br><span style='font-size:11px;font-weight:normal'>
    กู้ธุรกิจ</span></div>
</div>
""", unsafe_allow_html=True)
st.divider()

# ── Sidebar ───────────────────────────────────────
st.sidebar.header("📋 ข้อมูลธุรกิจ")
provinces = md['g1_le'].classes_.tolist()
province  = st.sidebar.selectbox("📍 จังหวัด", provinces)
biz_type  = st.sidebar.selectbox("🏪 ประเภทธุรกิจ", biz_types)

st.sidebar.markdown("**💬 บอกเราเพิ่มเติม**")
usp = st.sidebar.text_input("จุดขายหลัก",
    placeholder="เช่น วิวดี บริการเยี่ยม ราคาถูก")
pain_points = st.sidebar.text_input("ปัญหาที่เจออยู่",
    placeholder="เช่น ลูกค้าน้อย ต้นทุนสูง")

st.sidebar.divider()
st.sidebar.markdown("**💵 สถานะการเงินตอนนี้**")
monthly_revenue = st.sidebar.number_input(
    "รายได้ต่อเดือน (บาท)",
    min_value=0, max_value=10000000,
    value=80000, step=5000, format="%d")
monthly_cost = st.sidebar.number_input(
    "ค่าใช้จ่ายต่อเดือน (บาท)",
    min_value=0, max_value=10000000,
    value=70000, step=5000, format="%d")
cash_on_hand = st.sidebar.number_input(
    "เงินสดสำรอง (บาท)",
    min_value=0, max_value=50000000,
    value=150000, step=10000, format="%d")

st.sidebar.divider()
st.sidebar.markdown("**🧮 ข้อมูลสำหรับคำนวณจุดคุ้มทุน**")
customers_per_day = st.sidebar.number_input(
    "ลูกค้าเฉลี่ยต่อวัน (คน)",
    min_value=0, max_value=10000,
    value=30, step=1, format="%d")
avg_spend_per_customer = st.sidebar.number_input(
    "ยอดใช้จ่ายเฉลี่ยต่อคน (บาท)",
    min_value=0, max_value=100000,
    value=300, step=50, format="%d")

st.sidebar.divider()
horizon_label = st.sidebar.selectbox(
    "⏰ พยากรณ์ล่วงหน้า",
    list(horizon_options.keys()))
months_ahead = horizon_options[horizon_label]
now          = datetime.datetime.now()
target_date  = now + pd.DateOffset(months=months_ahead)
month        = target_date.month
year         = target_date.year

st.sidebar.info(
    f"**📅 {months_full[month-1]} {year+543}**\n\n"
    f"{'✅ แม่นยำสูง' if months_ahead<=6 else '⚠️ ระยะยาว'}")

predict_btn = st.sidebar.button(
    "🚨 วิเคราะห์ความเสี่ยงเชิงลึก",
    use_container_width=True, type="primary")

# ════════════════════════════════════════════════
# Main Analysis
# ════════════════════════════════════════════════
if predict_btn:
    g1_penc = md['g1_le'].transform([province])[0]
    g2_penc = md['g2_le'].transform([province])[0]
    tier    = get_tier(province)
    ma      = max(months_ahead, 1)

    l1t, l12t = get_avg_lag_t(province, month)
    l1r, l12r = get_avg_lag_r(province, month)
    if l1t is None or l1r is None:
        st.error("ไม่พบข้อมูลของจังหวัดนี้")
        st.stop()

    with st.spinner("🔄 กำลังวิเคราะห์ข้อมูล..."):
        tourist = predict_g1(g1_penc, month, year,
                             l1t, l12t, tier, ma)
        revenue = predict_g2(g2_penc, month, year,
                             l1r, l12r, tier, ma)
        season_label, conf = predict_g3(
            province, month, year,
            l1t, l12t, l1r, l12r, tourist, revenue)

    cfg = season_config.get(season_label, season_config['Normal'])

    monthly_profit  = monthly_revenue - monthly_cost
    cost_ratio      = monthly_cost / max(monthly_revenue,1) * 100
    survival_months = (cash_on_hand / max(abs(monthly_profit),1)
                       if monthly_profit < 0 else 99)

    row_avg     = md['g1_avg'][
        (md['g1_avg']['province_thai']==province)&
        (md['g1_avg']['month']==month)]
    avg_tourist = (row_avg['avg_tourist'].values[0]
                   if len(row_avg)>0 else tourist)

    tourist_trend = "เพิ่มขึ้น" if l1t > l12t else "ลดลง"
    revenue_trend = "เพิ่มขึ้น" if l1r > l12r else "ลดลง"

    risks = calc_3d_risk(
        season_label, tourist, avg_tourist,
        tourist_trend, revenue_trend,
        monthly_revenue, monthly_cost,
        cash_on_hand, survival_months,
        monthly_profit, cost_ratio)

    overall_color = ('#ef4444' if risks['overall'] >= 70
                     else '#f59e0b' if risks['overall'] >= 40
                     else '#22c55e')

    # Break-even คำนวณ
    daily_cost            = monthly_cost / 30
    daily_revenue_now     = customers_per_day * avg_spend_per_customer
    breakeven_customers   = (daily_cost / avg_spend_per_customer
                             if avg_spend_per_customer > 0 else 0)

    # ── Early Warning Banner ──────────────────────
    if risks['overall'] >= 70:
        st.error(
            f"🚨 **ธุรกิจอยู่ในโซนเสี่ยงสูง!** "
            f"คะแนนรวม {risks['overall']}/100 — "
            f"{'เงินสำรองรอดได้แค่ '+str(round(survival_months,1))+' เดือน ต้องลดต้นทุนทันที!' if survival_months < 6 else 'สถานการณ์ท่องเที่ยวไม่เอื้ออำนวย ควรระวังการลงทุน'}")
    elif risks['overall'] >= 40:
        worst = max([
            ('นักท่องเที่ยวน้อยกว่าปกติ', risks['tourism_risk']),
            ('การเงินตึงตัว ควรเพิ่มเงินสำรอง', risks['cf_risk']),
            ('แนวโน้มตลาดเริ่มชะลอตัว', risks['trend_risk']),
        ], key=lambda x: x[1])
        st.warning(
            f"⚠️ **ควรระวัง — {worst[0]}** "
            f"คะแนนรวม {risks['overall']}/100 "
            f"ดูคำแนะนำด้านล่างเพื่อรับมือ")
    else:
        st.success(
            f"✅ **สถานการณ์ดี** คะแนนรวม {risks['overall']}/100 — "
            f"นักท่องเที่ยวอยู่ในเกณฑ์ดี เหมาะลงทุนและขยายบริการ")

    st.subheader(
        f"📍 {province} — {months_full[month-1]} {year+543} "
        f"· {cfg['emoji']} {cfg['label']}")

    # ── Risk Score 3 มิติ ─────────────────────────
    st.subheader("🎯 ความเสี่ยง 3 มิติ")
    r1,r2,r3,r4 = st.columns(4)
    dim_colors = {
        'tourism': ('#ef4444' if risks['tourism_risk']>=70
                    else '#f59e0b' if risks['tourism_risk']>=40
                    else '#22c55e'),
        'cf':      ('#ef4444' if risks['cf_risk']>=70
                    else '#f59e0b' if risks['cf_risk']>=40
                    else '#22c55e'),
        'trend':   ('#ef4444' if risks['trend_risk']>=70
                    else '#f59e0b' if risks['trend_risk']>=40
                    else '#22c55e'),
    }
    risk_card(r1,"🧭 ความเสี่ยงด้านนักท่องเที่ยว",
              risks['tourism_risk'],risks['tourism_level'],
              dim_colors['tourism'])
    risk_card(r2,"💸 ความเสี่ยงด้านการเงิน",
              risks['cf_risk'],risks['cf_level'],
              dim_colors['cf'])
    risk_card(r3,"📈 ความเสี่ยงด้านแนวโน้มตลาด",
              risks['trend_risk'],risks['trend_level'],
              dim_colors['trend'])
    risk_card(r4,"⚡ ความเสี่ยงรวม",
              risks['overall'],risks['overall_level'],
              overall_color)

    pb1,pb2,pb3 = st.columns(3)
    for col, label, score in [
        (pb1,"🧭 ด้านนักท่องเที่ยว",   risks['tourism_risk']),
        (pb2,"💸 ด้านการเงิน",          risks['cf_risk']),
        (pb3,"📈 ด้านแนวโน้มตลาด",     risks['trend_risk']),
    ]:
        with col:
            st.markdown(f"**{label}**: {score}/100")
            st.progress(score/100)

    st.divider()

    # ── Cashflow Detail ───────────────────────────
    st.subheader("💸 รายละเอียดการเงิน")
    cf1,cf2,cf3,cf4 = st.columns(4)

    sv       = (f"{survival_months:.1f} เดือน"
                if survival_months < 99 else "มั่นคง ✅")
    sv_color = ('#ef4444' if survival_months < 3
                else '#f97316' if survival_months < 6
                else '#22c55e')
    pf_color = '#22c55e' if monthly_profit >= 0 else '#ef4444'

    small_card(cf1,"รายได้/เดือน",
               f"{monthly_revenue:,.0f} บาท","")
    small_card(cf2,"ต้นทุน/เดือน",
               f"{monthly_cost:,.0f} บาท",
               f"คิดเป็น {cost_ratio:.0f}% ของรายได้")
    small_card(cf3,"กำไร/ขาดทุน",
               f"{monthly_profit:+,.0f} บาท",
               "มีกำไร ✅" if monthly_profit>=0 else "ขาดทุน 🚨",
               pf_color)
    small_card(cf4,"เงินสำรองรอดได้",
               sv, f"เงินสด {cash_on_hand:,.0f} บาท",
               sv_color)

    if monthly_profit < 0 and survival_months < 3:
        st.error(
            f"🚨 **วิกฤต!** ขาดทุน {abs(monthly_profit):,.0f} บาท/เดือน "
            f"เงินจะหมดใน **{survival_months:.1f} เดือน** ต้องแก้ไขทันที!")
    elif monthly_profit < 0:
        st.warning(
            f"⚠️ ขาดทุน {abs(monthly_profit):,.0f} บาท/เดือน "
            f"เงินสำรองรอดได้ {survival_months:.1f} เดือน")
    elif cost_ratio > 90:
        st.warning(
            f"⚠️ ต้นทุนสูง {cost_ratio:.0f}% ของรายได้ ควรลดต้นทุน")

    st.divider()

    # ── Break-even Calculator ─────────────────────
    st.subheader("🧮 จุดคุ้มทุนของธุรกิจคุณ")
    st.caption("คำนวณว่าต้องมีลูกค้ากี่คน/วัน ถึงจะไม่ขาดทุน")

    gap_customers = breakeven_customers - customers_per_day
    be_color      = '#22c55e' if gap_customers <= 0 else '#ef4444'

    be1,be2,be3 = st.columns(3)
    small_card(be1,
        "🎯 ลูกค้าที่ต้องมี/วัน",
        f"{breakeven_customers:.0f} คน",
        "เพื่อไม่ให้ขาดทุน", be_color)
    small_card(be2,
        "👥 ลูกค้าตอนนี้",
        f"{customers_per_day} คน/วัน",
        f"รายได้ {daily_revenue_now:,.0f} บาท/วัน",
        '#22c55e' if gap_customers <= 0 else '#f97316')
    small_card(be3,
        "📊 ยังขาดหรือเกิน",
        f"{abs(gap_customers):.0f} คน/วัน",
        "✅ เกินจุดคุ้มทุน" if gap_customers<=0
        else "⚠️ ยังไม่ถึงจุดคุ้มทุน",
        '#22c55e' if gap_customers <= 0 else '#ef4444')

    # Scenario
    st.markdown("**📉 ถ้านักท่องเที่ยวลดลงจะเกิดอะไรขึ้น**")
    sc1,sc2,sc3 = st.columns(3)
    for col, pct, label in [
        (sc1, 10, "ลด 10%"),
        (sc2, 20, "ลด 20%"),
        (sc3, 30, "ลด 30%"),
    ]:
        rc = customers_per_day * (1-pct/100)
        rr = rc * avg_spend_per_customer
        rp = rr - daily_cost
        small_card(col,
            f"🔻 นักท่องเที่ยว{label}",
            f"{rc:.0f} คน/วัน",
            f"{'กำไร' if rp>=0 else 'ขาดทุน'} "
            f"{abs(rp):,.0f} บาท/วัน",
            '#22c55e' if rp>=0 else '#ef4444')

    if gap_customers > 0:
        st.error(
            f"🚨 ตอนนี้ยังไม่ถึงจุดคุ้มทุน "
            f"ต้องเพิ่มลูกค้าอีก **{gap_customers:.0f} คน/วัน** "
            f"หรือลดต้นทุนลง **{daily_cost-daily_revenue_now:,.0f} บาท/วัน**")
    elif gap_customers > -5:
        st.warning(
            f"⚠️ เกินจุดคุ้มทุนแค่ {abs(gap_customers):.0f} คน/วัน "
            f"ถ้าลูกค้าลดลงนิดเดียวจะขาดทุนทันที")
    else:
        st.success(
            f"✅ เกินจุดคุ้มทุน {abs(gap_customers):.0f} คน/วัน "
            f"มีกันชนพอสมควร")

    # กราฟ Break-even
    fig_be, ax_be = plt.subplots(figsize=(10, 3))
    max_c   = max(breakeven_customers*2, customers_per_day*1.5, 10)
    cust_range = np.arange(0, max_c+1, 1)
    rev_line   = cust_range * avg_spend_per_customer
    cost_line  = np.full_like(cust_range, daily_cost)

    ax_be.plot(cust_range, rev_line/1000,
               color='#22c55e', linewidth=2, label='รายได้')
    ax_be.axhline(daily_cost/1000, color='#ef4444',
                  linewidth=2, linestyle='--', label='ต้นทุน/วัน')
    ax_be.axvline(breakeven_customers, color='#f59e0b',
                  linewidth=2, linestyle='--',
                  label=f'จุดคุ้มทุน {breakeven_customers:.0f} คน')
    ax_be.axvline(customers_per_day, color='#3b82f6',
                  linewidth=2,
                  label=f'ลูกค้าปัจจุบัน {customers_per_day} คน')
    ax_be.fill_between(cust_range, rev_line/1000, daily_cost/1000,
                       where=rev_line >= cost_line,
                       alpha=0.15, color='#22c55e', label='โซนกำไร')
    ax_be.fill_between(cust_range, rev_line/1000, daily_cost/1000,
                       where=rev_line < cost_line,
                       alpha=0.15, color='#ef4444', label='โซนขาดทุน')
    ax_be.set_xlabel('จำนวนลูกค้า (คน/วัน)')
    ax_be.set_ylabel('บาท (พัน)')
    ax_be.set_title('กราฟจุดคุ้มทุน (Break-even Analysis)')
    ax_be.legend(fontsize=8, loc='upper left')
    ax_be.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_be)

    st.divider()

    # ── KPI + G1/G2 Summary ───────────────────────
    kpi_info = biz_kpi.get(biz_type, biz_kpi["ร้านอาหาร/คาเฟ่"])
    st.subheader(f"📊 KPI หลักสำหรับ {biz_type}")
    kpi_cols = st.columns(len(kpi_info['kpi']))
    for col, kpi in zip(kpi_cols, kpi_info['kpi']):
        with col:
            st.markdown(
                f"<div style='padding:8px;background:#f1f5f9;"
                f"border-radius:6px;text-align:center;"
                f"font-size:12px'><b>📌 {kpi}</b></div>",
                unsafe_allow_html=True)

    st.markdown("")
    m1,m2,m3 = st.columns(3)
    diff = (tourist-avg_tourist)/max(avg_tourist,1)*100
    small_card(m1,"🧭 นักท่องเที่ยวคาดการณ์",
               f"{tourist:,.0f} คน",
               f"{'มากกว่า' if diff>0 else 'น้อยกว่า'}ปกติ {abs(diff):.0f}%")
    small_card(m2,"💰 รายได้ท่องเที่ยวจังหวัด",
               f"{revenue/1e9:.2f} พันล้านบาท","")
    small_card(m3,f"🤖 สถานการณ์",
               f"{cfg['emoji']} {cfg['label']}",
               f"ความเชื่อมั่น {conf:.0f}%")

    st.divider()

    # ── AI Strategy ───────────────────────────────
    st.subheader("🤖 แผนกลยุทธ์เชิงลึก")
    st.caption(
        f"วิเคราะห์จาก Risk 3 มิติ + Cashflow + "
        f"Break-even + KPI เฉพาะ{biz_type}")

    with st.spinner("AI กำลังวิเคราะห์เชิงลึก..."):
        result = get_groq_strategy(
            province, months_full[month-1], year,
            biz_type, usp, pain_points,
            tourist, avg_tourist, revenue,
            season_label, risks,
            monthly_profit, survival_months,
            monthly_cost, monthly_revenue,
            tourist_trend, revenue_trend,
            breakeven_customers, customers_per_day)

    if result is None:
        result = build_fallback(
            season_label, risks, monthly_profit,
            survival_months, biz_type,
            tourist, avg_tourist, tourist_trend)
    elif "error" in result:
        st.warning("⏳ AI เกินโควต้า ใช้คำแนะนำสำรองแทน")
        result = build_fallback(
            season_label, risks, monthly_profit,
            survival_months, biz_type,
            tourist, avg_tourist, tourist_trend)

    if risks['overall'] < 40:
        st.success(result.get('summary',''))
    elif risks['overall'] < 70:
        st.warning(result.get('summary',''))
    else:
        st.error(result.get('summary',''))

    if result.get('survival_warning'):
        st.error(f"🚨 {result['survival_warning']}")

    ra = result.get('risk_analysis', {})
    ra1,ra2,ra3 = st.columns(3)
    with ra1:
        st.info(
            f"**🧭 ด้านนักท่องเที่ยว**\n\n{ra.get('tourism','')}")
    with ra2:
        fn = (st.error if risks['cf_risk']>=70
              else st.warning if risks['cf_risk']>=40
              else st.success)
        fn(f"**💸 ด้านการเงิน**\n\n{ra.get('cashflow','')}")
    with ra3:
        st.info(
            f"**📈 ด้านแนวโน้มตลาด**\n\n{ra.get('trend','')}")

    st.divider()

    col_s, col_a = st.columns(2)
    with col_s:
        st.markdown("**💡 กลยุทธ์เชิงลึก**")
        for i, s in enumerate(
                result.get('strategic_recommendations',[]),1):
            st.success(f"**{i}.** {s}")
        st.markdown("**✂️ ลดต้นทุน**")
        for i, c in enumerate(result.get('cost_cut_tips',[]),1):
            st.info(f"**{i}.** {c}")
    with col_a:
        st.markdown("**⚡ Action 7 วัน**")
        for a in result.get('immediate_actions_7_days',[]):
            st.warning(f"**{a}**")
        st.markdown("**🔀 If-Then Guide**")
        for ift in result.get('if_then_guide',[]):
            st.markdown(f"→ {ift}")

    st.divider()

    # ── กราฟ 12 เดือน ────────────────────────────
    st.subheader(f"📈 แนวโน้ม 12 เดือน ปี {year+543}")

    monthly_t, monthly_r, monthly_s = [], [], []
    for m in range(1,13):
        lt,l12t_ = get_avg_lag_t(province, m)
        lr,l12r_ = get_avg_lag_r(province, m)
        if lt is None:
            monthly_t.append(0); monthly_r.append(0)
            monthly_s.append('Normal'); continue
        t = predict_g1(g1_penc, m, year, lt, l12t_, tier, ma)
        r = predict_g2(g2_penc, m, year,
                       lr or 0, l12r_ or 0, tier, ma)
        s, _ = predict_g3(province, m, year,
                          lt, l12t_, lr or 0, l12r_ or 0, t, r)
        monthly_t.append(t); monthly_r.append(r)
        monthly_s.append(s)

    bar_colors_map = {
        'Golden Opportunity':'#eab308','Normal':'#22c55e',
        'Mixed':'#3b82f6','Survival':'#f97316',
        'Critical Risk':'#ef4444',
    }
    colors = [bar_colors_map.get(s,'#93c5fd') for s in monthly_s]

    fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(12,7))
    bars1 = ax1.bar(months_th,
                    [t/1000 for t in monthly_t],
                    color=colors, alpha=0.85)
    for bar,val in zip(bars1,monthly_t):
        ax1.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+max(monthly_t or [1])*0.01/1000,
                 f'{val/1000:.0f}K', ha='center', fontsize=8)
    bars1[month-1].set_edgecolor('black')
    bars1[month-1].set_linewidth(3)
    ax1.set_title(
        f'นักท่องเที่ยว {province} ปี {year+543} (พันคน)')
    ax1.set_ylabel('พันคน')
    ax1.grid(True, alpha=0.3, axis='y')

    bars2 = ax2.bar(months_th,
                    [r/1e9 for r in monthly_r],
                    color=colors, alpha=0.85)
    for bar,val in zip(bars2,monthly_r):
        ax2.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+max(monthly_r or [1])*0.01/1e9,
                 f'{val/1e9:.1f}B', ha='center', fontsize=8)
    bars2[month-1].set_edgecolor('black')
    bars2[month-1].set_linewidth(3)
    ax2.set_title(
        f'รายได้ท่องเที่ยว {province} ปี {year+543} (พันล้านบาท)')
    ax2.set_ylabel('พันล้านบาท')
    ax2.grid(True, alpha=0.3, axis='y')

    from matplotlib.patches import Patch
    legend_els = [
        Patch(color=bar_colors_map[s],
              label=f"{season_config[s]['emoji']} "
                    f"{season_config[s]['label']}")
        for s in bar_colors_map]
    ax1.legend(handles=legend_els, loc='upper right',
               fontsize=8, ncol=3)
    plt.suptitle('กรอบดำ = เดือนที่วิเคราะห์', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)

    # Season Timeline
    st.subheader("🗓️ สถานการณ์รายเดือน")
    cols_cal = st.columns(12)
    for i, (s,t,r) in enumerate(
            zip(monthly_s,monthly_t,monthly_r)):
        cfg_m = season_config.get(s, season_config['Normal'])
        with cols_cal[i]:
            border = "3px solid black" if i==month-1 \
                     else f"2px solid {cfg_m['border']}"
            st.markdown(
                f"<div style='background:{cfg_m['color']};"
                f"border:{border};border-radius:8px;"
                f"padding:5px;text-align:center;font-size:10px'>"
                f"<b>{months_th[i]}</b><br>{cfg_m['emoji']}<br>"
                f"{t/1000:.0f}K<br>{r/1e9:.1f}B"
                f"</div>", unsafe_allow_html=True)

    st.divider()

    # ── Model Performance ─────────────────────────
    st.subheader("📊 ประสิทธิภาพระบบ")
    metrics = md['metrics']
    p1,p2,p3,p4 = st.columns(4)

    small_card(p1,
        "🧭 ความแม่นยำพยากรณ์นักท่องเที่ยว",
        f"MAPE {metrics['g1_short_mape']}%",
        f"RMSLE {metrics['g1_short_rmsle']} | Corr {metrics['g1_short_corr']}")
    small_card(p2,
        "💰 ความแม่นยำพยากรณ์รายได้",
        f"MAPE {metrics['g2_short_mape']}%",
        f"RMSLE {metrics['g2_short_rmsle']}")
    small_card(p3,
        "🤖 ความแม่นยำวิเคราะห์สถานการณ์",
        f"{metrics['g3_rf_acc']}%",
        "ทดสอบจากข้อมูลจริง 2025-2026")
    small_card(p4,
        "✅ ระบบผ่านการทดสอบ",
        "8 โมเดล",
        "AI เลือกโมเดลที่ดีที่สุดอัตโนมัติ")
