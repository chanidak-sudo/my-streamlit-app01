import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import joblib
import os
import urllib.request
import datetime
from groq import Groq

st.set_page_config(
    page_title="พยากรณ์รายได้ท่องเที่ยว",
    page_icon="💰",
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
def load_models():
    return {
        'short_models': joblib.load("ensemble_short_models_g2.pkl"),
        'long_models':  joblib.load("ensemble_long_models_g2.pkl"),
        'scaler_short': joblib.load("scaler_g2_short.pkl"),
        'scaler_long':  joblib.load("scaler_g2_long.pkl"),
        'f_short':      joblib.load("features_g2_short.pkl"),
        'f_long':       joblib.load("features_g2_long.pkl"),
        'le':           joblib.load("label_encoder_g2.pkl"),
        'w_short':      joblib.load("weights_short_g2.pkl"),
        'w_long':       joblib.load("weights_long_g2.pkl"),
        'avg_df':       joblib.load("avg_revenue_by_province_month.pkl"),
    }

md = load_models()

GROQ_API_KEY = "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"  # ← ใส่ Key จริงครับ
groq_client  = Groq(api_key=GROQ_API_KEY)

months_th   = ["ม.ค.","ก.พ.","มี.ค.","เม.ย.","พ.ค.","มิ.ย.",
               "ก.ค.","ส.ค.","ก.ย.","ต.ค.","พ.ย.","ธ.ค."]
months_full = ["มกราคม","กุมภาพันธ์","มีนาคม","เมษายน",
               "พฤษภาคม","มิถุนายน","กรกฎาคม","สิงหาคม",
               "กันยายน","ตุลาคม","พฤศจิกายน","ธันวาคม"]
biz_types   = ["ที่พัก/โรงแรม","ร้านอาหาร","ทัวร์/นำเที่ยว",
               "ของที่ระลึก","สปา/นวด"]
# จุดที่ 1 — แก้ tier1
tier1_provs = ['กรุงเทพมหานคร','ภูเก็ต','เชียงใหม่',
               'ชลบุรี','กาญจนบุรี']
horizon_options = {
    "3 เดือนข้างหน้า":  3,
    "6 เดือนข้างหน้า":  6,
    "1 ปีข้างหน้า":    12,
    "2 ปีข้างหน้า":    24,
    "3 ปีข้างหน้า":    36,
    "4 ปีข้างหน้า":    48,
    "5 ปีข้างหน้า":    60,
}
# MAPE แยก Tier จากผลจริง
mape_table = {
    'short': {1: 11.7, 2: 13.0, 3: 14.8},
    'long':  {1: 42.0, 2: 96.8, 3: 84.5},
}

# จุดที่ 2 — แก้ฟังก์ชัน get_tier
def get_tier(province):
    if province in tier1_provs: return 1
    avg = md['avg_df'][
        md['avg_df']['province_thai']==province]\
        ['avg_revenue'].mean()
    # G2 ใช้ threshold รายได้ ≥ 1 พันล้านบาท
    return 2 if avg >= 1e9 else 3

def get_avg_lag(province, month):
    prev = month-1 if month>1 else 12
    c = md['avg_df'][(md['avg_df']['province_thai']==province)&
                     (md['avg_df']['month']==month)]
    p = md['avg_df'][(md['avg_df']['province_thai']==province)&
                     (md['avg_df']['month']==prev)]
    if len(c)==0 or len(p)==0: return None, None
    return p['avg_revenue'].values[0], c['avg_revenue'].values[0]

def build_input(province_enc, month, year,
                lag_1, lag_12, tier, features):
    is_covid     = 1 if year in [2020,2021] else 0
    moving_avg   = (lag_1+lag_12)/2
    growth_rate  = np.clip((lag_1-lag_12)/max(lag_12,1),-1,10)
    row = {
        'province_enc':   province_enc,
        'month':          month,
        'year':           year,
        'is_covid':       is_covid,
        'tier':           tier,
        'month_sin':      np.sin(2*np.pi*month/12),
        'month_cos':      np.cos(2*np.pi*month/12),
        'lag_1_log':      np.log1p(lag_1),
        'lag_12_log':     np.log1p(lag_12),
        'moving_avg_log': np.log1p(moving_avg),
        'growth_rate_12': growth_rate,
    }
    return pd.DataFrame([{f: row[f] for f in features}])

def predict_ensemble(models_dict, X_scaled, weights):
    preds = [np.maximum(np.expm1(m.predict(X_scaled)),0)
             for m in models_dict.values()]
    w = [weights['rf'], weights['xgb']]
    return float(sum(p*wi for p,wi in zip(preds,w))[0])

def smart_predict(province_enc, month, year,
                  lag_1, lag_12, tier, months_ahead):
    is_short = months_ahead <= 6
    feats    = md['f_short']      if is_short else md['f_long']
    scaler   = md['scaler_short'] if is_short else md['scaler_long']
    models   = md['short_models'] if is_short else md['long_models']
    weights  = md['w_short']      if is_short else md['w_long']
    X_raw    = build_input(province_enc, month, year,
                           lag_1, lag_12, tier, feats)
    X_sc     = scaler.transform(X_raw)
    result   = predict_ensemble(models, X_sc, weights)
    return result, is_short

def get_ai_advice(province, month_name, year, biz_type,
                  revenue, avg_val, mape_pct, season_label,
                  low_est, high_est, is_short):
    try:
        range_type = "ระยะสั้น (แม่นยำสูง)" if is_short \
                     else "ระยะยาว (ใช้ดู Trend)"
        prompt = f"""
คุณเป็นที่ปรึกษาธุรกิจท่องเที่ยวสำหรับ SME ไทย

ข้อมูลพื้นที่:
- จังหวัด: {province}
- เดือน: {month_name} {year}
- ฤดูกาล: {season_label}
- ประเภทธุรกิจ: {biz_type}

ข้อมูลการพยากรณ์ ({range_type}):
- รายได้รวมที่คาดการณ์: {revenue/1e9:.2f} พันล้านบาท
- ช่วงที่เป็นไปได้: {low_est/1e9:.2f} ถึง {high_est/1e9:.2f} พันล้านบาท
- รายได้ปกติ: {avg_val/1e9:.2f} พันล้านบาท
- มากกว่า/น้อยกว่าปกติ: {((revenue-avg_val)/avg_val*100):+.1f}%

ให้คำแนะนำ 4 ข้อ สำหรับธุรกิจ{biz_type}:
1. ควรปรับกลยุทธ์ราคาหรือโปรโมชันอย่างไร
2. ควรเตรียมพนักงานและสต็อกเท่าไหร่
3. ควรลงทุนเพิ่มหรือชะลอการลงทุน
4. ความเสี่ยงที่ควรระวัง

ตอบภาษาไทย ไม่ใช้คำเทคนิค ตัวเลขชัดเจน แต่ละข้อ 1-2 ประโยค
"""
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":prompt}],
            max_tokens=600, temperature=0.5
        )
        return resp.choices[0].message.content

    except Exception as e:
        err_msg = str(e)
        # Rate Limit
        if "rate_limit_exceeded" in err_msg or "429" in err_msg:
            return (
                "⏳ **ขณะนี้ระบบ AI ใช้งานเกินโควต้ารายวัน**\n\n"
                "กรุณาลองใหม่ในวันพรุ่งนี้ หรือใช้คำแนะนำทั่วไปด้านล่างแทน\n\n"
                f"**คำแนะนำเบื้องต้นสำหรับ{season_label}:**\n"
                + _fallback_advice(season_label, biz_type,
                                   revenue, avg_val)
            )
        # Error อื่น
        return (
            f"⚠️ ไม่สามารถเชื่อมต่อ AI ได้ ({str(e)[:50]})\n\n"
            + _fallback_advice(season_label, biz_type,
                               revenue, avg_val)
        )

def _fallback_advice(season_label, biz_type, revenue, avg_val):
    """คำแนะนำสำรองเมื่อ AI ไม่พร้อมใช้งาน"""
    diff = (revenue - avg_val) / avg_val * 100

    if "ไฮ" in season_label or "สูงสุด" in season_label:
        return (
            "1. **ราคา:** ตั้งราคาเต็มได้เลย ไม่จำเป็นต้องลดราคา\n"
            "2. **พนักงาน:** เพิ่มพนักงานชั่วคราว 20-30% สั่งสต็อกล่วงหน้า 30-40%\n"
            "3. **ลงทุน:** เหมาะกับการลงทุนเพิ่มบริการ Premium\n"
            "4. **ความเสี่ยง:** ระวังสต็อกไม่พอ และการบริการไม่ทัน"
        )
    elif "ปานกลาง" in season_label:
        return (
            "1. **ราคา:** จัด Early Bird หรือโปร Weekday ลด 10-15%\n"
            "2. **พนักงาน:** คงพนักงานเดิม สต็อกเพิ่มเล็กน้อย 10-15%\n"
            "3. **ลงทุน:** ลงทุนด้านการตลาดออนไลน์เพื่อดึงลูกค้า\n"
            "4. **ความเสี่ยง:** ระวังคู่แข่งจัดโปรโมชันตัดราคา"
        )
    else:
        return (
            "1. **ราคา:** Flash Sale ลด 20-30% ดึงลูกค้าท้องถิ่น\n"
            "2. **พนักงาน:** ลด Shift พนักงาน สั่งสต็อกน้อยลง 20-30%\n"
            "3. **ลงทุน:** ชะลอการลงทุนใหม่ ใช้เวลาซ่อมบำรุงแทน\n"
            "4. **ความเสี่ยง:** ระวังค่าใช้จ่ายคงที่สูงเกินรายได้"
        )

# ═══════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════
st.title("💰 ระบบพยากรณ์รายได้ท่องเที่ยว")
st.caption("สำหรับผู้ประกอบการ SME — ดูแนวโน้มรายได้ท่องเที่ยวเพื่อวางแผนธุรกิจ")
st.divider()

# ── Sidebar ──────────────────────────────────────
st.sidebar.header("📋 ข้อมูลธุรกิจของคุณ")

provinces     = md['le'].classes_.tolist()
province      = st.sidebar.selectbox("📍 จังหวัด", provinces)
biz_type      = st.sidebar.selectbox("🏪 ประเภทธุรกิจ", biz_types)
horizon_label = st.sidebar.selectbox(
    "⏰ ต้องการพยากรณ์ล่วงหน้าเท่าไหร่",
    list(horizon_options.keys()))

months_ahead = horizon_options[horizon_label]
now          = datetime.datetime.now()
target_date  = now + pd.DateOffset(months=months_ahead)
month        = target_date.month
year         = target_date.year
is_short_ui  = months_ahead <= 6

# แสดงเดือน/ปีที่จะพยากรณ์
st.sidebar.divider()
st.sidebar.markdown("**📅 ระบบจะพยากรณ์ให้คุณ:**")
st.sidebar.info(
    f"**{months_full[month-1]} {year+543}** (ค.ศ. {year})\n\n"
    f"{'✅ ความแม่นยำสูง' if is_short_ui else '⚠️ ประมาณการณ์ระยะยาว'}")

# ค่าเฉลี่ยรายได้ปกติ
st.sidebar.markdown("**💡 รายได้ปกติของจังหวัดนี้**")
st.sidebar.caption("รายได้รวมการท่องเที่ยวของจังหวัด")
for m in [1,4,7,10,12]:
    row = md['avg_df'][(md['avg_df']['province_thai']==province)&
                       (md['avg_df']['month']==m)]
    if len(row)>0:
        val = row['avg_revenue'].values[0]
        s   = "🔥" if m in [11,12,1] else \
              "🌤" if m in [2,3,10]  else "🌧"
        st.sidebar.markdown(
            f"- **{months_th[m-1]}**: {val/1e9:.1f} พันล้านบาท {s}")

predict_btn = st.sidebar.button(
    "🔍 พยากรณ์เลย!", use_container_width=True,
    type="primary")

# ── พยากรณ์ ──────────────────────────────────────
if predict_btn:
    province_enc  = md['le'].transform([province])[0]
    tier          = get_tier(province)
    lag_1, lag_12 = get_avg_lag(province, month)

    if lag_1 is None:
        st.error("ไม่พบข้อมูลของจังหวัดนี้")
        st.stop()

    result, is_short = smart_predict(
        province_enc, month, year,
        lag_1, lag_12, tier, months_ahead)

    # ── ค่าเฉลี่ยและ Season ─────────────────────
    row_ref  = md['avg_df'][
        (md['avg_df']['province_thai']==province)&
        (md['avg_df']['month']==month)]
    avg_val  = row_ref['avg_revenue'].values[0] \
               if len(row_ref)>0 else result
    diff_pct = (result-avg_val)/avg_val*100

    if month in [11,12,1]:   season_label = "🔥 ช่วงท่องเที่ยวสูงสุด"
    elif month in [2,3,10]:  season_label = "🌤 ช่วงท่องเที่ยวปานกลาง"
    else:                     season_label = "🌧 ช่วงท่องเที่ยวต่ำ"

    # ช่วงความคลาดเคลื่อน
    mape_pct = mape_table['short' if is_short else 'long'][tier]
    low_est  = result * (1-mape_pct/100)
    high_est = result * (1+mape_pct/100)

    # ═══════════════════════════════════════════
    # แสดงผล
    # ═══════════════════════════════════════════
    st.subheader(
        f"📍 {province} — {months_full[month-1]} {year+543}")

    # แถวที่ 1
    c1, c2 = st.columns(2)
    with c1:
        direction = "มากกว่า" if diff_pct>0 else "น้อยกว่า"
        st.metric(
            "💰 รายได้ท่องเที่ยวที่คาดการณ์",
            f"{result/1e9:.2f} พันล้านบาท",
            f"{direction}ปกติ {abs(diff_pct):.0f}%")
    with c2:
        st.metric("📅 ช่วงเวลา", season_label)

    # แถวที่ 2
    c3, c4 = st.columns(2)
    with c3:
        acc = "สูง" if is_short else "ปานกลาง"
        acc_mape = "±14.4%" if is_short else "±83.5%"
        st.metric("🎯 ความแม่นยำ", f"{acc} {acc_mape}")
    with c4:
        model_type = "ระยะสั้น 3-6 เดือน" if is_short \
                     else "ระยะยาว 1-5 ปี"
        st.metric("🤖 โมเดลที่ใช้", model_type)

    if not is_short:
        st.warning(
            "⚠️ การพยากรณ์รายได้ระยะยาวมีความคลาดเคลื่อนสูง "
            "แนะนำใช้เพื่อดูทิศทางแนวโน้มเท่านั้น "
            "ไม่ควรใช้ตัวเลขนี้ตัดสินใจลงทุนขนาดใหญ่")
    # ← เพิ่มตรงนี้
    if not is_short and mape_pct > 50:
        st.error(
            f"🚨 ความคลาดเคลื่อนสูงมาก ±{mape_pct:.0f}% "
            f"ช่วงตัวเลขกว้างมาก ({low_est/1e9:.1f} – {high_est/1e9:.1f} พันล้านบาท) "
            f"ไม่แนะนำใช้วางแผนการเงินหรือตัดสินใจลงทุน "
            f"ใช้เพื่อดูทิศทางแนวโน้มเท่านั้น")

    st.divider()

    # ── กล่องช่วงรายได้ ───────────────────────────
    st.subheader("📊 ช่วงรายได้ที่คาดว่าจะเกิดขึ้น")
    if mape_pct <= 50:
        st.caption(
            f"รายได้รวมการท่องเที่ยวของจังหวัด "
            f"คลาดเคลื่อนได้ ±{mape_pct:.0f}% "
            f"แนะนำวางแผนโดยใช้ช่วงตัวเลข")
    else:
        st.caption(
            f"⚠️ ความคลาดเคลื่อน ±{mape_pct:.0f}% สูงมาก "
            f"ช่วงตัวเลขด้านล่างกว้างมาก "
            f"แนะนำใช้ตัวเลขกรณีปกติเป็นหลัก "
            f"และดูกราฟแนวโน้มแทนการยึดตัวเลขแน่นอน")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.error(
            f"**กรณีรายได้น้อย**\n\n"
            f"# {low_est/1e9:.2f}\nพันล้านบาท\n\n"
            f"วางแผนรับมือสถานการณ์แย่")
    with col2:
        st.warning(
            f"**กรณีปกติ (ที่คาดการณ์)**\n\n"
            f"# {result/1e9:.2f}\nพันล้านบาท\n\n"
            f"ใช้เป็นแผนหลัก")
    with col3:
        st.success(
            f"**กรณีรายได้มาก**\n\n"
            f"# {high_est/1e9:.2f}\nพันล้านบาท\n\n"
            f"เตรียมรองรับถ้าเกิดขึ้น")
    # ← เพิ่มตรงนี้ ใต้กล่อง 3 กรณี
    if mape_pct > 50:
        st.caption(
            f"⚠️ ช่วงตัวเลขกว้างมากเพราะความคลาดเคลื่อน ±{mape_pct:.0f}% "
            f"แนะนำให้ใช้ **กรณีปกติ {result/1e9:.2f} พันล้านบาท** "
            f"เป็นหลักในการวางแผน และเตรียมแผนสำรองสำหรับกรณีต่ำสุด")

    # กราฟช่วงรายได้
    fig, ax = plt.subplots(figsize=(9, 2))
    ax.barh(['ช่วงที่คาดหวัง'],
            [(high_est-low_est)/1e9], left=[low_est/1e9],
            color='#fbbf24', alpha=0.8, height=0.4)
    ax.plot([result/1e9],   [0], 'ro', markersize=12, zorder=5,
            label=f'คาดการณ์: {result/1e9:.2f}B')
    ax.plot([low_est/1e9],  [0], 'b|', markersize=20,
            label=f'ต่ำสุด: {low_est/1e9:.2f}B')
    ax.plot([high_est/1e9], [0], 'g|', markersize=20,
            label=f'สูงสุด: {high_est/1e9:.2f}B')
    ax.axvline(avg_val/1e9, color='gray', linestyle='--',
               alpha=0.7, label=f'ปกติ: {avg_val/1e9:.2f}B')
    ax.set_xlabel('รายได้ (พันล้านบาท)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title('ช่วงรายได้ท่องเที่ยวที่คาดว่าจะเกิดขึ้น')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    # ── AI คำแนะนำ ────────────────────────────────
    st.subheader("💡 คำแนะนำสำหรับธุรกิจของคุณ")
    st.caption(
        f"อิงจากรายได้ท่องเที่ยว {low_est/1e9:.2f} "
        f"ถึง {high_est/1e9:.2f} พันล้านบาท "
        f"ใน{months_full[month-1]} {year+543}")

    with st.spinner("กำลังวิเคราะห์และสร้างคำแนะนำ..."):
        advice = get_ai_advice(
            province, months_full[month-1], year,
            biz_type, result, avg_val, mape_pct,
            season_label, low_est, high_est, is_short)

    if month in [11,12,1]:   st.success(advice)
    elif month in [2,3,10]:  st.warning(advice)
    else:                     st.info(advice)

    st.divider()

    # ── กราฟแนวโน้มทั้งปี ────────────────────────
    st.subheader(f"📈 แนวโน้มรายได้ตลอดปี {year+543}")

    monthly_preds = []
    monthly_low   = []
    monthly_high  = []

    for m in range(1, 13):
        l1, l12 = get_avg_lag(province, m)
        if l1 is None:
            monthly_preds.append(0)
            monthly_low.append(0)
            monthly_high.append(0)
            continue
        feats  = md['f_short']      if is_short else md['f_long']
        scaler = md['scaler_short'] if is_short else md['scaler_long']
        models = md['short_models'] if is_short else md['long_models']
        weights= md['w_short']      if is_short else md['w_long']
        X_r    = build_input(province_enc, m, year,
                             l1, l12, tier, feats)
        X_sc   = scaler.transform(X_r)
        p      = predict_ensemble(models, X_sc, weights)
        mp     = mape_table['short' if is_short else 'long'][tier]
        monthly_preds.append(p)
        monthly_low.append(p*(1-mp/100))
        monthly_high.append(p*(1+mp/100))

    colors_bar = ['#ef4444' if m in [11,12,1]
                  else '#f59e0b' if m in [2,3,10]
                  else '#3b82f6' for m in range(1,13)]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(months_th,
                  [p/1e9 for p in monthly_preds],
                  color=colors_bar, alpha=0.75,
                  label='คาดการณ์')
    ax.errorbar(
        months_th,
        [p/1e9 for p in monthly_preds],
        yerr=[[(p-l)/1e9 for p,l in
               zip(monthly_preds, monthly_low)],
              [(h-p)/1e9 for p,h in
               zip(monthly_preds, monthly_high)]],
        fmt='none', color='black',
        capsize=4, alpha=0.5, label='ช่วงคลาดเคลื่อน')
    for bar, val in zip(bars, monthly_preds):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+max(monthly_preds)*0.01/1e9,
                f'{val/1e9:.1f}B',
                ha='center', fontsize=8)
    bars[month-1].set_edgecolor('black')
    bars[month-1].set_linewidth(3)
    ax.set_title(
        f'รายได้ท่องเที่ยว {province} ปี {year+543} (หน่วย: พันล้านบาท)\n'
        f'แดง = ช่วงท่องเที่ยวสูงสุด   เหลือง = ช่วงท่องเที่ยวปานกลาง   น้ำเงิน = ช่วงท่องเที่ยวต่ำ   '
        f'เส้น = ช่วงคลาดเคลื่อน',
        fontsize=11)
    ax.set_ylabel('รายได้ (พันล้านบาท)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    st.pyplot(fig)

    # ── ข้อมูลเพิ่มเติม ───────────────────────────
    st.divider()
    st.subheader("ℹ️ ข้อมูลเพิ่มเติมสำหรับตัดสินใจ")

    c1, c2 = st.columns(2)
    with c1:
        st.info(
            f"**ความคลาดเคลื่อนที่คาดได้**\n\n"
            f"±{mape_pct:.0f}% หรือประมาณ "
            f"±{result*mape_pct/100/1e9:.2f} พันล้านบาท\n\n"
            f"{'ระยะสั้น แม่นยำกว่า' if is_short else 'ระยะยาว คลาดเคลื่อนมากกว่า'}")
    with c2:
        compare = "มากกว่าปกติ" if diff_pct>0 else "น้อยกว่าปกติ"
        st.info(
            f"**เทียบกับปกติ**\n\n"
            f"{compare} {abs(diff_pct):.0f}%\n\n"
            f"รายได้ปกติเดือนนี้: {avg_val/1e9:.2f} พันล้านบาท")

    tier_name = {1:"เมืองท่องเที่ยวหลัก",
                 2:"เมืองท่องเที่ยวรอง",
                 3:"เมืองขนาดเล็ก"}[tier]
    note = "ข้อมูลมาก พยากรณ์แม่นยำสูง" if tier==1 \
           else "อาจคลาดเคลื่อนได้มากกว่าเมืองหลัก"
    st.info(f"**ประเภทพื้นที่:** {tier_name} — {note}")