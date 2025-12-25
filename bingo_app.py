import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import re
from collections import Counter
from datetime import datetime
import plotly.express as px

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG
# ==============================================================================
st.set_page_config(
    page_title="Bingo Quantum AI - Z-Score Edition", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# CSS Tùy chỉnh (Giao diện Chuyên gia)
st.markdown("""
<style>
    div.stButton > button:first-child {
        min-height: 65px; width: 100%; margin: 0px 1px;
        font-weight: bold; border-radius: 6px; font-size: 18px;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem; font-weight: bold;
    }
    [data-testid="column"] { padding: 0px 2px; }
    .anomaly-box-hot { background-color: #ffe6e6; padding: 10px; border-radius: 5px; border-left: 5px solid #ff4b4b; margin-bottom: 5px; color: #c0392b;}
    .anomaly-box-cold { background-color: #e8f8f5; padding: 10px; border-radius: 5px; border-left: 5px solid #1abc9c; margin-bottom: 5px; color: #16a085;}
    .kelly-box { background-color: #fff8e1; padding: 15px; border-radius: 8px; border: 2px solid #f1c40f; text-align: center; font-weight: bold; font-size: 18px; }
</style>
""", unsafe_allow_html=True)

DATA_FILE = 'bingo_history.csv'

# ==============================================================================
# 2. STATE & DATA
# ==============================================================================
if 'selected_nums' not in st.session_state: st.session_state['selected_nums'] = [] 
if 'predict_data' not in st.session_state: st.session_state['predict_data'] = None 
if 'z_score_data' not in st.session_state: st.session_state['z_score_data'] = None # New State
if 'selected_algo' not in st.session_state: st.session_state['selected_algo'] = "🔮 AI Master (Tổng Hợp)"
if 'paste_key_id' not in st.session_state: st.session_state['paste_key_id'] = 0

def load_data():
    num_cols = [f'num_{i}' for i in range(1, 21)]
    columns = ['draw_id', 'time'] + num_cols + ['super_num']
    df = pd.DataFrame(columns=columns)
    if os.path.exists(DATA_FILE):
        try:
            loaded_df = pd.read_csv(DATA_FILE)
            if not loaded_df.empty: df = loaded_df
        except: pass
    if 'draw_id' in df.columns:
        df['draw_id'] = pd.to_numeric(df['draw_id'], errors='coerce').fillna(0).astype(int)
    if 'time' in df.columns: df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df[df['draw_id'] > 0].sort_values(by='draw_id', ascending=False).drop_duplicates(subset=['draw_id'], keep='first')
    return df

def save_data(df):
    df.sort_values(by='draw_id', ascending=False).to_csv(DATA_FILE, index=False)

def delete_last_row():
    df = load_data()
    if not df.empty: df = df.iloc[1:]; save_data(df); return True
    return False

def delete_all_data():
    if os.path.exists(DATA_FILE): os.remove(DATA_FILE); return True
    return False

# ==============================================================================
# 3. LOGIC & PARSER
# ==============================================================================
def toggle_number(num):
    if num in st.session_state['selected_nums']: st.session_state['selected_nums'].remove(num)
    else:
        if len(st.session_state['selected_nums']) < 20: st.session_state['selected_nums'].append(num)
        else: st.toast("⚠️ Max 20 số!", icon="🚫")

def clear_selection(): st.session_state['selected_nums'] = []
def clear_paste_box(): st.session_state['paste_key_id'] += 1

def parse_multi_draws(text, selected_date):
    results = []
    matches = list(re.finditer(r'\b114\d{6}\b', text))
    for i in range(len(matches)):
        try:
            did = int(matches[i].group())
            s = matches[i].end()
            e = matches[i+1].start() if i + 1 < len(matches) else len(text)
            seg = text[s:e]
            nums = sorted(list(set([int(n) for n in re.findall(r'\d{2}', seg) if 1 <= int(n) <= 80]))[:20])
            if len(nums) >= 15:
                results.append({'draw_id': did, 'time': datetime.combine(selected_date, datetime.now().time()), 'nums': nums, 'super_num': nums[-1]})
        except: continue
    return results

# ==============================================================================
# 4. QUANTUM ANALYSIS (Z-SCORE & KELLY)
# ==============================================================================
def calculate_z_scores(df):
    """Tính toán Z-Score cho 80 số dựa trên 30 kỳ gần nhất"""
    if df.empty: return None, [], []
    
    recent = df.head(30)
    all_nums = []
    for i in range(1, 21): all_nums.extend(recent[f'num_{i}'].tolist())
    
    counts = pd.Series(all_nums).value_counts().reindex(range(1, 81), fill_value=0)
    
    # Thống kê cơ bản
    mean = counts.mean() # Trung bình
    std = counts.std()   # Độ lệch chuẩn
    
    # Tính Z-Score: (Giá trị - Trung bình) / Độ lệch chuẩn
    z_scores = (counts - mean) / std
    
    # Lọc Dị Biệt (Anomalies)
    # Z > 1.5: Nóng bất thường
    # Z < -1.5: Lạnh bất thường
    hot_anomalies = z_scores[z_scores > 1.5].sort_values(ascending=False)
    cold_anomalies = z_scores[z_scores < -1.5].sort_values(ascending=True)
    
    return z_scores, hot_anomalies, cold_anomalies

def kelly_criterion_suggestion(win_prob=0.25, odds=3.0, bankroll=10000):
    """Gợi ý đi tiền theo Kelly"""
    # f = (bp - q) / b
    # b = odds - 1 (Tỷ lệ cược ròng)
    # p = win_prob (Xác suất thắng)
    # q = 1 - p (Xác suất thua)
    b = odds - 1
    p = win_prob
    q = 1 - p
    f = (b * p - q) / b
    
    # Kelly an toàn (Half Kelly) để giảm rủi ro
    safe_f = f * 0.5 
    if safe_f < 0: safe_f = 0
    
    bet_amount = bankroll * safe_f
    return safe_f * 100, bet_amount

def run_prediction(df, strategy):
    if df.empty: return []
    recent = df.head(10)
    all_nums = []
    for i in range(1, 21): all_nums.extend(recent[f'num_{i}'].tolist())
    freq = pd.Series(all_nums).value_counts()
    last = [df.iloc[0][f'num_{i}'] for i in range(1, 21)]
    scores = {}
    for n in range(1, 81):
        if strategy == "🔮 AI Master":
            s = freq.get(n, 0) * 1.5
            if n in last: s += 3.0
            if (n-1) in last or (n+1) in last: s += 1.0
            s += random.uniform(0, 1.0)
            scores[n] = s
        elif strategy == "🔥 Soi Cầu Nóng": scores[n] = freq.get(n, 0) + (random.random() * 0.1)
        elif strategy == "❄️ Soi Cầu Lạnh": scores[n] = (freq.max() if not freq.empty else 0 - freq.get(n, 0)) + random.uniform(0, 1.5)
        elif strategy == "♻️ Soi Cầu Bệt": scores[n] = (1000 if n in last else 0) + freq.get(n, 0)*0.1
    return sorted(scores, key=scores.get, reverse=True)

# ==============================================================================
# 5. UI CHÍNH
# ==============================================================================
st.title("🎲 BINGO QUANTUM - Z-SCORE EDITION")
df_history = load_data()

with st.container(border=True):
    t1, t2 = st.tabs(["🖱️ BÀN PHÍM SỐ", "📋 DÁN (COPY)"])
    with t1:
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1: nid = str(int(df_history['draw_id'].max()) + 1) if not df_history.empty else ""; mid = st.text_input("Mã Kỳ:", value=nid, key="mid")
        with c2: mdate = st.date_input("Ngày:", datetime.now(), key="mdate")
        with c3: st.write(""); st.write(""); st.button("Xóa chọn", key="b_clr", on_click=clear_selection)
        
        st.markdown(f"**🔢 Chọn: <span style='color:red'>{len(st.session_state['selected_nums'])}/20</span>**", unsafe_allow_html=True)
        for r in range(8):
            cols = st.columns(10)
            for c in range(10):
                n = r*10 + c + 1
                with cols[c]:
                    sel = n in st.session_state['selected_nums']
                    if st.button(f"{n:02d}", key=f"g_{n}", type="primary" if sel else "secondary"): toggle_number(n); st.rerun()
        
        st.markdown("---")
        v_supers = sorted(st.session_state['selected_nums']) if st.session_state['selected_nums'] else range(1, 81)
        msuper = st.selectbox("🔥 Siêu Cấp:", v_supers, key="msup")
        
        if st.button("💾 LƯU THỦ CÔNG", type="primary", use_container_width=True):
            if not mid or len(st.session_state['selected_nums']) != 20: st.error("Lỗi nhập liệu!")
            elif not df_history.empty and int(mid) in df_history['draw_id'].values: st.warning("Đã tồn tại!")
            else:
                row = {'draw_id': int(mid), 'time': datetime.combine(mdate, datetime.now().time()), 'super_num': msuper}
                for i, v in enumerate(sorted(st.session_state['selected_nums'])): row[f'num_{i+1}'] = v
                save_data(pd.concat([pd.DataFrame([row]), df_history], ignore_index=True))
                st.success("Đã lưu!"); clear_selection(); st.rerun()

    with t2:
        c1, c2 = st.columns([3, 1])
        with c1: pdate = st.date_input("Ngày:", datetime.now(), key="pdate")
        with c2: st.button("🗑 Xóa ô dán", on_click=clear_paste_box, use_container_width=True)
        ptext = st.text_area("Dán dữ liệu:", height=150, key=f"parea_{st.session_state['paste_key_id']}")
        if st.button("💾 XỬ LÝ & LƯU", type="primary", use_container_width=True):
            ext = parse_multi_draws(ptext, pdate)
            if ext:
                added = 0
                for it in ext:
                    if df_history.empty or it['draw_id'] not in df_history['draw_id'].values:
                        r = {'draw_id': it['draw_id'], 'time': it['time'], 'super_num': it['super_num']}
                        for i, v in enumerate(it['nums']): r[f'num_{i+1}'] = v
                        df_history = pd.concat([pd.DataFrame([r]), df_history], ignore_index=True)
                        added += 1
                if added: save_data(df_history); st.success(f"Thêm {added} kỳ!"); st.rerun()
                else: st.warning("Dữ liệu cũ!")
            else: st.error("Lỗi dữ liệu!")

st.write(""); st.markdown("### 📊 PHÂN TÍCH ĐỊNH LƯỢNG (QUANTITATIVE)")

if st.button("🚀 CHẠY PHÂN TÍCH TOÀN DIỆN", type="primary", use_container_width=True):
    if not df_history.empty:
        st.session_state['predict_data'] = run_prediction(df_history, st.session_state['selected_algo'])
        st.session_state['z_score_data'] = calculate_z_scores(df_history)
        st.toast("Phân tích hoàn tất!", icon="✅")
    else: st.error("Chưa có dữ liệu.")

if st.session_state['predict_data'] or not df_history.empty:
    st.markdown("---")
    rt1, rt2 = st.tabs(["📉 PHÂN TÍCH Z-SCORE (DỊ BIỆT)", "🎯 DỰ ĐOÁN & QUẢN LÝ VỐN"])
    
    # --- TAB Z-SCORE (TÍNH NĂNG MỚI) ---
    with rt1:
        st.subheader("🔍 Tìm Kiếm Sự Dị Biệt (Statistical Anomalies)")
        st.caption("Dựa trên 30 kỳ gần nhất. Nếu Z-Score > 1.5 là RẤT NÓNG. Z-Score < -1.5 là RẤT LẠNH (Sắp nổ).")
        
        if st.session_state['z_score_data']:
            z_all, hots, colds = st.session_state['z_score_data']
            
            c_hot, c_cold = st.columns(2)
            with c_hot:
                st.markdown("#### 🔥 CÁC SỐ 'NÓNG' BẤT THƯỜNG (Z > 1.5)")
                st.write("👉 *Chiến thuật: Bám theo dây đỏ (Đánh tiếp)*")
                if not hots.empty:
                    for n, score in hots.items():
                        st.markdown(f"<div class='anomaly-box-hot'>🔴 Số <b>{n:02d}</b> (Z-Score: {score:.2f}) - Siêu Hot</div>", unsafe_allow_html=True)
                else: st.info("Không có số nào nóng bất thường.")
                
            with c_cold:
                st.markdown("#### ❄️ CÁC SỐ 'LẠNH' BẤT THƯỜNG (Z < -1.5)")
                st.write("👉 *Chiến thuật: Nuôi gấp thếp (Sắp nổ)*")
                if not colds.empty:
                    for n, score in colds.items():
                        st.markdown(f"<div class='anomaly-box-cold'>🔵 Số <b>{n:02d}</b> (Z-Score: {score:.2f}) - Siêu Lạnh</div>", unsafe_allow_html=True)
                else: st.info("Không có số nào lạnh bất thường.")
                
            # Biểu đồ Z-Score
            st.markdown("---")
            st.markdown("##### 📈 Biểu đồ phân phối Z-Score toàn bộ 80 số")
            fig = px.bar(x=z_all.index, y=z_all.values, labels={'x': 'Số (1-80)', 'y': 'Z-Score (Độ lệch chuẩn)'}, color=z_all.values, color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Chưa chạy phân tích.")

    # --- TAB DỰ ĐOÁN & KELLY ---
    with rt2:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("KẾT QUẢ DỰ ĐOÁN")
            algos = ["🔮 AI Master", "🔥 Soi Cầu Nóng", "❄️ Soi Cầu Lạnh", "♻️ Soi Cầu Bệt"]
            salgo = st.selectbox("Thuật toán:", algos, index=0)
            if salgo != st.session_state['selected_algo']:
                st.session_state['selected_algo'] = salgo
                if not df_history.empty: st.session_state['predict_data'] = run_prediction(df_history, salgo); st.rerun()
            
            modes = {"10 Tinh": 10, "6 Tinh": 6, "1 Tinh": 1}
            smode = st.selectbox("Dàn:", list(modes.keys()), index=1)
            
            if st.session_state['predict_data']:
                fnums = sorted(st.session_state['predict_data'][:modes[smode]])
                cols = st.columns(5)
                for i, n in enumerate(fnums): 
                    cols[i%5].markdown(f"<div style='background-color:{'#E74C3C' if n>40 else '#3498DB'}; color:white; padding:10px; border-radius:5px; text-align:center; font-weight:bold; font-size:20px; margin-bottom:5px'>{n:02d}</div>", unsafe_allow_html=True)

        # GỢI Ý ĐI TIỀN KELLY
        with c2:
            st.subheader("💰 QUẢN LÝ VỐN (KELLY)")
            st.caption("Công thức Kelly giúp bạn biết nên đánh bao nhiêu tiền.")
            
            my_money = st.number_input("Vốn hiện có (Đài tệ):", value=10000, step=1000)
            
            # Giả định tỷ lệ thắng cho 1 Tinh (~25%)
            win_pct = 0.25 
            if smode == "6 Tinh": win_pct = 0.15 # Khó hơn xíu
            
            kelly_pct, kelly_money = kelly_criterion_suggestion(win_prob=win_pct, odds=2.0, bankroll=my_money) # Odds 1 ăn 2
            
            st.markdown(f"""
            <div class='kelly-box'>
                💡 GỢI Ý ĐI TIỀN:<br>
                <span style='color:#e67e22; font-size: 24px'>{kelly_pct:.1f}% Vốn</span><br>
                Tương đương: <span style='color:#27ae60; font-size: 24px'>${kelly_money:,.0f} TWD</span>
            </div>
            """, unsafe_allow_html=True)
            st.info("⚠️ Đây là mức cược tối ưu toán học (Kelly an toàn). Đừng đánh hơn số này.")

st.markdown("---")
with st.expander("LỊCH SỬ"):
    if st.button("Xóa kỳ cuối"): delete_last_row(); st.rerun()
    if not df_history.empty: st.dataframe(df_history, use_container_width=True, hide_index=True)
