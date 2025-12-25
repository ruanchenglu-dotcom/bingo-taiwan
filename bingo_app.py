import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import re
import shutil
from collections import Counter
from datetime import datetime
import plotly.express as px
from PIL import Image, ImageOps
import pytesseract
import cv2

# ==============================================================================
# 1. CẤU HÌNH & HỆ THỐNG
# ==============================================================================
st.set_page_config(page_title="Bingo AI - V11 Full Option", layout="wide")

st.markdown("""
<style>
    div.stButton > button:first-child { min-height: 65px; width: 100%; margin: 0px 1px; font-weight: bold; border-radius: 6px; font-size: 18px; }
    .raw-text-box { background-color: #f8f9fa; border: 1px solid #ddd; padding: 10px; font-family: monospace; font-size: 12px; height: 100px; overflow-y: scroll; white-space: pre-wrap;}
    .success-msg { color: #155724; background-color: #d4edda; border-color: #c3e6cb; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
    .anomaly-box-hot { background-color: #ffe6e6; padding: 10px; border-radius: 5px; border-left: 5px solid #ff4b4b; color: #c0392b; margin-bottom: 5px;}
    .anomaly-box-cold { background-color: #e8f8f5; padding: 10px; border-radius: 5px; border-left: 5px solid #1abc9c; color: #16a085; margin-bottom: 5px;}
    .kelly-box { background-color: #fff8e1; padding: 15px; border-radius: 8px; border: 2px solid #f1c40f; text-align: center; font-weight: bold; font-size: 18px; }
</style>
""", unsafe_allow_html=True)

DATA_FILE = 'bingo_history.csv'

def check_tesseract():
    path = shutil.which("tesseract")
    if path is None: return False, "❌ LỖI: Chưa cài Tesseract!"
    return True, "✅ System OK"

# ==============================================================================
# 2. XỬ LÝ ẢNH (GIỮ NGUYÊN CÔNG NGHỆ V9 TỐT NHẤT)
# ==============================================================================
def preprocess_image_v9(image):
    # Upscale & HSV Filter (Lọc trắng - loại bỏ bóng/lửa)
    img = np.array(image.convert('RGB'))
    img = cv2.resize(img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Lọc lấy màu trắng (Số)
    lower_white = np.array([0, 0, 130]) 
    upper_white = np.array([180, 80, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Đảo màu (Chữ đen nền trắng)
    result = cv2.bitwise_not(mask)
    result = cv2.copyMakeBorder(result, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
    return result

def extract_text_v9(image):
    try:
        processed_img = preprocess_image_v9(image)
        st.image(processed_img, caption="Ảnh máy tính đọc (Đã lọc sạch màu)", width=400)
        config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789: preserve_interword_spaces=1'
        text = pytesseract.image_to_string(processed_img, config=config)
        return text
    except Exception as e:
        return f"ERROR: {str(e)}"

# ==============================================================================
# 3. BỘ PHÂN TÍCH ẢNH
# ==============================================================================
def parse_bingo_results_v9(text, selected_date, start_draw_id):
    results = []
    lines = text.split('\n')
    current_draw_id = start_draw_id
    
    for line in lines:
        if not line.strip(): continue
        clean_line = line.replace('O', '0').replace('o', '0').replace('l', '1').replace('I', '1').replace('|', '1').replace('S','5')
        
        match_id = re.search(r'114\d{6,}', clean_line)
        found_draw_id = 0
        if match_id:
            raw_id_str = match_id.group()
            found_draw_id = int(raw_id_str[:9])
            clean_line = clean_line.replace(raw_id_str, "") 
        
        raw_chunks = re.findall(r'\d+', clean_line)
        bingo_nums = []
        for chunk in raw_chunks:
            if len(chunk) > 2: 
                split_nums = [chunk[i:i+2] for i in range(0, len(chunk), 2)]
                for n_str in split_nums:
                    try:
                        val = int(n_str)
                        if 1 <= val <= 80: bingo_nums.append(val)
                    except: pass
            else:
                try:
                    val = int(chunk)
                    if 1 <= val <= 80: bingo_nums.append(val)
                except: pass
        
        if len(bingo_nums) >= 15:
            unique = []
            seen = set()
            for x in bingo_nums:
                if x not in seen:
                    unique.append(x)
                    seen.add(x)
            
            final_id = found_draw_id if found_draw_id > 0 else current_draw_id
            main_20 = sorted(unique[:20])
            while len(main_20) < 20: main_20.append(0)
            super_n = unique[20] if len(unique) > 20 else 0
            
            results.append({
                'draw_id': final_id,
                'time': datetime.combine(selected_date, datetime.now().time()),
                'nums': main_20,
                'super_num': super_n
            })
            
            if found_draw_id == 0: current_draw_id -= 1
            else: current_draw_id = found_draw_id - 1
            
    return results

# ==============================================================================
# 4. MODULE PHÂN TÍCH & KELLY (UPDATE ĐẦY ĐỦ 1-10 SAO)
# ==============================================================================
def calculate_z_scores(df):
    if df.empty: return None, pd.Series(), pd.Series()
    recent = df.head(30)
    all_nums = []
    for i in range(1, 21): all_nums.extend(recent[f'num_{i}'].tolist())
    
    counts = pd.Series(all_nums).value_counts().reindex(range(1, 81), fill_value=0)
    mean = counts.mean(); std = counts.std()
    
    if std == 0: return pd.Series(), pd.Series(), pd.Series() 
    
    z_scores = (counts - mean) / std
    hot = z_scores[z_scores > 1.5].sort_values(ascending=False)
    cold = z_scores[z_scores < -1.5].sort_values(ascending=True)
    return z_scores, hot, cold

def kelly_suggestion(win_prob, odds, bankroll):
    b = odds - 1
    p = win_prob
    q = 1 - p
    f = (b * p - q) / b
    safe_f = max(0, f * 0.5) 
    return safe_f * 100, bankroll * safe_f

def run_prediction(df, algo):
    if df.empty: return []
    recent = df.head(10)
    nums = [n for i in range(1,21) for n in recent[f'num_{i}']]
    freq = pd.Series(nums).value_counts()
    last = [df.iloc[0][f'num_{i}'] for i in range(1,21)]
    scores = {}
    for n in range(1, 81):
        if algo == "🔮 AI Master (Tổng Hợp)":
            s = freq.get(n, 0) * 1.5
            if n in last: s += 3.0
            if (n-1) in last or (n+1) in last: s += 1.0
            s += random.uniform(0, 1.0)
            scores[n] = s
        elif algo == "🔥 Soi Cầu Nóng (Hot)": scores[n] = freq.get(n, 0) + random.random()*0.1
        elif algo == "❄️ Soi Cầu Lạnh (Nuôi)": scores[n] = (freq.max() if not freq.empty else 0 - freq.get(n, 0)) + random.uniform(0, 1.5)
        elif algo == "♻️ Soi Cầu Bệt (Lại)": scores[n] = (1000 if n in last else 0) + freq.get(n, 0)*0.1
    return sorted(scores, key=scores.get, reverse=True)

# ==============================================================================
# 5. CORE LOGIC (Load/Save)
# ==============================================================================
def load_data():
    num_cols = [f'num_{i}' for i in range(1, 21)]
    columns = ['draw_id', 'time'] + num_cols + ['super_num']
    df = pd.DataFrame(columns=columns)
    if os.path.exists(DATA_FILE):
        try: loaded_df = pd.read_csv(DATA_FILE); df = loaded_df if not loaded_df.empty else df
        except: pass
    if 'draw_id' in df.columns: df['draw_id'] = pd.to_numeric(df['draw_id'], errors='coerce').fillna(0).astype(int)
    if 'time' in df.columns: df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df[df['draw_id'] > 0].sort_values(by='draw_id', ascending=False).drop_duplicates(subset=['draw_id'], keep='first')
    return df

def save_data(df): df.sort_values(by='draw_id', ascending=False).to_csv(DATA_FILE, index=False)
def delete_last_row(): df = load_data(); df=df.iloc[1:] if not df.empty else df; save_data(df); return True
def toggle_number(n): 
    if n in st.session_state.selected_nums: st.session_state.selected_nums.remove(n)
    else: st.session_state.selected_nums.append(n) if len(st.session_state.selected_nums)<20 else st.toast("Max 20!")

if 'selected_nums' not in st.session_state: st.session_state.selected_nums = []
if 'ocr_result' not in st.session_state: st.session_state.ocr_result = []
if 'predict_data' not in st.session_state: st.session_state.predict_data = None
if 'z_score_data' not in st.session_state: st.session_state.z_score_data = None
if 'selected_algo' not in st.session_state: st.session_state.selected_algo = "🔮 AI Master (Tổng Hợp)"

# ==============================================================================
# 6. GIAO DIỆN CHÍNH (UI)
# ==============================================================================
st.title("🎲 BINGO V11 - FULL OPTION")
df_history = load_data()
status, msg = check_tesseract()

with st.container(border=True):
    t1, t2 = st.tabs(["📸 QUÉT ẢNH", "⚙️ NHẬP TAY"])
    
    with t1:
        st.info("💡 Mẹo: Chụp phần số rõ ràng. Nhập Mã kỳ dòng đầu tiên để máy tự điền.")
        c_up, c_setting = st.columns([2, 1])
        with c_up:
            up_file = st.file_uploader("Upload ảnh:", type=['png','jpg','jpeg'])
        with c_setting:
            s_date = st.date_input("Ngày:", datetime.now())
            suggest_id = int(df_history['draw_id'].max()) + 1 if not df_history.empty else 114000001
            start_id_input = st.number_input("Mã kỳ dòng đầu (Gợi ý):", value=suggest_id, step=1, format="%d")

        if up_file and st.button("🔍 QUÉT NGAY"):
            if status:
                img = Image.open(up_file)
                with st.spinner("Đang xử lý ảnh..."):
                    raw_txt = extract_text_v9(img)
                    st.caption("Raw Text (Debug):")
                    st.markdown(f"<div class='raw-text-box'>{raw_txt}</div>", unsafe_allow_html=True)
                    res = parse_bingo_results_v9(raw_txt, s_date, start_id_input)
                    if res:
                        st.session_state.ocr_result = res
                        st.markdown(f"<div class='success-msg'>✅ Tìm thấy {len(res)} dòng số!</div>", unsafe_allow_html=True)
                    else:
                        st.error("❌ Không tìm thấy dãy số nào (cần ít nhất 15 số/dòng).")

        if st.session_state.ocr_result:
            st.write("### 👇 KIỂM TRA & LƯU:")
            for i, it in enumerate(st.session_state.ocr_result):
                with st.expander(f"Kỳ {it['draw_id']} (Đã tách số)", expanded=True):
                    c1, c2, c3 = st.columns([1, 3, 1])
                    new_id = c1.number_input("Mã Kỳ:", value=it['draw_id'], key=f"id_{i}", format="%d")
                    n_str = c2.text_area("Dãy số:", ", ".join(map(str, it['nums'])), key=f"n{i}", height=68)
                    s_num = c3.number_input("Siêu cấp:", value=it['super_num'], key=f"s{i}")
                    try:
                        st.session_state.ocr_result[i]['draw_id'] = new_id
                        st.session_state.ocr_result[i]['nums'] = sorted([int(x) for x in n_str.split(',') if x.strip().isdigit()])
                        st.session_state.ocr_result[i]['super_num'] = s_num
                    except: pass
            
            if st.button("💾 LƯU TẤT CẢ VÀO LỊCH SỬ", type="primary"):
                cnt = 0
                for it in st.session_state.ocr_result:
                    if df_history.empty or it['draw_id'] not in df_history['draw_id'].values:
                        r = {'draw_id': it['draw_id'], 'time': it['time'], 'super_num': it['super_num']}
                        for k, v in enumerate(it['nums']): r[f'num_{k+1}'] = v if k<20 else 0
                        for k in range(len(it['nums']), 20): r[f'num_{k+1}'] = 0
                        df_history = pd.concat([pd.DataFrame([r]), df_history], ignore_index=True)
                        cnt+=1
                if cnt: save_data(df_history); st.success(f"Đã lưu {cnt} kỳ!"); st.session_state.ocr_result=[]; st.rerun()
                else: st.warning("Dữ liệu trùng lặp!")

    with t2:
        c1, c2, c3 = st.columns([2,2,1])
        nid = str(int(df_history['draw_id'].max()) + 1) if not df_history.empty else ""
        mid = c1.text_input("Mã Kỳ:", value=nid)
        mdate = c2.date_input("Ngày:", datetime.now(), key="d2")
        if c3.button("Xóa"): st.session_state.selected_nums = []
        for r in range(8):
            cols = st.columns(10)
            for c in range(10):
                n = r*10 + c + 1
                bg = "primary" if n in st.session_state.selected_nums else "secondary"
                if cols[c].button(f"{n:02d}", key=f"b{n}", type=bg): toggle_number(n); st.rerun()
        if st.button("LƯU TAY"):
            r = {'draw_id': int(mid) if mid else 0, 'time': datetime.combine(mdate, datetime.now().time()), 'super_num': 0}
            for i,v in enumerate(sorted(st.session_state.selected_nums)): r[f'num_{i+1}'] = v
            save_data(pd.concat([pd.DataFrame([r]), df_history], ignore_index=True)); st.success("Lưu!"); st.rerun()

# --- KHU VỰC PHÂN TÍCH ---
st.markdown("---")
st.header("📊 PHÂN TÍCH & DỰ ĐOÁN")

if st.button("🚀 CHẠY PHÂN TÍCH TOÀN DIỆN", type="primary", use_container_width=True):
    if not df_history.empty:
        st.session_state.predict_data = run_prediction(df_history, st.session_state.selected_algo)
        st.session_state.z_score_data = calculate_z_scores(df_history)
        st.toast("Phân tích hoàn tất!", icon="✅")
    else: st.error("Chưa có dữ liệu lịch sử để phân tích.")

if st.session_state.predict_data:
    rt1, rt2 = st.tabs(["📉 Z-SCORE (SĂN CẦU)", "💰 DỰ ĐOÁN & KELLY"])
    
    with rt1:
        z_all, hot, cold = st.session_state.z_score_data
        c1, c2 = st.columns(2)
        with c1:
            st.write("#### 🔥 SỐ NÓNG (Z > 1.5)")
            if not hot.empty:
                for n,s in hot.items(): st.markdown(f"<div class='anomaly-box-hot'>🔴 Số <b>{n:02d}</b> (Z: {s:.2f})</div>", unsafe_allow_html=True)
            else: st.info("Không có số nóng bất thường.")
        with c2:
            st.write("#### ❄️ SỐ LẠNH (Z < -1.5)")
            if not cold.empty:
                for n,s in cold.items(): st.markdown(f"<div class='anomaly-box-cold'>🔵 Số <b>{n:02d}</b> (Z: {s:.2f})</div>", unsafe_allow_html=True)
            else: st.info("Không có số lạnh bất thường.")
        st.plotly_chart(px.bar(x=z_all.index, y=z_all.values, labels={'x': 'Số', 'y': 'Z-Score'}, color=z_all.values, color_continuous_scale='RdBu_r'), use_container_width=True)

    with rt2:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("KẾT QUẢ DỰ ĐOÁN AI")
            salgo = st.selectbox("Thuật toán:", ["🔮 AI Master (Tổng Hợp)", "🔥 Soi Cầu Nóng (Hot)", "❄️ Soi Cầu Lạnh (Nuôi)", "♻️ Soi Cầu Bệt (Lại)"])
            if salgo != st.session_state.selected_algo:
                st.session_state.selected_algo = salgo
                st.session_state.predict_data = run_prediction(df_history, salgo)
                st.rerun()
            
            # --- MENU CHỌN DÀN FULL 1-10 ---
            modes_dict = {
                "10 Tinh (10 số)": 10,
                "9 Tinh (9 số)": 9,
                "8 Tinh (8 số)": 8,
                "7 Tinh (7 số)": 7,
                "6 Tinh (6 số)": 6,
                "5 Tinh (5 số)": 5,
                "4 Tinh (4 số)": 4,
                "3 Tinh (3 số)": 3,
                "2 Tinh (2 số)": 2,
                "1 Tinh (1 số)": 1
            }
            # Mặc định chọn 6 tinh (index 4)
            smode_label = st.selectbox("Chọn dàn:", list(modes_dict.keys()), index=4)
            pick_n = modes_dict[smode_label]
            
            fnums = sorted(list(st.session_state.predict_data)[:pick_n])
            
            cols = st.columns(5)
            for i, n in enumerate(fnums): 
                cols[i%5].markdown(f"<div style='background-color:{'#E74C3C' if n>40 else '#3498DB'}; color:white; padding:10px; border-radius:5px; text-align:center; font-weight:bold; font-size:20px; margin-bottom:5px'>{n:02d}</div>", unsafe_allow_html=True)

        with c2:
            st.subheader("💰 QUẢN LÝ VỐN (KELLY)")
            my_money = st.number_input("Vốn hiện có (Đài tệ):", value=10000, step=1000)
            
            # --- CẤU HÌNH KELLY TỰ ĐỘNG THEO SỐ SAO ---
            # Số sao càng cao, xác suất trúng càng thấp nhưng tỷ lệ ăn (odds) càng cao
            if pick_n <= 4:
                ai_win = 0.55; odds_val = 2.0
            elif pick_n <= 7:
                ai_win = 0.35; odds_val = 4.0
            else: # 8,9,10 Tinh
                ai_win = 0.15; odds_val = 10.0
            
            kp, km = kelly_suggestion(ai_win, odds_val, my_money)
            
            if kp > 0:
                st.markdown(f"<div class='kelly-box'>💡 GỢI Ý:<br><span style='color:#e67e22'>{kp:.1f}% Vốn</span><br><span style='color:#27ae60'>${km:,.0f} TWD</span></div>", unsafe_allow_html=True)
            else:
                st.warning("⚠️ Rủi ro cao. Nên bảo toàn vốn.")

with st.expander("LỊCH SỬ KỲ QUAY"):
    if st.button("Xóa kỳ cuối"): delete_last_row(); st.rerun()
    st.dataframe(df_history, use_container_width=True)
