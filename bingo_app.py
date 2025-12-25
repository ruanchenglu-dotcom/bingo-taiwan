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
# 1. CẤU HÌNH & GIAO DIỆN MOBILE
# ==============================================================================
st.set_page_config(page_title="Bingo V13 - Final Mobile", layout="wide")

# CSS TỐI ƯU GIAO DIỆN
st.markdown("""
<style>
    /* Nút bấm số: To, Rộng, Sát nhau */
    div.stButton > button:first-child { 
        min-height: 50px !important; 
        width: 100% !important; 
        margin: 1px 0px !important;
        padding: 0px !important;
        font-weight: bold; 
        border-radius: 4px; 
        font-size: 18px; /* Chữ to hơn */
    }
    
    /* Thu hẹp khoảng cách cột để khít màn hình */
    [data-testid="column"] {
        padding: 0px 1px !important;
        min-width: 0px !important;
        gap: 0px !important;
    }
    
    .kelly-box { background-color: #fff8e1; padding: 15px; border-radius: 8px; border: 2px solid #f1c40f; text-align: center; font-weight: bold; font-size: 18px; }
    .success-msg { color: #155724; background-color: #d4edda; border-color: #c3e6cb; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

DATA_FILE = 'bingo_history.csv'

def check_tesseract():
    path = shutil.which("tesseract")
    if path is None: return False, "❌ LỖI: Chưa cài Tesseract!"
    return True, "✅ System OK"

# ==============================================================================
# 2. XỬ LÝ ẢNH (V9 ENGINE - GIỮ NGUYÊN)
# ==============================================================================
def preprocess_image_v9(image):
    img = np.array(image.convert('RGB'))
    img = cv2.resize(img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_white = np.array([0, 0, 130]); upper_white = np.array([180, 80, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    result = cv2.bitwise_not(mask)
    result = cv2.copyMakeBorder(result, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
    return result

def extract_text_v9(image):
    try:
        processed_img = preprocess_image_v9(image)
        config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789: preserve_interword_spaces=1'
        text = pytesseract.image_to_string(processed_img, config=config)
        return text
    except Exception as e: return f"ERROR: {str(e)}"

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
            raw_id_str = match_id.group(); found_draw_id = int(raw_id_str[:9]); clean_line = clean_line.replace(raw_id_str, "") 
        
        raw_chunks = re.findall(r'\d+', clean_line)
        bingo_nums = []
        for chunk in raw_chunks:
            if len(chunk) > 2: 
                split_nums = [chunk[i:i+2] for i in range(0, len(chunk), 2)]
                for n_str in split_nums:
                    try: val = int(n_str); 
                    except: continue
                    if 1 <= val <= 80: bingo_nums.append(val)
            else:
                try: val = int(chunk); 
                except: continue
                if 1 <= val <= 80: bingo_nums.append(val)
        
        if len(bingo_nums) >= 15:
            unique = []; seen = set()
            for x in bingo_nums:
                if x not in seen: unique.append(x); seen.add(x)
            final_id = found_draw_id if found_draw_id > 0 else current_draw_id
            main_20 = sorted(unique[:20])
            while len(main_20) < 20: main_20.append(0)
            super_n = unique[20] if len(unique) > 20 else 0
            results.append({'draw_id': final_id, 'time': datetime.combine(selected_date, datetime.now().time()), 'nums': main_20, 'super_num': super_n})
            if found_draw_id == 0: current_draw_id -= 1
            else: current_draw_id = found_draw_id - 1
    return results

# ==============================================================================
# 3. ANALYSIS & KELLY (V11 FULL OPTION)
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
    return z_scores, z_scores[z_scores > 1.5].sort_values(ascending=False), z_scores[z_scores < -1.5].sort_values(ascending=True)

def kelly_suggestion(win_prob, odds, bankroll):
    b = odds - 1; p = win_prob; q = 1 - p
    f = (b * p - q) / b
    return max(0, f * 0.5) * 100, bankroll * max(0, f * 0.5)

def run_prediction(df, algo):
    if df.empty: return []
    recent = df.head(10)
    nums = [n for i in range(1,21) for n in recent[f'num_{i}']]
    freq = pd.Series(nums).value_counts()
    last = [df.iloc[0][f'num_{i}'] for i in range(1,21)]
    scores = {}
    for n in range(1, 81):
        if algo == "🔮 AI Master (Tổng Hợp)": s = freq.get(n, 0)*1.5 + (3.0 if n in last else 0) + random.uniform(0, 1.0)
        elif algo == "🔥 Soi Cầu Nóng (Hot)": s = freq.get(n, 0) + random.random()*0.1
        elif algo == "❄️ Soi Cầu Lạnh (Nuôi)": s = (freq.max() if not freq.empty else 0 - freq.get(n, 0)) + random.uniform(0, 1.5)
        elif algo == "♻️ Soi Cầu Bệt (Lại)": s = (1000 if n in last else 0) + freq.get(n, 0)*0.1
        else: s = 0
        scores[n] = s
    return sorted(scores, key=scores.get, reverse=True)

# ==============================================================================
# 4. DATA MANAGEMENT
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
def clear_all_data(): df = pd.DataFrame(columns=['draw_id', 'time'] + [f'num_{i}' for i in range(1, 21)] + ['super_num']); save_data(df)
def toggle_number(n): 
    if n in st.session_state.selected_nums: st.session_state.selected_nums.remove(n)
    else: st.session_state.selected_nums.append(n) if len(st.session_state.selected_nums)<20 else st.toast("Max 20!")

if 'selected_nums' not in st.session_state: st.session_state.selected_nums = []
if 'ocr_result' not in st.session_state: st.session_state.ocr_result = []
if 'predict_data' not in st.session_state: st.session_state.predict_data = None
if 'selected_algo' not in st.session_state: st.session_state.selected_algo = "🔮 AI Master (Tổng Hợp)"

# ==============================================================================
# 5. GIAO DIỆN CHÍNH
# ==============================================================================
st.title("🎲 BINGO V13 - MOBILE FINAL")
df_history = load_data()
status, msg = check_tesseract()

# --- NÚT XÓA TẤT CẢ (ĐÃ KHÔI PHỤC) ---
c_del1, c_del2 = st.columns([3, 1])
with c_del2:
    if st.button("🚨 XÓA SẠCH DỮ LIỆU", type="primary", use_container_width=True):
        clear_all_data()
        st.success("Đã xóa toàn bộ!")
        st.rerun()

with st.container(border=True):
    t1, t2 = st.tabs(["📸 QUÉT ẢNH", "🖱️ NHẬP TAY (MOBILE)"])
    
    with t1:
        c_up, c_set = st.columns([2, 1])
        with c_up: up_file = st.file_uploader("Upload ảnh:", type=['png','jpg','jpeg'])
        with c_set: 
            s_date = st.date_input("Ngày:", datetime.now())
            suggest = int(df_history['draw_id'].max()) + 1 if not df_history.empty else 114000001
            start_id = st.number_input("Mã kỳ đầu:", value=suggest, format="%d")

        if up_file and st.button("🔍 QUÉT NGAY", type="primary", use_container_width=True):
            if status:
                img = Image.open(up_file)
                with st.spinner("Đang xử lý..."):
                    raw = extract_text_v9(img)
                    res = parse_bingo_results_v9(raw, s_date, start_id)
                    if res:
                        st.session_state.ocr_result = res
                        st.markdown(f"<div class='success-msg'>✅ Tìm thấy {len(res)} dòng!</div>", unsafe_allow_html=True)
                    else: st.error("❌ Không đọc được số.")

        if st.session_state.ocr_result:
            if st.button("💾 LƯU TẤT CẢ", type="primary", use_container_width=True):
                cnt = 0
                for it in st.session_state.ocr_result:
                    if df_history.empty or it['draw_id'] not in df_history['draw_id'].values:
                        r = {'draw_id': it['draw_id'], 'time': it['time'], 'super_num': it['super_num']}
                        for k, v in enumerate(it['nums']): r[f'num_{k+1}'] = v if k<20 else 0
                        for k in range(len(it['nums']), 20): r[f'num_{k+1}'] = 0
                        df_history = pd.concat([pd.DataFrame([r]), df_history], ignore_index=True)
                        cnt+=1
                if cnt: save_data(df_history); st.success(f"Lưu {cnt} kỳ!"); st.session_state.ocr_result=[]; st.rerun()

    # --- TAB NHẬP TAY (LAYOUT 8x10 CHUẨN) ---
    with t2:
        st.caption("Chế độ nhập tay nhanh (Khít màn hình điện thoại)")
        c1, c2 = st.columns([2,1])
        nid = str(int(df_history['draw_id'].max()) + 1) if not df_history.empty else ""
        mid = c1.text_input("Mã Kỳ:", value=nid)
        if c2.button("XÓA CHỌN", type="secondary", use_container_width=True): st.session_state.selected_nums = []
        
        # --- LƯỚI SỐ 8 HÀNG x 10 CỘT ---
        # Hàng 1: 01 -> 10. Điều này làm cho Cột 1 là: 01, 11, 21, 31... đúng ý bạn.
        for r in range(8):
            cols = st.columns(10)
            for c in range(10):
                # r=0, c=0 -> n=1. r=0, c=9 -> n=10.
                # r=1, c=0 -> n=11.
                n = r*10 + c + 1
                bg = "primary" if n in st.session_state.selected_nums else "secondary"
                if cols[c].button(f"{n:02d}", key=f"b{n}", type=bg): toggle_number(n); st.rerun()
        
        st.write("")
        if st.button("💾 LƯU KẾT QUẢ", type="primary", use_container_width=True):
            r = {'draw_id': int(mid) if mid else 0, 'time': datetime.combine(datetime.now(), datetime.now().time()), 'super_num': 0}
            for i,v in enumerate(sorted(st.session_state.selected_nums)): r[f'num_{i+1}'] = v
            save_data(pd.concat([pd.DataFrame([r]), df_history], ignore_index=True)); st.success("Lưu!"); st.rerun()

# ==============================================================================
# 6. PHÂN TÍCH V11 (FULL OPTION 1-10 SAO)
# ==============================================================================
st.markdown("---")
st.header("📊 PHÂN TÍCH & DỰ ĐOÁN")

if st.button("🚀 CHẠY PHÂN TÍCH", type="primary", use_container_width=True):
    if not df_history.empty:
        st.session_state.predict_data = run_prediction(df_history, st.session_state.selected_algo)
        st.session_state.z_score_data = calculate_z_scores(df_history)
        st.toast("Xong!", icon="✅")
    else: st.error("Chưa có dữ liệu.")

if st.session_state.predict_data:
    tabs = st.tabs(["DỰ ĐOÁN", "Z-SCORE"])
    
    with tabs[0]:
        c1, c2 = st.columns([2, 1])
        with c1:
            salgo = st.selectbox("Thuật toán:", ["🔮 AI Master (Tổng Hợp)", "🔥 Soi Cầu Nóng (Hot)", "❄️ Soi Cầu Lạnh (Nuôi)", "♻️ Soi Cầu Bệt (Lại)"])
            if salgo != st.session_state.selected_algo:
                st.session_state.selected_algo = salgo
                st.session_state.predict_data = run_prediction(df_history, salgo)
                st.rerun()
                
            # --- MENU CHỌN SAO ĐẦY ĐỦ (1-10) ---
            modes = {f"{i} Tinh": i for i in range(10, 0, -1)} # 10 xuống 1
            smode_k = st.selectbox("Chọn dàn:", list(modes.keys()), index=4) # Mặc định 6 Tinh
            pick_n = modes[smode_k]
            
            fnums = sorted(list(st.session_state.predict_data)[:pick_n])
            cols = st.columns(5)
            for i, n in enumerate(fnums): 
                cols[i%5].markdown(f"<div style='background-color:{'#E74C3C' if n>40 else '#3498DB'}; color:white; padding:10px; border-radius:5px; text-align:center; font-weight:bold; font-size:20px; margin-bottom:5px'>{n:02d}</div>", unsafe_allow_html=True)
        
        with c2:
            st.subheader("💰 QUẢN LÝ VỐN")
            my_money = st.number_input("Vốn:", value=10000, step=1000)
            
            # Cấu hình Kelly
            if pick_n <= 4: ai_win = 0.55; odds_val = 2.0
            elif pick_n <= 7: ai_win = 0.35; odds_val = 4.0
            else: ai_win = 0.15; odds_val = 10.0
            
            kp, km = kelly_suggestion(ai_win, odds_val, my_money)
            if kp > 0: st.markdown(f"<div class='kelly-box'>ĐÁNH: {kp:.1f}%<br>${km:,.0f} TWD</div>", unsafe_allow_html=True)
            else: st.warning("Bảo toàn vốn")

    with tabs[1]:
        z, h, c = st.session_state.z_score_data
        st.write("**🔥 NÓNG:** " + ", ".join([f"{n}" for n in h.index[:10]]))
        st.write("**❄️ LẠNH:** " + ", ".join([f"{n}" for n in c.index[:10]]))

with st.expander("LỊCH SỬ KỲ QUAY"):
    st.dataframe(df_history, use_container_width=True)
