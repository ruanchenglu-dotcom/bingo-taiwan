import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import re
from collections import Counter
from datetime import datetime
import plotly.express as px
from PIL import Image, ImageOps
import pytesseract
import cv2

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG
# ==============================================================================
st.set_page_config(
    page_title="Bingo Quantum AI - V5 HSV", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    div.stButton > button:first-child { min-height: 65px; width: 100%; margin: 0px 1px; font-weight: bold; border-radius: 6px; font-size: 18px; }
    .anomaly-box-hot { background-color: #ffe6e6; padding: 10px; border-radius: 5px; border-left: 5px solid #ff4b4b; color: #c0392b;}
    .anomaly-box-cold { background-color: #e8f8f5; padding: 10px; border-radius: 5px; border-left: 5px solid #1abc9c; color: #16a085;}
    .kelly-box { background-color: #fff8e1; padding: 15px; border-radius: 8px; border: 2px solid #f1c40f; text-align: center; font-weight: bold; font-size: 18px; }
</style>
""", unsafe_allow_html=True)

DATA_FILE = 'bingo_history.csv'

# ==============================================================================
# 2. QUẢN LÝ DỮ LIỆU
# ==============================================================================
if 'selected_nums' not in st.session_state: st.session_state['selected_nums'] = [] 
if 'ocr_result' not in st.session_state: st.session_state['ocr_result'] = [] 
if 'predict_data' not in st.session_state: st.session_state['predict_data'] = None
if 'z_score_data' not in st.session_state: st.session_state['z_score_data'] = None

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

def toggle_number(num):
    if num in st.session_state['selected_nums']: st.session_state['selected_nums'].remove(num)
    else:
        if len(st.session_state['selected_nums']) < 20: st.session_state['selected_nums'].append(num)
        else: st.toast("⚠️ Max 20 số!", icon="🚫")

# ==============================================================================
# 3. CÔNG NGHỆ XỬ LÝ ẢNH V5 (HSV COLOR FILTER)
# ==============================================================================
def preprocess_image_v5(image, debug_mode=False):
    """
    Chiến thuật V5: Chuyển sang hệ màu HSV và lọc bỏ màu sắc.
    Chỉ giữ lại điểm ảnh có độ bão hòa (Saturation) thấp và độ sáng (Value) cao.
    """
    # 1. Convert PIL to OpenCV
    img = np.array(image.convert('RGB'))
    
    # 2. Upscale (Phóng to 2x)
    img = cv2.resize(img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    
    # 3. Chuyển sang không gian màu HSV (Hue, Saturation, Value)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # 4. TẠO MẶT NẠ (MASK) ĐỂ LỌC SỐ TRẮNG
    # Màu trắng có đặc điểm: Saturation (Độ đậm màu) rất thấp, Value (Độ sáng) rất cao.
    # Ngọn lửa/Bóng màu: Saturation rất cao -> Sẽ bị loại bỏ.
    
    # Ngưỡng dưới: S=0 (không màu), V=130 (khá sáng)
    lower_white = np.array([0, 0, 130]) 
    # Ngưỡng trên: H=180 (mọi màu), S=60 (chỉ chấp nhận hơi ám màu tí xíu), V=255 (sáng nhất)
    upper_white = np.array([180, 80, 255])
    
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # 5. Khử nhiễu (Dọn sạch các đốm trắng nhỏ do viền bóng tạo ra)
    # Morphological Opening
    kernel = np.ones((2,2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 6. Đảo ngược để chữ Đen nền Trắng (Tesseract thích cái này)
    result = cv2.bitwise_not(mask)
    
    # 7. Thêm viền trắng xung quanh để số không bị sát mép
    result = cv2.copyMakeBorder(result, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)

    return result

def extract_text_v5(image, debug_mode=False):
    try:
        processed_img = preprocess_image_v5(image, debug_mode)
        
        # Nếu bật chế độ Debug, hiển thị ảnh đã xử lý ra màn hình
        if debug_mode:
            st.image(processed_img, caption="Ảnh máy tính 'nhìn thấy' (Sau khi lọc màu)", use_container_width=True)
            
        config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789: '
        text = pytesseract.image_to_string(processed_img, config=config)
        return text
    except Exception as e:
        return f"Error: {e}"

def parse_bingo_results(text, selected_date):
    results = []
    # Vệ sinh text cực mạnh
    text = text.replace('O', '0').replace('o', '0').replace('l', '1').replace('I', '1').replace('|', '1')
    text = text.replace('S', '5').replace('B', '8').replace('G', '6')
    
    # Tìm mã kỳ 114...
    matches = list(re.finditer(r'114\d{6}', text))
    
    for i in range(len(matches)):
        try:
            did_str = matches[i].group()
            did = int(did_str)
            
            s = matches[i].end()
            e = matches[i+1].start() if i + 1 < len(matches) else len(text)
            seg = text[s:e]
            
            raw_nums = re.findall(r'\b\d{1,2}\b', seg)
            
            valid_nums = []
            for n in raw_nums:
                v = int(n)
                if 1 <= v <= 80: valid_nums.append(v)
            
            if len(valid_nums) >= 15: # Chấp nhận nếu đọc được ít nhất 15 số
                # Tách số siêu cấp (số cuối cùng)
                # Logic: Nếu đọc đủ 21 số trở lên thì số cuối là siêu cấp
                # Nếu chỉ đọc 20 số, có thể số siêu cấp bị sót, tạm lấy số cuối
                
                # Để an toàn: Lấy 20 số đầu tiên làm main, số thứ 21 (nếu có) là super
                main_temp = []
                super_n = 0
                
                # Loại bỏ trùng lặp nhưng giữ thứ tự
                seen = set()
                ordered_unique = []
                for x in valid_nums:
                    if x not in seen:
                        ordered_unique.append(x)
                        seen.add(x)
                
                if len(ordered_unique) >= 20:
                    main_temp = ordered_unique[:20]
                    if len(ordered_unique) > 20:
                        super_n = ordered_unique[20]
                    else:
                        super_n = 0 # Thiếu số siêu cấp
                else:
                    main_temp = ordered_unique # Lấy hết
                    super_n = 0
                
                # Sort lại 20 số chính
                main_20 = sorted(main_temp)
                while len(main_20) < 20: main_20.append(0)
                
                results.append({
                    'draw_id': did,
                    'time': datetime.combine(selected_date, datetime.now().time()),
                    'nums': main_20,
                    'super_num': super_n
                })
        except: continue
    return results

# ==============================================================================
# 4. MODULE PHÂN TÍCH (QUANT)
# ==============================================================================
def calculate_z_scores(df):
    if df.empty: return None, [], []
    recent = df.head(30)
    all_nums = []
    for i in range(1, 21): all_nums.extend(recent[f'num_{i}'].tolist())
    counts = pd.Series(all_nums).value_counts().reindex(range(1, 81), fill_value=0)
    mean = counts.mean(); std = counts.std()
    z = (counts - mean) / std
    return z, z[z > 1.5].sort_values(ascending=False), z[z < -1.5].sort_values(ascending=True)

def kelly_suggestion(win_prob, odds, bankroll):
    f = ((odds - 1) * win_prob - (1 - win_prob)) / (odds - 1)
    return max(0, f * 0.5) * 100, bankroll * max(0, f * 0.5)

def run_prediction(df, algo):
    if df.empty: return []
    recent = df.head(10)
    nums = [n for i in range(1,21) for n in recent[f'num_{i}']]
    freq = pd.Series(nums).value_counts()
    last = [df.iloc[0][f'num_{i}'] for i in range(1,21)]
    scores = {}
    for n in range(1, 81):
        if algo == "🔮 AI Master": 
            s = freq.get(n,0)*1.5 + (3.0 if n in last else 0) + random.random()
        else: s = freq.get(n,0) + random.random()
        scores[n] = s
    return sorted(scores, key=scores.get, reverse=True)

# ==============================================================================
# 5. GIAO DIỆN NGƯỜI DÙNG
# ==============================================================================
st.title("🎲 BINGO QUANTUM - V5 HSV")
df_history = load_data()

with st.container(border=True):
    t1, t2, t3 = st.tabs(["📸 QUÉT ẢNH (V5)", "🖱️ NHẬP TAY", "📋 DÁN"])
    
    # --- TAB SCAN V5 ---
    with t1:
        st.caption("Công nghệ HSV Filter: Lọc bỏ bóng màu và lửa, chỉ giữ số trắng.")
        
        c_scan1, c_scan2 = st.columns([2, 1])
        with c_scan1:
            up_file = st.file_uploader("Upload ảnh:", type=['png','jpg','jpeg'])
            s_date = st.date_input("Ngày:", datetime.now())
        with c_scan2:
            st.write("")
            st.write("")
            debug_chk = st.checkbox("🛠 Chế độ Debug (Xem ảnh máy đọc)")
            st.caption("Bật cái này lên để xem tại sao máy đọc sai (nếu có).")
        
        if up_file and st.button("🔍 QUÉT NGAY (V5)", type="primary"):
            img = Image.open(up_file)
            st.image(img, caption='Ảnh gốc', width=400)
            
            with st.spinner("Đang lọc quang phổ HSV..."):
                raw_txt = extract_text_v5(img, debug_chk) # Truyền biến debug vào
                res = parse_bingo_results(raw_txt, s_date)
                
                if res:
                    st.session_state['ocr_result'] = res
                    st.success(f"Tuyệt vời! Đọc được {len(res)} kỳ.")
                else:
                    st.error("Vẫn chưa đọc được. Hãy bật 'Chế độ Debug' xem ảnh bị đen hay trắng quá không?")

        if st.session_state['ocr_result']:
            st.write("### 📝 Kiểm tra kết quả:")
            for i, it in enumerate(st.session_state['ocr_result']):
                with st.expander(f"Kỳ {it['draw_id']} - SC: {it['super_num']}", expanded=True):
                    c1, c2 = st.columns([4, 1])
                    n_str = ", ".join(map(str, it['nums']))
                    new_n = c1.text_area(f"Dãy số:", n_str, key=f"n_{i}")
                    new_s = c2.number_input(f"Siêu Cấp:", value=it['super_num'], key=f"s_{i}")
                    
                    try:
                        st.session_state['ocr_result'][i]['nums'] = sorted([int(x) for x in new_n.split(',') if x.strip().isdigit()])
                        st.session_state['ocr_result'][i]['super_num'] = new_s
                    except: pass
            
            if st.button("💾 LƯU VÀO LỊCH SỬ"):
                add_cnt = 0
                for it in st.session_state['ocr_result']:
                    if df_history.empty or it['draw_id'] not in df_history['draw_id'].values:
                        r = {'draw_id': it['draw_id'], 'time': it['time'], 'super_num': it['super_num']}
                        for k, v in enumerate(it['nums']): 
                            if k < 20: r[f'num_{k+1}'] = v
                        for k in range(len(it['nums']), 20): r[f'num_{k+1}'] = 0
                        df_history = pd.concat([pd.DataFrame([r]), df_history], ignore_index=True)
                        add_cnt += 1
                if add_cnt: save_data(df_history); st.success(f"Đã lưu {add_cnt} kỳ!"); st.session_state['ocr_result']=[]; st.rerun()
                else: st.warning("Dữ liệu đã tồn tại!")

    # --- TAB NHẬP TAY ---
    with t2:
        c1, c2, c3 = st.columns([2,2,1])
        nid = str(int(df_history['draw_id'].max()) + 1) if not df_history.empty else ""
        mid = c1.text_input("Mã Kỳ:", value=nid)
        mdate = c2.date_input("Ngày:", datetime.now(), key="d2")
        if c3.button("Xóa"): st.session_state['selected_nums'] = []
        
        st.markdown(f"**Chọn: {len(st.session_state['selected_nums'])}/20**")
        for r in range(8):
            cols = st.columns(10)
            for c in range(10):
                n = r*10 + c + 1
                bg = "primary" if n in st.session_state['selected_nums'] else "secondary"
                if cols[c].button(f"{n:02d}", key=f"b{n}", type=bg): toggle_number(n); st.rerun()
        
        sup = st.selectbox("Siêu Cấp:", sorted(st.session_state['selected_nums']) if st.session_state['selected_nums'] else range(1,81))
        if st.button("LƯU TAY", type="primary"):
            r = {'draw_id': int(mid) if mid else 0, 'time': datetime.combine(mdate, datetime.now().time()), 'super_num': sup}
            for i,v in enumerate(sorted(st.session_state['selected_nums'])): r[f'num_{i+1}'] = v
            save_data(pd.concat([pd.DataFrame([r]), df_history], ignore_index=True)); st.success("Lưu!"); st.rerun()

    # --- TAB DÁN ---
    with t3:
        txt = st.text_area("Dán text:", height=150)
        if st.button("XỬ LÝ TEXT"):
            res = parse_bingo_results(txt, datetime.now())
            if res:
                cnt = 0
                for it in res:
                    if df_history.empty or it['draw_id'] not in df_history['draw_id'].values:
                        r = {'draw_id': it['draw_id'], 'time': it['time'], 'super_num': it['super_num']}
                        for k,v in enumerate(it['nums']): r[f'num_{k+1}'] = v
                        df_history = pd.concat([pd.DataFrame([r]), df_history], ignore_index=True); cnt+=1
                if cnt: save_data(df_history); st.success(f"Thêm {cnt} kỳ!"); st.rerun()

# --- PHÂN TÍCH ---
st.markdown("---")
if st.button("🚀 PHÂN TÍCH Z-SCORE & KELLY", type="primary"):
    st.session_state['predict_data'] = run_prediction(df_history, "🔮 AI Master")
    st.session_state['z_score_data'] = calculate_z_scores(df_history)

if st.session_state['predict_data']:
    z, hot, cold = st.session_state['z_score_data']
    c1, c2 = st.columns(2)
    with c1:
        st.write("#### 🔥 SỐ NÓNG (Z>1.5)")
        if not hot.empty: 
            for n,s in hot.items(): st.markdown(f"<div class='anomaly-box-hot'>🔴 {n:02d} (Z:{s:.2f})</div>", unsafe_allow_html=True)
    with c2:
        st.write("#### ❄️ SỐ LẠNH (Z<-1.5)")
        if not cold.empty:
            for n,s in cold.items(): st.markdown(f"<div class='anomaly-box-cold'>🔵 {n:02d} (Z:{s:.2f})</div>", unsafe_allow_html=True)
            
    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        st.write("#### 💰 KELLY (Vốn 10k, Win 55%)")
        kp, km = kelly_suggestion(0.55, 2.0, 10000)
        st.markdown(f"<div class='kelly-box'>{kp:.1f}% Vốn<br>${km:,.0f} TWD</div>", unsafe_allow_html=True)
    with c4:
        st.write("#### 🎯 DỰ ĐOÁN 10 SỐ")
        top10 = list(st.session_state['predict_data'])[:10]
        st.write(", ".join([f"{x:02d}" for x in sorted(top10)]))

with st.expander("Lịch sử"):
    if st.button("Xóa cuối"): delete_last_row(); st.rerun()
    st.dataframe(df_history, use_container_width=True)
