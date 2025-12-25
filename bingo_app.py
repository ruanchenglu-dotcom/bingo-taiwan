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
    page_title="Bingo Quantum AI - V4 Final", 
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
# 2. XỬ LÝ DỮ LIỆU
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
# 3. CÔNG NGHỆ XỬ LÝ ẢNH V4 (TÁCH NỀN TRẮNG)
# ==============================================================================
def preprocess_image_v4(image):
    """
    Kỹ thuật tách ngưỡng cao (High Thresholding) để loại bỏ bóng màu.
    Chỉ giữ lại phần text màu trắng sáng nhất.
    """
    # 1. Chuyển sang ảnh OpenCV
    img = np.array(image.convert('RGB'))
    
    # 2. Phóng to ảnh (Upscale) gấp 2 lần để số rõ hơn
    height, width = img.shape[:2]
    img = cv2.resize(img, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
    
    # 3. Chuyển sang ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 4. Tách màu trắng:
    # Số màu trắng có giá trị màu ~255. Bóng màu xanh/đỏ/xám có giá trị thấp hơn (<150).
    # Ta đặt ngưỡng 180 để loại bỏ tất cả bóng màu, chỉ giữ lại số.
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    # 5. Đảo ngược màu (Tesseract thích chữ Đen trên nền Trắng)
    # Lúc này: Số thành màu Đen, Nền (bóng đã bị xóa) thành màu Trắng.
    result = cv2.bitwise_not(thresh)
    
    return result

def extract_text_v4(image):
    try:
        processed_img = preprocess_image_v4(image)
        # psm 6: Đọc theo khối văn bản thống nhất (dạng bảng)
        # whitelist: Chỉ cho phép đọc số và dấu hai chấm (cho giờ)
        config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789: '
        text = pytesseract.image_to_string(processed_img, config=config)
        return text
    except Exception as e:
        return ""

def parse_bingo_results(text, selected_date):
    results = []
    # Vệ sinh text cơ bản
    text = text.replace('O', '0').replace('o', '0').replace('l', '1').replace('I', '1')
    
    # Tìm mã kỳ (114xxxxxx)
    matches = list(re.finditer(r'114\d{6}', text))
    
    for i in range(len(matches)):
        try:
            did_str = matches[i].group()
            did = int(did_str)
            
            # Vùng text của kỳ này
            s = matches[i].end()
            e = matches[i+1].start() if i + 1 < len(matches) else len(text)
            seg = text[s:e]
            
            # Lấy tất cả số tìm được trong vùng, giữ nguyên thứ tự
            raw_nums = re.findall(r'\b\d{1,2}\b', seg)
            
            valid_nums = []
            for n in raw_nums:
                v = int(n)
                if 1 <= v <= 80: valid_nums.append(v)
            
            # Logic tách số: 20 số đầu là chính, số thứ 21 là siêu cấp
            if len(valid_nums) >= 20:
                main_20 = sorted(list(set(valid_nums[:20])))
                # Bù số nếu thiếu (do lọc trùng)
                while len(main_20) < 20: main_20.append(0)
                
                # Số siêu cấp
                super_n = valid_nums[20] if len(valid_nums) > 20 else (main_20[-1] if main_20[-1]!=0 else 0)
                
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
st.title("🎲 BINGO QUANTUM - V4 FINAL")
df_history = load_data()

with st.container(border=True):
    t1, t2, t3 = st.tabs(["📸 QUÉT ẢNH (V4)", "🖱️ NHẬP TAY", "📋 DÁN"])
    
    # --- TAB SCAN V4 ---
    with t1:
        st.caption("Công nghệ tách nền trắng - Chuyên trị ảnh tối màu.")
        up_file = st.file_uploader("Upload ảnh:", type=['png','jpg','jpeg'])
        s_date = st.date_input("Ngày:", datetime.now())
        
        if up_file and st.button("🔍 QUÉT NGAY", type="primary"):
            img = Image.open(up_file)
            st.image(img, caption='Ảnh gốc', width=400)
            
            with st.spinner("Đang tách màu và đọc số..."):
                raw_txt = extract_text_v4(img)
                # st.code(raw_txt) # Debug text
                res = parse_bingo_results(raw_txt, s_date)
                
                if res:
                    st.session_state['ocr_result'] = res
                    st.success(f"Đọc được {len(res)} kỳ!")
                else:
                    st.error("Không đọc được. Hãy thử cắt ảnh sát bảng số hơn.")

        if st.session_state['ocr_result']:
            st.write("### 📝 Kiểm tra kết quả:")
            for i, it in enumerate(st.session_state['ocr_result']):
                with st.expander(f"Kỳ {it['draw_id']} - SC: {it['super_num']}", expanded=True):
                    c1, c2 = st.columns([4, 1])
                    n_str = ", ".join(map(str, it['nums']))
                    new_n = c1.text_area(f"Dãy số:", n_str, key=f"n_{i}")
                    new_s = c2.number_input(f"Siêu Cấp:", value=it['super_num'], key=f"s_{i}")
                    
                    # Update lại
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
