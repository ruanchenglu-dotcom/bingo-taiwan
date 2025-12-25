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
# 1. CẤU HÌNH & KIỂM TRA HỆ THỐNG
# ==============================================================================
st.set_page_config(page_title="Bingo AI - V6 System Check", layout="wide")

st.markdown("""
<style>
    div.stButton > button:first-child { min-height: 65px; width: 100%; margin: 0px 1px; font-weight: bold; border-radius: 6px; font-size: 18px; }
    .kelly-box { background-color: #fff8e1; padding: 15px; border-radius: 8px; border: 2px solid #f1c40f; text-align: center; font-weight: bold; font-size: 18px; }
</style>
""", unsafe_allow_html=True)

DATA_FILE = 'bingo_history.csv'

# --- KIỂM TRA TESSERACT (QUAN TRỌNG) ---
def check_tesseract():
    # Kiểm tra xem phần mềm Tesseract đã được cài trong Linux chưa
    path = shutil.which("tesseract")
    if path is None:
        return False, "❌ LỖI: Chưa tìm thấy Tesseract! Bạn đã quên tạo file 'packages.txt' trên GitHub chưa?"
    return True, f"✅ Hệ thống OK: Tesseract đang chạy tại {path}"

# ==============================================================================
# 2. XỬ LÝ ẢNH & OCR
# ==============================================================================
def preprocess_image_v6(image):
    """Xử lý ảnh: Lọc màu, giữ số đen trên nền trắng"""
    img = np.array(image.convert('RGB'))
    img = cv2.resize(img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Lọc lấy màu trắng (Số)
    lower_white = np.array([0, 0, 140]) 
    upper_white = np.array([180, 80, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Làm sạch nhiễu
    kernel = np.ones((2,2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Đảo màu (Chữ đen nền trắng)
    result = cv2.bitwise_not(mask)
    result = cv2.copyMakeBorder(result, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    return result

def extract_text_v6(image):
    try:
        processed_img = preprocess_image_v6(image)
        # Hiển thị ảnh đã xử lý để debug
        st.image(processed_img, caption="Ảnh máy tính nhìn thấy (Đã lọc màu)", width=400)
        
        # Cấu hình OCR
        config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789: '
        text = pytesseract.image_to_string(processed_img, config=config)
        return text
    except Exception as e:
        return f"ERROR_OCR: {str(e)}"

def parse_bingo_results(text, selected_date):
    results = []
    # Vệ sinh text
    text = text.replace('O', '0').replace('o', '0').replace('l', '1').replace('I', '1').replace('|', '1')
    
    # Tìm mã kỳ (114xxxxxx)
    matches = list(re.finditer(r'114\d{6}', text))
    
    for i in range(len(matches)):
        try:
            did = int(matches[i].group())
            s = matches[i].end()
            e = matches[i+1].start() if i+1 < len(matches) else len(text)
            seg = text[s:e]
            
            # Tìm số
            raw_nums = re.findall(r'\b\d{1,2}\b', seg)
            valid_nums = [int(n) for n in raw_nums if 1 <= int(n) <= 80]
            
            if len(valid_nums) >= 15:
                # Logic tách số siêu cấp
                # Lấy 20 số duy nhất đầu tiên làm main
                unique = []
                seen = set()
                for x in valid_nums:
                    if x not in seen:
                        unique.append(x)
                        seen.add(x)
                
                main_20 = sorted(unique[:20])
                while len(main_20) < 20: main_20.append(0)
                
                # Số siêu cấp là số thứ 21 (nếu có)
                super_n = unique[20] if len(unique) > 20 else 0
                
                results.append({
                    'draw_id': did,
                    'time': datetime.combine(selected_date, datetime.now().time()),
                    'nums': main_20,
                    'super_num': super_n
                })
        except: continue
    return results

# ==============================================================================
# 3. CORE LOGIC & UI
# ==============================================================================
# ... (Phần Load/Save Data giữ nguyên) ...
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

def save_data(df): df.sort_values(by='draw_id', ascending=False).to_csv(DATA_FILE, index=False)
def delete_last_row(): df = load_data(); df=df.iloc[1:] if not df.empty else df; save_data(df); return True
def toggle_number(n): 
    if n in st.session_state.selected_nums: st.session_state.selected_nums.remove(n)
    else: st.session_state.selected_nums.append(n) if len(st.session_state.selected_nums)<20 else st.toast("Max 20!")

# Init State
if 'selected_nums' not in st.session_state: st.session_state.selected_nums = []
if 'ocr_result' not in st.session_state: st.session_state.ocr_result = []

# --- MAIN UI ---
st.title("🎲 BINGO V6 - HỆ THỐNG KIỂM TRA")
df_history = load_data()

# CHECK HỆ THỐNG
status, msg = check_tesseract()
if not status:
    st.error(msg)
    st.info("💡 Hướng dẫn sửa: Vào GitHub > Tạo file 'packages.txt' > Viết chữ 'tesseract-ocr' vào đó > Commit > Reboot App.")
else:
    st.success(msg)

with st.container(border=True):
    t1, t2 = st.tabs(["📸 QUÉT ẢNH (V6)", "📋 DỮ LIỆU THÔ (DEBUG)"])
    
    with t1:
        up_file = st.file_uploader("Upload ảnh:", type=['png','jpg','jpeg'])
        s_date = st.date_input("Ngày:", datetime.now())
        
        if up_file and st.button("🔍 QUÉT NGAY"):
            if not status:
                st.error("Không thể quét vì thiếu Tesseract!")
            else:
                img = Image.open(up_file)
                st.image(img, caption='Ảnh gốc', width=400)
                
                with st.spinner("Đang xử lý..."):
                    raw_txt = extract_text_v6(img)
                    st.session_state['raw_text_debug'] = raw_txt # Lưu lại để soi
                    
                    if "ERROR" in raw_txt:
                        st.error(f"Lỗi phần mềm: {raw_txt}")
                    else:
                        res = parse_bingo_results(raw_txt, s_date)
                        if res:
                            st.session_state.ocr_result = res
                            st.success(f"Đọc được {len(res)} kỳ!")
                        else:
                            st.warning("Ảnh đã xử lý tốt, nhưng không tìm thấy mã kỳ 114xxxxxx. Hãy xem tab 'Dữ liệu thô' để biết tại sao.")

        # Hiển thị kết quả & Lưu
        if st.session_state.ocr_result:
            for i, it in enumerate(st.session_state.ocr_result):
                with st.expander(f"Kỳ {it['draw_id']} - SC: {it['super_num']}", expanded=True):
                    c1, c2 = st.columns([4,1])
                    n_str = c1.text_area("Số:", ", ".join(map(str, it['nums'])), key=f"n{i}")
                    s_num = c2.number_input("Siêu cấp:", value=it['super_num'], key=f"s{i}")
                    # Update logic (Simplified)
                    try:
                        st.session_state.ocr_result[i]['nums'] = sorted([int(x) for x in n_str.split(',') if x.strip().isdigit()])
                        st.session_state.ocr_result[i]['super_num'] = s_num
                    except: pass
            
            if st.button("💾 LƯU KẾT QUẢ"):
                cnt = 0
                for it in st.session_state.ocr_result:
                    if df_history.empty or it['draw_id'] not in df_history['draw_id'].values:
                        r = {'draw_id': it['draw_id'], 'time': it['time'], 'super_num': it['super_num']}
                        for k, v in enumerate(it['nums']): r[f'num_{k+1}'] = v if k<20 else 0
                        for k in range(len(it['nums']), 20): r[f'num_{k+1}'] = 0
                        df_history = pd.concat([pd.DataFrame([r]), df_history], ignore_index=True)
                        cnt+=1
                if cnt: save_data(df_history); st.success(f"Lưu {cnt} kỳ!"); st.session_state.ocr_result=[]; st.rerun()
                else: st.warning("Dữ liệu đã có!")

    with t2:
        st.write("### 🕵️‍♂️ Máy tính đọc được gì?")
        if 'raw_text_debug' in st.session_state:
            st.code(st.session_state['raw_text_debug'])
            st.caption("Nếu bạn thấy số ở đây mà App không nhận, nghĩa là định dạng mã kỳ bị sai.")
        else:
            st.info("Chưa có dữ liệu. Hãy quét ảnh trước.")

# --- PHẦN LỊCH SỬ (GIỮ NGUYÊN) ---
st.markdown("---")
with st.expander("Lịch sử"):
    st.dataframe(df_history, use_container_width=True)
