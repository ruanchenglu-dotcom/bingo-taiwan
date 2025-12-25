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
st.set_page_config(page_title="Bingo AI - V9 Freedom", layout="wide")

st.markdown("""
<style>
    div.stButton > button:first-child { min-height: 65px; width: 100%; margin: 0px 1px; font-weight: bold; border-radius: 6px; font-size: 18px; }
    .raw-text-box { background-color: #f8f9fa; border: 1px solid #ddd; padding: 10px; font-family: monospace; font-size: 12px; height: 100px; overflow-y: scroll; white-space: pre-wrap;}
    .success-msg { color: #155724; background-color: #d4edda; border-color: #c3e6cb; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

DATA_FILE = 'bingo_history.csv'

def check_tesseract():
    path = shutil.which("tesseract")
    if path is None: return False, "❌ LỖI: Chưa cài Tesseract!"
    return True, "✅ System OK"

# ==============================================================================
# 2. XỬ LÝ ẢNH (V9 - GIỮ NGUYÊN BỘ LỌC TỐT NHẤT)
# ==============================================================================
def preprocess_image_v9(image):
    # Upscale & HSV Filter (Lọc trắng)
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
        # Hiển thị ảnh debug nhỏ
        st.image(processed_img, caption="Ảnh máy tính đọc (Chữ đen nền trắng)", width=400)
        
        # Cấu hình OCR
        config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789: preserve_interword_spaces=1'
        text = pytesseract.image_to_string(processed_img, config=config)
        return text
    except Exception as e:
        return f"ERROR: {str(e)}"

# ==============================================================================
# 3. BỘ PHÂN TÍCH V9 (KHÔNG CẦN MÃ KỲ)
# ==============================================================================
def parse_bingo_results_v9(text, selected_date, start_draw_id):
    results = []
    lines = text.split('\n')
    
    # Dùng biến đếm để tự sinh mã kỳ nếu không tìm thấy
    current_draw_id = start_draw_id
    
    for line in lines:
        if not line.strip(): continue
        
        # 1. Vệ sinh dòng chữ
        clean_line = line.replace('O', '0').replace('o', '0').replace('l', '1').replace('I', '1').replace('|', '1').replace('S','5')
        
        # 2. TÌM MÃ KỲ CỨNG (Nếu có)
        match_id = re.search(r'114\d{6,}', clean_line)
        found_draw_id = 0
        
        if match_id:
            raw_id_str = match_id.group()
            found_draw_id = int(raw_id_str[:9])
            clean_line = clean_line.replace(raw_id_str, "") # Xóa đi để không lẫn vào số
        
        # 3. TÁCH SỐ (LOGIC CẮT CHUỖI V8)
        raw_chunks = re.findall(r'\d+', clean_line)
        bingo_nums = []
        for chunk in raw_chunks:
            if len(chunk) > 2: # Cắt chuỗi dính 040915...
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
        
        # 4. QUYẾT ĐỊNH: ĐÂY CÓ PHẢI LÀ DÒNG KẾT QUẢ KHÔNG?
        # Điều kiện lỏng hơn: Chỉ cần tìm thấy >= 15 số hợp lệ
        if len(bingo_nums) >= 15:
            
            # Lọc trùng
            unique = []
            seen = set()
            for x in bingo_nums:
                if x not in seen:
                    unique.append(x)
                    seen.add(x)
            
            # Xử lý Mã kỳ: Nếu tìm thấy trong ảnh thì dùng, không thì dùng mã tự điền
            final_id = found_draw_id if found_draw_id > 0 else current_draw_id
            
            # Tách số siêu cấp
            main_20 = sorted(unique[:20])
            while len(main_20) < 20: main_20.append(0)
            super_n = unique[20] if len(unique) > 20 else 0
            
            results.append({
                'draw_id': final_id,
                'time': datetime.combine(selected_date, datetime.now().time()),
                'nums': main_20,
                'super_num': super_n
            })
            
            # Nếu dùng mã tự điền, thì dòng tiếp theo sẽ là mã nhỏ hơn (trừ lùi)
            if found_draw_id == 0:
                current_draw_id -= 1
            else:
                # Nếu tìm thấy mã thật, cập nhật lại dòng chảy cho dòng sau
                current_draw_id = found_draw_id - 1
            
    return results

# ==============================================================================
# 4. LOGIC CŨ (GIỮ NGUYÊN)
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

# --- UI ---
st.title("🎲 BINGO V9 - FREEDOM MODE")
df_history = load_data()
status, msg = check_tesseract()

with st.container(border=True):
    t1, t2 = st.tabs(["📸 QUÉT ẢNH (LINH HOẠT)", "⚙️ NHẬP LIỆU"])
    
    with t1:
        st.info("💡 Mẹo: Bạn chỉ cần chụp phần chứa 20 con số. Không cần chụp Mã Kỳ hay Giờ nữa.")
        
        c_up, c_setting = st.columns([2, 1])
        with c_up:
            up_file = st.file_uploader("Upload ảnh:", type=['png','jpg','jpeg'])
        with c_setting:
            s_date = st.date_input("Ngày:", datetime.now())
            # Tự động lấy mã kỳ lớn nhất trong lịch sử + 1 để gợi ý
            suggest_id = int(df_history['draw_id'].max()) + 1 if not df_history.empty else 114000001
            start_id_input = st.number_input("Mã kỳ dòng đầu tiên (Nếu ảnh mất mã):", value=suggest_id, step=1, format="%d")

        if up_file and st.button("🔍 QUÉT NGAY"):
            if status:
                img = Image.open(up_file)
                
                with st.spinner("Đang tách số khỏi ảnh..."):
                    raw_txt = extract_text_v9(img)
                    st.caption("Dữ liệu thô máy đọc được:")
                    st.markdown(f"<div class='raw-text-box'>{raw_txt}</div>", unsafe_allow_html=True)
                    
                    # Truyền mã kỳ người dùng nhập vào để tự điền nếu thiếu
                    res = parse_bingo_results_v9(raw_txt, s_date, start_id_input)
                    
                    if res:
                        st.session_state.ocr_result = res
                        st.markdown(f"<div class='success-msg'>✅ Tìm thấy {len(res)} dòng số! Hãy kiểm tra Mã Kỳ bên dưới.</div>", unsafe_allow_html=True)
                    else:
                        st.error("❌ Không tìm thấy dãy số nào (cần ít nhất 15 số/dòng). Ảnh có thể quá mờ.")

        if st.session_state.ocr_result:
            st.write("### 👇 KIỂM TRA & LƯU:")
            for i, it in enumerate(st.session_state.ocr_result):
                with st.expander(f"Kỳ {it['draw_id']} (Đã tách số)", expanded=True):
                    c1, c2, c3 = st.columns([1, 3, 1])
                    # Cho phép sửa Mã Kỳ nếu máy điền sai
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
        # Nhập tay & Dán
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

# --- ANALYSIS (GIỮ NGUYÊN) ---
st.markdown("---")
# (Phần phân tích giữ nguyên như cũ)
# ...

with st.expander("Lịch sử"):
    st.dataframe(df_history, use_container_width=True)
