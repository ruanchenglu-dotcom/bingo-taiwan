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
st.set_page_config(page_title="Bingo V12 - Mobile & Columns", layout="wide")

# CSS TỐI ƯU CHO ĐIỆN THOẠI (Nút to, Khít lề)
st.markdown("""
<style>
    /* Tối ưu nút bấm cho ngón tay trên điện thoại */
    div.stButton > button:first-child { 
        min-height: 55px !important; /* Cao hơn để dễ bấm */
        width: 100% !important; 
        margin: 1px 0px !important; /* Sát nhau */
        padding: 0px !important;
        font-weight: bold; 
        border-radius: 4px; 
        font-size: 16px;
    }
    
    /* Thu hẹp khoảng cách giữa các cột để vừa khít màn hình nhỏ */
    [data-testid="column"] {
        padding: 0px 1px !important;
        min-width: 0px !important;
    }
    
    /* Ẩn bớt padding thừa của trang web */
    .block-container {
        padding-top: 1rem;
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }

    .success-msg { color: #155724; background-color: #d4edda; border-color: #c3e6cb; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
    .kelly-box { background-color: #fff8e1; padding: 10px; border-radius: 8px; border: 2px solid #f1c40f; text-align: center; font-weight: bold; font-size: 16px; }
    
    /* Box hiển thị Cột Hot */
    .col-hot-box {
        background-color: #ffcccc; 
        border-left: 5px solid #ff0000; 
        padding: 10px; 
        margin-bottom: 5px;
        color: #990000;
        font-weight: bold;
    }
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
                    try: 
                        val = int(n_str); 
                        if 1 <= val <= 80: bingo_nums.append(val)
                    except: pass
            else:
                try: 
                    val = int(chunk); 
                    if 1 <= val <= 80: bingo_nums.append(val)
                except: pass
        
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
# 3. PHÂN TÍCH CỘT (TÍNH NĂNG MỚI)
# ==============================================================================
def analyze_columns(df):
    """Phân tích hiệu suất của 10 cột (Đuôi 0-9)"""
    if df.empty: return None
    
    # Lấy 10 kỳ gần nhất để soi xu hướng
    recent = df.head(10)
    all_nums = []
    for i in range(1, 21): all_nums.extend(recent[f'num_{i}'].tolist())
    
    # Đếm số lần xuất hiện của từng đuôi (0-9)
    # Ví dụ: Số 11, 21, 31 -> Đuôi 1
    # Số 10, 20, 80 -> Đuôi 0
    tail_counts = {i: 0 for i in range(10)}
    
    for n in all_nums:
        if n > 0:
            tail = n % 10
            tail_counts[tail] += 1
            
    # Sắp xếp từ cao xuống thấp
    sorted_tails = sorted(tail_counts.items(), key=lambda item: item[1], reverse=True)
    return sorted_tails

def get_column_numbers(tail):
    """Trả về dàn 8 số của cột đó"""
    if tail == 0: return [10, 20, 30, 40, 50, 60, 70, 80]
    return [tail + 10*i for i in range(8)]

# ==============================================================================
# 4. CORE LOGIC & KELLY
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

def clear_all_data():
    """Xóa sạch sành sanh dữ liệu"""
    num_cols = [f'num_{i}' for i in range(1, 21)]
    columns = ['draw_id', 'time'] + num_cols + ['super_num']
    df = pd.DataFrame(columns=columns)
    save_data(df)
    return True

def toggle_number(n): 
    if n in st.session_state.selected_nums: st.session_state.selected_nums.remove(n)
    else: st.session_state.selected_nums.append(n) if len(st.session_state.selected_nums)<20 else st.toast("Max 20!")

def kelly_suggestion(win_prob, odds, bankroll):
    b = odds - 1; p = win_prob; q = 1 - p
    f = (b * p - q) / b
    return max(0, f * 0.5) * 100, bankroll * max(0, f * 0.5)

# Init State
if 'selected_nums' not in st.session_state: st.session_state.selected_nums = []
if 'ocr_result' not in st.session_state: st.session_state.ocr_result = []

# ==============================================================================
# 5. GIAO DIỆN CHÍNH
# ==============================================================================
st.title("🎲 BINGO V12 - MOBILE & COLUMNS")
df_history = load_data()
status, msg = check_tesseract()

# NÚT XÓA TOÀN BỘ (DANGER ZONE)
with st.expander("🗑️ QUẢN LÝ DỮ LIỆU"):
    st.warning("Nút dưới đây sẽ xóa sạch lịch sử để chơi ca mới.")
    if st.button("🚨 XÓA TẤT CẢ DỮ LIỆU", type="primary", use_container_width=True):
        clear_all_data()
        st.success("Đã xóa sạch!")
        st.rerun()

with st.container(border=True):
    t1, t2 = st.tabs(["📸 QUÉT ẢNH", "🖱️ NHẬP TAY (MOBILE)"])
    
    # --- TAB SCAN ---
    with t1:
        c_up, c_set = st.columns([2, 1])
        with c_up: up_file = st.file_uploader("Upload ảnh:", type=['png','jpg','jpeg'])
        with c_set: 
            s_date = st.date_input("Ngày:", datetime.now())
            suggest_id = int(df_history['draw_id'].max()) + 1 if not df_history.empty else 114000001
            start_id_input = st.number_input("Mã kỳ đầu:", value=suggest_id, format="%d")

        if up_file and st.button("🔍 QUÉT NGAY", type="primary", use_container_width=True):
            if status:
                img = Image.open(up_file)
                with st.spinner("Đang xử lý..."):
                    raw_txt = extract_text_v9(img)
                    res = parse_bingo_results_v9(raw_txt, s_date, start_id_input)
                    if res:
                        st.session_state.ocr_result = res
                        st.markdown(f"<div class='success-msg'>✅ Đọc được {len(res)} kỳ!</div>", unsafe_allow_html=True)
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

    # --- TAB NHẬP TAY (TỐI ƯU CHO ĐIỆN THOẠI) ---
    with t2:
        c1, c2 = st.columns([2,1])
        nid = str(int(df_history['draw_id'].max()) + 1) if not df_history.empty else ""
        mid = c1.text_input("Mã Kỳ:", value=nid)
        if c2.button("XÓA CHỌN", type="secondary", use_container_width=True): st.session_state.selected_nums = []
        
        # BÀN PHÍM SỐ SÁT NHAU CHO MOBILE
        st.markdown("---")
        for r in range(8):
            # Dùng 10 cột nhưng CSS đã ép sát lề
            cols = st.columns(10) 
            for c in range(10):
                n = r*10 + c + 1
                bg = "primary" if n in st.session_state.selected_nums else "secondary"
                # Nút bấm to và dễ chạm
                if cols[c].button(f"{n}", key=f"b{n}", type=bg): toggle_number(n); st.rerun()
        
        st.markdown("---")
        if st.button("💾 LƯU KẾT QUẢ", type="primary", use_container_width=True):
            r = {'draw_id': int(mid) if mid else 0, 'time': datetime.combine(datetime.now(), datetime.now().time()), 'super_num': 0}
            for i,v in enumerate(sorted(st.session_state.selected_nums)): r[f'num_{i+1}'] = v
            save_data(pd.concat([pd.DataFrame([r]), df_history], ignore_index=True)); st.success("Lưu!"); st.rerun()

# ==============================================================================
# 6. PHÂN TÍCH SOI CẦU CỘT (ĐUÔI) - TÍNH NĂNG MỚI
# ==============================================================================
st.markdown("---")
st.header("📊 PHÂN TÍCH & DỰ ĐOÁN")

if not df_history.empty:
    col_anal = analyze_columns(df_history)
    
    tabs = st.tabs(["📊 SOI CẦU CỘT (ĐUÔI)", "💰 QUẢN LÝ VỐN"])
    
    # TAB 1: PHÂN TÍCH CỘT
    with tabs[0]:
        st.info("💡 Mẹo: Cột (Đuôi) là chữ số cuối cùng. Ví dụ Cột 1 gồm: 01, 11, 21, ..., 71, 81.")
        
        # Lấy cột tốt nhất
        best_tail, hit_count = col_anal[0]
        column_nums = get_column_numbers(best_tail)
        
        st.markdown(f"""
        <div class='col-hot-box'>
            🔥 CỘT {best_tail} ĐANG NỔ MẠNH NHẤT!<br>
            (Xuất hiện {hit_count} lần trong 10 kỳ gần đây)<br>
            👉 Gợi ý đánh: {', '.join(map(str, column_nums))}
        </div>
        """, unsafe_allow_html=True)
        
        # Biểu đồ cột
        tails = [str(x[0]) for x in col_anal]
        counts = [x[1] for x in col_anal]
        fig = px.bar(x=tails, y=counts, labels={'x': 'Cột (Đuôi)', 'y': 'Số lần ra'}, title="Tần suất ra số theo Cột (10 kỳ gần nhất)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Hiển thị chi tiết các cột khác
        with st.expander("Xem chi tiết các cột khác"):
            for tail, count in col_anal[1:]:
                st.write(f"**Cột {tail}:** ra {count} lần - {get_column_numbers(tail)}")

    # TAB 2: KELLY
    with tabs[1]:
        my_money = st.number_input("Vốn hiện có:", value=10000, step=1000)
        # Giả định đánh dàn 8 số (Cột) thì tỷ lệ trúng khoảng 40-50% nhưng ăn ít hơn 6 tinh
        kp, km = kelly_suggestion(0.45, 3.0, my_money)
        st.markdown(f"<div class='kelly-box'>💡 GỢI Ý ĐÁNH CỘT:<br><span style='color:#e67e22'>{kp:.1f}% Vốn</span><br><span style='color:#27ae60'>${km:,.0f} TWD</span></div>", unsafe_allow_html=True)

else:
    st.warning("Chưa có dữ liệu lịch sử.")

with st.expander("LỊCH SỬ KỲ QUAY"):
    st.dataframe(df_history, use_container_width=True)
