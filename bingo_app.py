import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import re
from datetime import datetime

# ==============================================================================
# 1. CẤU HÌNH TRANG WEB & FILE DỮ LIỆU
# ==============================================================================
st.set_page_config(
    page_title="Bingo Mobile VIP Pro", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# Tên file lưu trữ lịch sử
DATA_FILE = 'bingo_history.csv'

# ==============================================================================
# 2. CÁC HÀM QUẢN LÝ DỮ LIỆU (ĐỌC, GHI, XÓA)
# ==============================================================================
def load_data():
    """
    Hàm đọc dữ liệu từ file CSV.
    Khởi tạo đầy đủ 20 cột số để không bị lỗi hiển thị.
    """
    num_cols = [f'num_{i}' for i in range(1, 21)]
    columns = ['draw_id', 'time'] + num_cols + ['super_num']
    df = pd.DataFrame(columns=columns)
    
    if os.path.exists(DATA_FILE):
        try:
            loaded_df = pd.read_csv(DATA_FILE)
            if not loaded_df.empty: 
                df = loaded_df
        except Exception: 
            pass
    
    # Chuẩn hóa cột thời gian
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
    
    # Sắp xếp dữ liệu mới nhất lên trên cùng
    df = df.dropna(subset=['time'])
    df = df.sort_values(by='time', ascending=False)
    # Loại bỏ trùng lặp mã kỳ quay
    df = df.drop_duplicates(subset=['draw_id'], keep='first')
    
    return df

def save_data(df):
    """Lưu dữ liệu xuống file CSV"""
    df.to_csv(DATA_FILE, index=False)

def delete_last_row():
    """Xóa kỳ quay gần nhất"""
    df = load_data()
    if not df.empty:
        df = df.iloc[1:]
        save_data(df)
        return True
    return False

def delete_all_data():
    """Xóa sạch toàn bộ dữ liệu lịch sử"""
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
        return True
    return False

# ==============================================================================
# 3. THUẬT TOÁN TÁCH DỮ LIỆU THÔNG MINH (SỬA LỖI CHỈ NHẬN 1 KỲ)
# ==============================================================================
def parse_multi_draws(text, selected_date):
    """
    Hàm này sẽ quét toàn bộ đoạn văn bản bạn dán vào.
    Nó tìm kiếm mọi chuỗi có dạng Mã Kỳ (9 số) và các bộ 20 số đi kèm.
    """
    results = []
    
    # Bước 1: Tìm tất cả các Mã Kỳ Quay (thường là dãy 9 chữ số như 114072268)
    # Chúng ta tìm mọi dãy số có độ dài từ 8 đến 10 chữ số
    draw_matches = list(re.finditer(r'\b\d{8,10}\b', text))
    
    for i in range(len(draw_matches)):
        try:
            draw_id = draw_matches[i].group()
            
            # Xác định vùng văn bản chứa các con số kết quả (nằm giữa mã kỳ này và mã kỳ tiếp theo)
            start_pos = draw_matches[i].end()
            if i + 1 < len(draw_matches):
                end_pos = draw_matches[i+1].start()
                segment = text[start_pos:end_pos]
            else:
                segment = text[start_pos:]
            
            # Trích xuất tất cả các số từ 01 đến 80 trong phân đoạn này
            # Xử lý cả trường hợp số dính liền bằng cách tìm mọi cặp 2 chữ số
            numbers_in_segment = re.findall(r'\d{1,2}', segment)
            valid_numbers = []
            for n in numbers_in_segment:
                val = int(n)
                if 1 <= val <= 80:
                    valid_numbers.append(val)
            
            # Loại bỏ trùng lặp trong cùng 1 kỳ và lấy đúng 20 số đầu tiên tìm thấy
            unique_nums = []
            for n in valid_numbers:
                if n not in unique_nums:
                    unique_nums.append(n)
                if len(unique_nums) == 20:
                    break
            
            # Nếu tìm thấy đủ (hoặc gần đủ) 20 số thì mới ghi nhận là 1 kỳ hợp lệ
            if len(unique_nums) >= 15:
                results.append({
                    'draw_id': draw_id,
                    'time': datetime.combine(selected_date, datetime.now().time()),
                    'nums': sorted(unique_nums),
                    'super_num': unique_nums[-1]
                })
        except Exception:
            continue
            
    return results

# ==============================================================================
# 4. THUẬT TOÁN DỰ ĐOÁN AI 2.0
# ==============================================================================
def run_prediction(df):
    if df.empty:
        return [], "Không có dữ liệu"
    
    # Phân tích dựa trên 20 kỳ gần nhất
    recent_df = df.head(20)
    all_numbers = []
    for i in range(1, 21):
        all_numbers.extend(recent_df[f'num_{i}'].tolist())
    
    # Tính tần suất
    freq = pd.Series(all_numbers).value_counts()
    
    # Chấm điểm 80 con số
    scores = {}
    last_draw = [df.iloc[0][f'num_{i}'] for i in range(1, 21)]
    
    for n in range(1, 81):
        score = freq.get(n, 0) * 1.5 # Tần suất
        if n in last_draw: score += 3.0 # Cầu bệt
        if (n-1) in last_draw or (n+1) in last_draw: score += 1.0 # Cầu hàng xóm
        score += random.uniform(0, 1.0) # Ngẫu nhiên hóa
        scores[n] = score
        
    # Lấy top các số điểm cao nhất
    sorted_nums = sorted(scores, key=scores.get, reverse=True)
    return sorted_nums[:25], "Phân tích đa luồng"

# ==============================================================================
# 5. GIAO DIỆN NGƯỜI DÙNG (FULL UI)
# ==============================================================================

st.title("🚀 BINGO VIP - HỆ THỐNG TỰ ĐỘNG")

# Khởi tạo trạng thái bộ nhớ tạm
if 'predict_data' not in st.session_state:
    st.session_state['predict_data'] = None
if 'input_key' not in st.session_state:
    st.session_state['input_key'] = 0

# Tải dữ liệu
df_history = load_data()

# --- KHU VỰC NHẬP LIỆU ---
with st.expander("📥 NHẬP DỮ LIỆU (DÁN CẢ BẢNG TẠI ĐÂY)", expanded=True):
    col_a, col_b = st.columns([2, 1])
    with col_a:
        input_date = st.date_input("Chọn ngày quay:", datetime.now())
    with col_b:
        if st.button("🗑 Xóa ô nhập"):
            st.session_state['input_key'] += 1
            st.rerun()
            
    raw_text = st.text_area(
        "Dán nội dung copy từ trang kết quả:", 
        height=200, 
        placeholder="Mã kỳ quay: 114072... Kết quả: 01 05 10...",
        key=f"text_input_{st.session_state['input_key']}"
    )
    
    if st.button("🔥 XỬ LÝ & LƯU LỊCH SỬ", type="primary", use_container_width=True):
        if raw_text:
            extracted_draws = parse_multi_draws(raw_text, input_date)
            
            if extracted_draws:
                new_count = 0
                for item in extracted_draws:
                    # Kiểm tra trùng mã kỳ
                    if not df_history.empty and str(item['draw_id']) in df_history['draw_id'].astype(str).values:
                        continue
                    
                    # Tạo dòng mới
                    new_data = {'draw_id': item['draw_id'], 'time': item['time']}
                    for i, val in enumerate(item['nums']):
                        new_data[f'num_{i+1}'] = val
                    new_data['super_num'] = item['super_num']
                    
                    df_history = pd.concat([pd.DataFrame([new_data]), df_history], ignore_index=True)
                    new_count += 1
                
                if new_count > 0:
                    save_data(df_history)
                    st.success(f"✅ Đã thêm mới {new_count} kỳ quay vào lịch sử!")
                else:
                    st.warning("⚠️ Dữ liệu này đã tồn tại trong lịch sử.")
                
                # Cập nhật dự đoán ngay lập tức
                top_nums, _ = run_prediction(df_history)
                st.session_state['predict_data'] = top_nums
                st.rerun()
            else:
                st.error("❌ Không tìm thấy dữ liệu hợp lệ. Vui lòng kiểm tra lại nội dung dán.")

# --- KHU VỰC DỰ ĐOÁN & CÁCH CHƠI ---
if st.session_state['predict_data']:
    st.markdown("---")
    st.header("🎯 KẾT QUẢ DỰ ĐOÁN AI")
    
    # Định nghĩa đầy đủ các kiểu chơi (Giải quyết vấn đề 2)
    modes = {
        "10 Tinh (10 số)": 10,
        "9 Tinh (9 số)": 9,
        "8 Tinh (8 số)": 8,
        "7 Tinh (7 số)": 7,
        "6 Tinh (6 số)": 6,
        "5 Tinh (5 số)": 5,
        "4 Tinh (4 số)": 4,
        "3 Tinh (3 số)": 3,
        "2 Tinh (2 số)": 2,
        "1 Tinh (1 số)": 1,
        "Dàn 20 số": 20
    }
    
    selected_mode = st.selectbox("Chọn kiểu chơi (7, 8, 9 Tinh ở đây):", list(modes.keys()), index=4)
    num_to_pick = modes[selected_mode]
    
    # Lấy các số dự đoán
    final_numbers = sorted(st.session_state['predict_data'][:num_to_pick])
    
    # Hiển thị số đẹp
    cols = st.columns(5)
    for i, n in enumerate(final_numbers):
        with cols[i % 5]:
            bg_color = "#E74C3C" if n > 40 else "#3498DB"
            st.markdown(
                f"<div style='background-color:{bg_color}; color:white; padding:15px; border-radius:10px; text-align:center; font-size:24px; font-weight:bold; margin-bottom:10px;'>{n:02d}</div>", 
                unsafe_allow_html=True
            )

# --- KHU VỰC LỊCH SỬ CHI TIẾT ---
st.markdown("---")
with st.expander("📋 XEM LỊCH SỬ CHI TIẾT", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("↩️ Xóa kỳ vừa nhập"):
            if delete_last_row(): st.rerun()
    with col2:
        if st.button("🧨 XÓA TẤT CẢ DỮ LIỆU"):
            if delete_all_data(): st.rerun()
            
    if not df_history.empty:
        # Cấu hình hiển thị bảng đầy đủ các cột số
        st.dataframe(
            df_history.head(30), 
            use_container_width=True, 
            hide_index=True
        )
    else:
        st.info("Chưa có dữ liệu lịch sử.")
