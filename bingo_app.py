import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import re
from datetime import datetime

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG
# ==============================================================================
st.set_page_config(
    page_title="Bingo Taiwan VIP Pro", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# Tên file cơ sở dữ liệu
DATA_FILE = 'bingo_history.csv'

# ==============================================================================
# 2. CÁC HÀM XỬ LÝ DỮ LIỆU (DATABASE)
# ==============================================================================
def load_data():
    """
    Hàm tải dữ liệu từ file CSV.
    Tạo sẵn 20 cột (num_1 đến num_20) để tránh lỗi hiển thị.
    """
    # Tạo danh sách tên cột
    num_cols = [f'num_{i}' for i in range(1, 21)]
    columns = ['draw_id', 'time'] + num_cols + ['super_num']
    
    # Tạo bảng rỗng trước
    df = pd.DataFrame(columns=columns)
    
    # Nếu file đã tồn tại thì đọc nó
    if os.path.exists(DATA_FILE):
        try:
            loaded_df = pd.read_csv(DATA_FILE)
            if not loaded_df.empty: 
                df = loaded_df
        except Exception: 
            pass
    
    # Xử lý cột thời gian
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
    
    # Sắp xếp: Mới nhất lên đầu
    df = df.dropna(subset=['time'])
    df = df.sort_values(by='time', ascending=False)
    # Xóa trùng lặp mã kỳ
    df = df.drop_duplicates(subset=['draw_id'], keep='first')
    
    return df

def save_data(df):
    """Lưu dữ liệu xuống file"""
    df.to_csv(DATA_FILE, index=False)

def delete_last_row():
    """Xóa kỳ quay gần nhất (dòng đầu tiên)"""
    df = load_data()
    if not df.empty:
        df = df.iloc[1:]
        save_data(df)
        return True
    return False

def delete_all_data():
    """Xóa trắng toàn bộ dữ liệu"""
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
        return True
    return False

# ==============================================================================
# 3. THUẬT TOÁN ĐỌC DỮ LIỆU WEB (PARSER)
# ==============================================================================
def parse_multi_draws(text, selected_date):
    """
    Hàm đọc dữ liệu copy từ web.
    Nhận diện mã kỳ 114xxxxxx và lấy 20 số đi kèm.
    """
    results = []
    
    # 1. Tìm tất cả mã kỳ (9 chữ số, bắt đầu bằng 114)
    # Regex này bắt chính xác mã kỳ Bingo năm nay
    draw_pattern = r'\b114\d{6}\b'
    draw_matches = list(re.finditer(draw_pattern, text))
    
    for i in range(len(draw_matches)):
        try:
            draw_id = draw_matches[i].group()
            
            # Xác định vùng chứa số của kỳ này
            start_pos = draw_matches[i].end()
            if i + 1 < len(draw_matches):
                end_pos = draw_matches[i+1].start()
                segment = text[start_pos:end_pos]
            else:
                segment = text[start_pos:]
            
            # 2. Lọc lấy các con số trong vùng này
            # Tìm tất cả các chuỗi 2 chữ số (để xử lý trường hợp dính liền)
            # Ví dụ: "010203" sẽ được tách thành 01, 02, 03
            all_digits = re.findall(r'\d{2}', segment)
            
            valid_numbers = []
            for n in all_digits:
                val = int(n)
                # Chỉ lấy số từ 01 đến 80
                if 1 <= val <= 80:
                    valid_numbers.append(val)
            
            # 3. Lấy 20 số duy nhất đầu tiên
            unique_nums = []
            for n in valid_numbers:
                if n not in unique_nums:
                    unique_nums.append(n)
                if len(unique_nums) == 20:
                    break
            
            # Nếu đủ 20 số (hoặc ít nhất 15 số để trừ hao) thì lưu
            if len(unique_nums) >= 15:
                # Lấy số siêu cấp (thường là số cuối cùng)
                super_n = unique_nums[-1]
                
                # Sắp xếp lại dãy số cho đẹp
                sorted_nums = sorted(unique_nums)
                
                results.append({
                    'draw_id': draw_id,
                    'time': datetime.combine(selected_date, datetime.now().time()),
                    'nums': sorted_nums,
                    'super_num': super_n
                })
        except Exception:
            continue
            
    return results

# ==============================================================================
# 4. THUẬT TOÁN AI (CORE LOGIC)
# ==============================================================================
def run_prediction(df):
    """
    Thuật toán phân tích số.
    """
    if df.empty:
        return []
    
    # Lấy 30 kỳ gần nhất để phân tích xu hướng
    recent_df = df.head(30)
    all_numbers = []
    for i in range(1, 21):
        all_numbers.extend(recent_df[f'num_{i}'].tolist())
    
    # Tính tần suất
    freq = pd.Series(all_numbers).value_counts()
    
    # Lấy kỳ vừa quay xong để bắt cầu bệt
    last_draw = [df.iloc[0][f'num_{i}'] for i in range(1, 21)]
    
    scores = {}
    for n in range(1, 81):
        # Điểm cơ bản từ tần suất
        score = freq.get(n, 0) * 1.5 
        
        # Điểm cộng cầu bệt (số vừa ra) - Rất quan trọng trong Bingo
        if n in last_draw: score += 4.0 
        
        # Điểm cộng cầu hàng xóm (n-1 và n+1)
        if (n-1) in last_draw or (n+1) in last_draw: score += 1.2
        
        # Yếu tố ngẫu nhiên nhẹ để thay đổi bộ số
        score += random.uniform(0, 1.2)
        
        scores[n] = score
        
    # Sắp xếp từ điểm cao xuống thấp
    ranked_numbers = sorted(scores, key=scores.get, reverse=True)
    return ranked_numbers

# ==============================================================================
# 5. GIAO DIỆN NGƯỜI DÙNG (UI) - ĐÃ THÊM NÚT BẠN CẦN
# ==============================================================================

st.title("🎲 BINGO TAIWAN VIP PRO")

# Khởi tạo Session State (Bộ nhớ tạm)
if 'predict_data' not in st.session_state:
    st.session_state['predict_data'] = None
if 'input_key' not in st.session_state:
    st.session_state['input_key'] = 0

df_history = load_data()

# --- KHUNG NHẬP LIỆU & NÚT BẤM ---
with st.container(border=True):
    st.subheader("1. NHẬP DỮ LIỆU & PHÂN TÍCH")
    
    # Chọn ngày và nút Xóa ô nhập
    c1, c2 = st.columns([3, 1])
    with c1:
        input_date = st.date_input("Ngày quay:", datetime.now(), label_visibility="collapsed")
    with c2:
        if st.button("🗑 Xóa ô nhập", use_container_width=True):
            st.session_state['input_key'] += 1
            st.rerun()
            
    # Ô nhập liệu văn bản
    raw_text = st.text_area(
        "Dán kết quả vào đây:", 
        height=150,
        placeholder="Copy bảng kết quả từ web dán vào đây (Hỗ trợ dán nhiều kỳ cùng lúc)...",
        key=f"text_input_{st.session_state['input_key']}"
    )

    # --- ĐÂY LÀ PHẦN NÚT BẤM BẠN YÊU CẦU ---
    st.write("") # Tạo khoảng cách nhỏ
    col_btn_1, col_btn_2 = st.columns(2)
    
    # Nút 1: Lưu dữ liệu mới
    with col_btn_1:
        if st.button("💾 LƯU DỮ LIỆU MỚI", type="primary", use_container_width=True):
            if raw_text.strip():
                extracted = parse_multi_draws(raw_text, input_date)
                if extracted:
                    added = 0
                    for item in extracted:
                        # Kiểm tra xem mã kỳ này đã có chưa
                        if not df_history.empty and str(item['draw_id']) in df_history['draw_id'].astype(str).values:
                            continue
                        
                        # Thêm dòng mới
                        new_row = {'draw_id': item['draw_id'], 'time': item['time']}
                        for i, val in enumerate(item['nums']):
                            new_row[f'num_{i+1}'] = val
                        new_row['super_num'] = item['super_num']
                        
                        df_history = pd.concat([pd.DataFrame([new_row]), df_history], ignore_index=True)
                        added += 1
                    
                    if added > 0:
                        save_data(df_history)
                        st.success(f"Đã lưu thành công {added} kỳ mới!")
                        # Tự động chạy phân tích sau khi lưu
                        st.session_state['predict_data'] = run_prediction(df_history)
                        st.rerun()
                    else:
                        st.warning("Dữ liệu này đã có trong máy rồi!")
                else:
                    st.error("Lỗi: Không đọc được số nào. Hãy kiểm tra lại nội dung dán.")
            else:
                st.warning("Bạn chưa dán nội dung nào cả!")

    # Nút 2: NÚT PHÂN TÍCH (Vị trí bạn muốn bổ sung)
    with col_btn_2:
        # Nút này dùng để chạy lại AI trên dữ liệu cũ mà không cần paste
        if st.button("🚀 CHẠY PHÂN TÍCH (AI)", use_container_width=True):
            if not df_history.empty:
                st.session_state['predict_data'] = run_prediction(df_history)
                st.toast("Đã phân tích xong dữ liệu hiện có!", icon="✅")
            else:
                st.error("Chưa có lịch sử để phân tích. Hãy nạp dữ liệu trước.")

# --- HIỂN THỊ KẾT QUẢ ---
if st.session_state['predict_data']:
    st.markdown("---")
    st.header("🎯 KẾT QUẢ SOI CẦU")
    
    # Menu chọn cách chơi đầy đủ
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
    
    # Selectbox chọn cách chơi
    mode_name = st.selectbox("Chọn cách đánh:", list(modes.keys()), index=4)
    pick_count = modes[mode_name]
    
    # Lấy số từ kết quả dự đoán
    final_result = sorted(st.session_state['predict_data'][:pick_count])
    
    # Hiển thị đẹp mắt
    cols = st.columns(5)
    for idx, num in enumerate(final_result):
        with cols[idx % 5]:
            # Màu đỏ nếu > 40, Xanh nếu <= 40
            color = "#E74C3C" if num > 40 else "#3498DB"
            st.markdown(
                f"<div style='background-color:{color}; color:white; padding:15px; border-radius:10px; text-align:center; font-weight:bold; font-size:20px; margin-bottom:10px;'>{num:02d}</div>",
                unsafe_allow_html=True
            )

# --- QUẢN LÝ LỊCH SỬ ---
st.markdown("---")
with st.expander("📋 LỊCH SỬ KỲ QUAY", expanded=True):
    col_del_1, col_del_2 = st.columns(2)
    with col_del_1:
        if st.button("↩️ Xóa kỳ mới nhất"):
            if delete_last_row(): st.rerun()
    with col_del_2:
        if st.button("🧨 Xóa tất cả"):
            if delete_all_data(): st.rerun()
            
    if not df_history.empty:
        # Hiển thị bảng dữ liệu
        st.dataframe(df_history, use_container_width=True, hide_index=True)
    else:
        st.info("Lịch sử trống.")
