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
    page_title="Bingo Taiwan VIP Final", 
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
    QUAN TRỌNG: Sắp xếp theo Mã Kỳ (draw_id) từ LớN đến NHỎ.
    """
    # Tạo danh sách tên cột 20 số
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
    
    # Chuyển đổi draw_id sang số nguyên để sắp xếp cho chuẩn xác
    if 'draw_id' in df.columns:
        df['draw_id'] = pd.to_numeric(df['draw_id'], errors='coerce')
    
    # Chuyển đổi cột thời gian
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
    
    # Sắp xếp: Mã kỳ LỚN NHẤT (Mới nhất) lên đầu
    df = df.dropna(subset=['draw_id'])
    df = df.sort_values(by='draw_id', ascending=False)
    
    # Xóa trùng lặp mã kỳ (Giữ lại dòng mới nhất nếu trùng)
    df = df.drop_duplicates(subset=['draw_id'], keep='first')
    
    return df

def save_data(df):
    """Lưu dữ liệu xuống file"""
    # Trước khi lưu, đảm bảo sắp xếp lại lần nữa cho chắc chắn
    df = df.sort_values(by='draw_id', ascending=False)
    df.to_csv(DATA_FILE, index=False)

def delete_last_row():
    """Xóa kỳ quay mới nhất (dòng đầu tiên)"""
    df = load_data()
    if not df.empty:
        # Xóa dòng đầu tiên (index 0)
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
    draw_pattern = r'\b114\d{6}\b'
    draw_matches = list(re.finditer(draw_pattern, text))
    
    for i in range(len(draw_matches)):
        try:
            draw_id = int(draw_matches[i].group()) # Chuyển thành số nguyên ngay
            
            # Xác định vùng chứa số của kỳ này
            start_pos = draw_matches[i].end()
            if i + 1 < len(draw_matches):
                end_pos = draw_matches[i+1].start()
                segment = text[start_pos:end_pos]
            else:
                segment = text[start_pos:]
            
            # 2. Lọc lấy các con số trong vùng này
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
            
            if len(unique_nums) >= 15:
                # Lấy số siêu cấp
                super_n = unique_nums[-1]
                
                # Sắp xếp lại dãy số kết quả cho đẹp
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
# 4. THUẬT TOÁN AI (CORE LOGIC - DÙNG TOÀN BỘ DỮ LIỆU)
# ==============================================================================
def run_prediction(df):
    """
    Thuật toán phân tích số dựa trên TOÀN BỘ lịch sử.
    """
    if df.empty:
        return []
    
    # 1. Tần suất tổng thể (Dựa trên TẤT CẢ các kỳ)
    # Lấy toàn bộ dữ liệu số ra một danh sách phẳng
    all_numbers_history = []
    for i in range(1, 21):
        all_numbers_history.extend(df[f'num_{i}'].tolist())
    
    # Tính tần suất xuất hiện của từng số trong toàn bộ lịch sử
    freq = pd.Series(all_numbers_history).value_counts()
    
    # 2. Lấy kỳ vừa quay xong (dòng đầu tiên) để bắt cầu bệt
    last_draw = [df.iloc[0][f'num_{i}'] for i in range(1, 21)]
    
    scores = {}
    for n in range(1, 81):
        # Điểm cơ bản = Tần suất xuất hiện trong toàn bộ lịch sử
        # Chia cho tổng số kỳ để chuẩn hóa điểm số
        base_score = freq.get(n, 0)
        
        # Hệ số điều chỉnh:
        score = base_score * 1.0
        
        # Điểm cộng cầu bệt (số vừa ra kỳ trước) - Quan trọng
        if n in last_draw: 
            score += (len(df) * 0.05) # Cộng điểm tương ứng 5% trọng số lịch sử
        
        # Điểm cộng cầu hàng xóm (n-1 và n+1)
        if (n-1) in last_draw or (n+1) in last_draw: 
            score += (len(df) * 0.02)
        
        # Yếu tố ngẫu nhiên nhẹ (để tránh trả về kết quả giống hệt nhau mãi)
        score += random.uniform(0, 1.0)
        
        scores[n] = score
        
    # Sắp xếp từ điểm cao xuống thấp
    ranked_numbers = sorted(scores, key=scores.get, reverse=True)
    return ranked_numbers

# ==============================================================================
# 5. GIAO DIỆN NGƯỜI DÙNG (UI)
# ==============================================================================

st.title("🎲 BINGO TAIWAN - MASTER AI")

# Khởi tạo Session State
if 'predict_data' not in st.session_state:
    st.session_state['predict_data'] = None
if 'input_key' not in st.session_state:
    st.session_state['input_key'] = 0

# Tải dữ liệu (Đã được sắp xếp Lớn -> Nhỏ trong hàm load_data)
df_history = load_data()

# --- KHUNG NHẬP LIỆU ---
with st.container(border=True):
    st.subheader("1. DỮ LIỆU ĐẦU VÀO")
    
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
        placeholder="Copy bảng kết quả từ web dán vào đây...",
        key=f"text_input_{st.session_state['input_key']}"
    )

    st.write("") # Khoảng cách
    
    # --- HAI NÚT BẤM (ĐỎ & XANH) ---
    col_btn_1, col_btn_2 = st.columns(2)
    
    # Nút 1: LƯU DỮ LIỆU (Màu Đỏ)
    with col_btn_1:
        if st.button("💾 LƯU DỮ LIỆU MỚI", type="primary", use_container_width=True):
            if raw_text.strip():
                extracted = parse_multi_draws(raw_text, input_date)
                if extracted:
                    added = 0
                    for item in extracted:
                        # Kiểm tra trùng (dùng draw_id dạng số)
                        if not df_history.empty and item['draw_id'] in df_history['draw_id'].values:
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
                        # Reload lại để bảng cập nhật thứ tự
                        st.rerun()
                    else:
                        st.warning("Dữ liệu này đã có trong máy rồi!")
                else:
                    st.error("Lỗi: Không đọc được số nào. Hãy kiểm tra lại.")
            else:
                st.warning("Bạn chưa dán nội dung nào cả!")

    # Nút 2: PHÂN TÍCH (Màu Xám/Trắng - Nổi bật chức năng riêng)
    with col_btn_2:
        if st.button("🚀 CHẠY PHÂN TÍCH (TẤT CẢ KỲ)", use_container_width=True):
            if not df_history.empty:
                # Chạy phân tích trên toàn bộ dữ liệu df_history
                st.session_state['predict_data'] = run_prediction(df_history)
                st.toast(f"Đã phân tích dựa trên {len(df_history)} kỳ quay!", icon="✅")
            else:
                st.error("Chưa có lịch sử để phân tích.")

# --- HIỂN THỊ KẾT QUẢ ---
if st.session_state['predict_data']:
    st.markdown("---")
    st.header("🎯 KẾT QUẢ DỰ ĐOÁN")
    
    # Menu chọn cách chơi
    modes = {
        "10 Tinh (10 số)": 10, "9 Tinh (9 số)": 9, "8 Tinh (8 số)": 8,
        "7 Tinh (7 số)": 7, "6 Tinh (6 số)": 6, "5 Tinh (5 số)": 5,
        "4 Tinh (4 số)": 4, "3 Tinh (3 số)": 3, "2 Tinh (2 số)": 2,
        "1 Tinh (1 số)": 1, "Dàn 20 số": 20
    }
    
    mode_name = st.selectbox("Chọn cách đánh:", list(modes.keys()), index=4)
    pick_count = modes[mode_name]
    
    # Lấy kết quả
    final_result = sorted(st.session_state['predict_data'][:pick_count])
    
    # Hiển thị
    cols = st.columns(5)
    for idx, num in enumerate(final_result):
        with cols[idx % 5]:
            color = "#E74C3C" if num > 40 else "#3498DB"
            st.markdown(
                f"<div style='background-color:{color}; color:white; padding:15px; border-radius:10px; text-align:center; font-weight:bold; font-size:20px; margin-bottom:10px;'>{num:02d}</div>",
                unsafe_allow_html=True
            )

# --- QUẢN LÝ LỊCH SỬ (ĐÃ SẮP XẾP LỚN -> NHỎ) ---
st.markdown("---")
with st.expander("📋 LỊCH SỬ KỲ QUAY (MỚI NHẤT TRÊN CÙNG)", expanded=True):
    col_del_1, col_del_2 = st.columns(2)
    with col_del_1:
        if st.button("↩️ Xóa kỳ mới nhất"):
            if delete_last_row(): st.rerun()
    with col_del_2:
        if st.button("🧨 Xóa tất cả"):
            if delete_all_data(): st.rerun()
            
    if not df_history.empty:
        # Cấu hình hiển thị cột draw_id là chuỗi số để không bị format có dấu phẩy
        st.dataframe(
            df_history, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "draw_id": st.column_config.NumberColumn(
                    "Mã Kỳ",
                    format="%d" # Hiển thị số nguyên không có dấu phẩy
                )
            }
        )
    else:
        st.info("Lịch sử trống.")
