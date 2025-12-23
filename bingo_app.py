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
    page_title="Bingo Master - Đa Chiến Thuật", 
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
    Sắp xếp theo Mã Kỳ (draw_id) từ LỚN đến NHỎ.
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
    
    # Xóa trùng lặp mã kỳ
    df = df.drop_duplicates(subset=['draw_id'], keep='first')
    
    return df

def save_data(df):
    """Lưu dữ liệu xuống file"""
    # Trước khi lưu, đảm bảo sắp xếp lại lần nữa
    df = df.sort_values(by='draw_id', ascending=False)
    df.to_csv(DATA_FILE, index=False)

def delete_last_row():
    """Xóa kỳ quay mới nhất (dòng đầu tiên)"""
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
    
    # Tìm tất cả mã kỳ (9 chữ số, bắt đầu bằng 114)
    draw_pattern = r'\b114\d{6}\b'
    draw_matches = list(re.finditer(draw_pattern, text))
    
    for i in range(len(draw_matches)):
        try:
            draw_id = int(draw_matches[i].group()) # Chuyển thành số nguyên
            
            # Xác định vùng chứa số của kỳ này
            start_pos = draw_matches[i].end()
            if i + 1 < len(draw_matches):
                end_pos = draw_matches[i+1].start()
                segment = text[start_pos:end_pos]
            else:
                segment = text[start_pos:]
            
            # Lọc lấy các con số trong vùng này
            all_digits = re.findall(r'\d{2}', segment)
            
            valid_numbers = []
            for n in all_digits:
                val = int(n)
                if 1 <= val <= 80:
                    valid_numbers.append(val)
            
            # Lấy 20 số duy nhất đầu tiên
            unique_nums = []
            for n in valid_numbers:
                if n not in unique_nums:
                    unique_nums.append(n)
                if len(unique_nums) == 20:
                    break
            
            if len(unique_nums) >= 15:
                super_n = unique_nums[-1]
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
# 4. HỆ THỐNG ĐA THUẬT TOÁN (STRATEGY ENGINE)
# ==============================================================================
def run_prediction(df, strategy="AI Master"):
    """
    Hàm phân tích số dựa trên chiến thuật được chọn.
    """
    if df.empty:
        return []
    
    # Lấy dữ liệu toàn bộ lịch sử để tính tần suất
    all_numbers_history = []
    for i in range(1, 21):
        all_numbers_history.extend(df[f'num_{i}'].tolist())
    freq = pd.Series(all_numbers_history).value_counts()
    
    # Lấy kỳ vừa quay xong (để bắt bệt)
    last_draw = [df.iloc[0][f'num_{i}'] for i in range(1, 21)]
    
    scores = {}
    
    # --- CHIẾN THUẬT 1: AI MASTER (TỔNG HỢP) ---
    if strategy == "🔮 AI Master (Tổng Hợp)":
        total_draws = len(df)
        for n in range(1, 81):
            base_score = freq.get(n, 0)
            score = base_score * 1.0
            if n in last_draw: score += (total_draws * 0.05) # Ưu tiên Bệt
            if (n-1) in last_draw or (n+1) in last_draw: score += (total_draws * 0.02) # Ưu tiên Hàng xóm
            score += random.uniform(0, 1.0)
            scores[n] = score

    # --- CHIẾN THUẬT 2: SOI CẦU NÓNG (HOT TREND) ---
    elif strategy == "🔥 Soi Cầu Nóng (Hot)":
        # Chỉ quan tâm đến những số ra nhiều nhất
        for n in range(1, 81):
            # Điểm = Tần suất xuất hiện (Không cộng điểm ngẫu nhiên để thuần Hot)
            scores[n] = freq.get(n, 0) + (random.random() * 0.1) # Random rất nhỏ để phá hòa

    # --- CHIẾN THUẬT 3: SOI CẦU LẠNH (NUÔI SỐ) ---
    elif strategy == "❄️ Soi Cầu Lạnh (Nuôi)":
        # Tìm những số ÍT ra nhất (Đảo ngược điểm số)
        max_freq = freq.max()
        for n in range(1, 81):
            f = freq.get(n, 0)
            # Tần suất càng thấp, điểm càng cao
            scores[n] = (max_freq - f) + random.uniform(0, 2.0)

    # --- CHIẾN THUẬT 4: SOI CẦU BỆT (REPEATER) ---
    elif strategy == "♻️ Soi Cầu Bệt (Lại)":
        # Cực kỳ ưu tiên các số vừa ra ở kỳ trước
        for n in range(1, 81):
            score = freq.get(n, 0) * 0.1 # Tần suất chỉ đóng vai trò phụ
            if n in last_draw:
                score += 1000 # Điểm siêu lớn để chắc chắn lọt Top
            else:
                score += random.uniform(0, 5.0)
            scores[n] = score

    # --- CHIẾN THUẬT 5: THẦN SỐ HỌC (PYTHAGORAS) ---
    elif strategy == "u2728 Thần Số Học (Pythagoras)":
        # Tính toán dựa trên Ngày/Tháng/Năm/Giờ hiện tại (Vũ trụ)
        now = datetime.now()
        # Số chủ đạo ngày hôm nay
        day_sum = sum(int(digit) for digit in str(now.day) + str(now.month) + str(now.year))
        hour_seed = now.hour + now.minute
        
        # Seed random bằng con số thời gian để tạo ra bộ số "Định mệnh" tại thời điểm bấm nút
        random.seed(day_sum + hour_seed)
        
        for n in range(1, 81):
            # Tạo ra bộ số ngẫu nhiên nhưng cố định theo thời gian (Pseudo-random)
            # Kết hợp nhẹ với tần suất để không quá ảo
            mystic_score = random.randint(1, 100) 
            real_score = freq.get(n, 0) * 0.5
            scores[n] = mystic_score + real_score
            
        # Reset seed về mặc định để không ảnh hưởng các hàm khác
        random.seed(None)

    # Sắp xếp từ điểm cao xuống thấp
    ranked_numbers = sorted(scores, key=scores.get, reverse=True)
    return ranked_numbers

# ==============================================================================
# 5. GIAO DIỆN NGƯỜI DÙNG (UI)
# ==============================================================================

st.title("🎲 BINGO TAIWAN - ĐA CHIẾN THUẬT")

# Khởi tạo Session State
if 'predict_data' not in st.session_state:
    st.session_state['predict_data'] = None
if 'input_key' not in st.session_state:
    st.session_state['input_key'] = 0
if 'selected_algo' not in st.session_state:
    st.session_state['selected_algo'] = "🔮 AI Master (Tổng Hợp)"

# Tải dữ liệu (Đã được sắp xếp Lớn -> Nhỏ)
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
        height=120,
        placeholder="Copy bảng kết quả từ web dán vào đây...",
        key=f"text_input_{st.session_state['input_key']}"
    )

    st.write("") 
    
    # --- HAI NÚT BẤM ---
    col_btn_1, col_btn_2 = st.columns(2)
    
    # Nút 1: LƯU DỮ LIỆU
    with col_btn_1:
        if st.button("💾 LƯU DỮ LIỆU MỚI", type="primary", use_container_width=True):
            if raw_text.strip():
                extracted = parse_multi_draws(raw_text, input_date)
                if extracted:
                    added = 0
                    for item in extracted:
                        if not df_history.empty and item['draw_id'] in df_history['draw_id'].values:
                            continue
                        
                        new_row = {'draw_id': item['draw_id'], 'time': item['time']}
                        for i, val in enumerate(item['nums']):
                            new_row[f'num_{i+1}'] = val
                        new_row['super_num'] = item['super_num']
                        
                        df_history = pd.concat([pd.DataFrame([new_row]), df_history], ignore_index=True)
                        added += 1
                    
                    if added > 0:
                        save_data(df_history)
                        st.success(f"Đã lưu thành công {added} kỳ mới!")
                        st.rerun()
                    else:
                        st.warning("Dữ liệu này đã có trong máy rồi!")
                else:
                    st.error("Lỗi: Không đọc được số nào.")
            else:
                st.warning("Bạn chưa dán nội dung nào cả!")

    # Nút 2: PHÂN TÍCH
    with col_btn_2:
        if st.button("🚀 CHẠY PHÂN TÍCH", use_container_width=True):
            if not df_history.empty:
                # Chạy phân tích dựa trên Thuật toán đang chọn trong Session State
                st.session_state['predict_data'] = run_prediction(df_history, st.session_state['selected_algo'])
                st.toast(f"Đã chạy thuật toán: {st.session_state['selected_algo']}", icon="✅")
            else:
                st.error("Chưa có lịch sử để phân tích.")

# --- KHUNG CẤU HÌNH CHIẾN THUẬT & KẾT QUẢ ---
if st.session_state['predict_data'] or not df_history.empty:
    st.markdown("---")
    st.header("🎯 CẤU HÌNH & KẾT QUẢ")
    
    # --- PHẦN CHỌN THUẬT TOÁN (MỚI) ---
    col_algo, col_mode = st.columns(2)
    
    with col_algo:
        # Danh sách thuật toán
        algo_options = [
            "🔮 AI Master (Tổng Hợp)",
            "🔥 Soi Cầu Nóng (Hot)",
            "❄️ Soi Cầu Lạnh (Nuôi)",
            "♻️ Soi Cầu Bệt (Lại)",
            "u2728 Thần Số Học (Pythagoras)"
        ]
        
        selected_algo = st.selectbox(
            "🧠 Chọn Thuật Toán Phân Tích:", 
            algo_options, 
            index=0
        )
        
        # Nếu người dùng đổi thuật toán, lưu vào session và chạy lại ngay nếu đã có data
        if selected_algo != st.session_state['selected_algo']:
            st.session_state['selected_algo'] = selected_algo
            if not df_history.empty:
                st.session_state['predict_data'] = run_prediction(df_history, selected_algo)
                st.rerun()

    with col_mode:
        # Menu chọn cách chơi
        modes = {
            "10 Tinh (10 số)": 10, "9 Tinh (9 số)": 9, "8 Tinh (8 số)": 8,
            "7 Tinh (7 số)": 7, "6 Tinh (6 số)": 6, "5 Tinh (5 số)": 5,
            "4 Tinh (4 số)": 4, "3 Tinh (3 số)": 3, "2 Tinh (2 số)": 2,
            "1 Tinh (1 số)": 1, "Dàn 20 số": 20
        }
        mode_name = st.selectbox("🎯 Chọn Cách Đánh:", list(modes.keys()), index=4)
        pick_count = modes[mode_name]

    # --- HIỂN THỊ KẾT QUẢ ---
    if st.session_state['predict_data']:
        st.markdown(f"### Kết quả từ: **{st.session_state['selected_algo']}**")
        
        final_result = sorted(st.session_state['predict_data'][:pick_count])
        
        # Hiển thị số
        cols = st.columns(5)
        for idx, num in enumerate(final_result):
            with cols[idx % 5]:
                color = "#E74C3C" if num > 40 else "#3498DB"
                st.markdown(
                    f"<div style='background-color:{color}; color:white; padding:15px; border-radius:10px; text-align:center; font-weight:bold; font-size:20px; margin-bottom:10px;'>{num:02d}</div>",
                    unsafe_allow_html=True
                )
                
        # --- THỐNG KÊ CHI TIẾT ---
        st.markdown("#### 📊 Thống kê dàn số:")
        tai = len([n for n in final_result if n > 40])
        xiu = len([n for n in final_result if n <= 40])
        le = len([n for n in final_result if n % 2 != 0])
        chan = len([n for n in final_result if n % 2 == 0])
        
        stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
        with stat_c1: st.metric("🔴 TÀI", f"{tai}")
        with stat_c2: st.metric("🔵 XỈU", f"{xiu}")
        with stat_c3: st.metric("⚡ LẺ", f"{le}")
        with stat_c4: st.metric("📦 CHẴN", f"{chan}")

# --- QUẢN LÝ LỊCH SỬ ---
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
        st.dataframe(
            df_history, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "draw_id": st.column_config.NumberColumn("Mã Kỳ", format="%d")
            }
        )
    else:
        st.info("Lịch sử trống.")
