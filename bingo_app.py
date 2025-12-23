import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import re
from datetime import datetime

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG & GIAO DIỆN
# ==============================================================================
st.set_page_config(
    page_title="Bingo Taiwan Hybrid - 10 Draws", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# CSS Tùy chỉnh: Làm đẹp nút bấm bàn phím số
st.markdown("""
<style>
    /* Style cho nút bấm số trong lưới */
    div.stButton > button:first-child {
        min-height: 45px;
        font-weight: bold;
        border-radius: 8px;
    }
    /* Màu đỏ cho tab đang chọn */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Tên file cơ sở dữ liệu
DATA_FILE = 'bingo_history.csv'

# ==============================================================================
# 2. CÁC HÀM XỬ LÝ DỮ LIỆU (DATABASE)
# ==============================================================================
def load_data():
    """Tải dữ liệu và sắp xếp theo Mã Kỳ giảm dần (Mới nhất lên đầu)."""
    # Tạo danh sách cột
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
    
    # Ép kiểu dữ liệu để tránh lỗi
    if 'draw_id' in df.columns:
        df['draw_id'] = pd.to_numeric(df['draw_id'], errors='coerce')
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
    
    # Sắp xếp và lọc
    df = df.dropna(subset=['draw_id'])
    df = df.sort_values(by='draw_id', ascending=False)
    df = df.drop_duplicates(subset=['draw_id'], keep='first')
    
    return df

def save_data(df):
    """Lưu dữ liệu xuống file CSV."""
    df = df.sort_values(by='draw_id', ascending=False)
    df.to_csv(DATA_FILE, index=False)

def delete_last_row():
    """Xóa kỳ mới nhất."""
    df = load_data()
    if not df.empty:
        df = df.iloc[1:]
        save_data(df)
        return True
    return False

def delete_all_data():
    """Xóa tất cả dữ liệu."""
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
        return True
    return False

# ==============================================================================
# 3. QUẢN LÝ TRẠNG THÁI & HÀM HỖ TRỢ NHẬP LIỆU
# ==============================================================================
if 'selected_nums' not in st.session_state:
    st.session_state['selected_nums'] = [] # Danh sách số đang chọn thủ công
if 'predict_data' not in st.session_state:
    st.session_state['predict_data'] = None
if 'selected_algo' not in st.session_state:
    st.session_state['selected_algo'] = "🔮 AI Master (Tổng Hợp)"

def toggle_number(num):
    """Hàm bật/tắt chọn số trên bàn phím."""
    if num in st.session_state['selected_nums']:
        st.session_state['selected_nums'].remove(num)
    else:
        if len(st.session_state['selected_nums']) < 20:
            st.session_state['selected_nums'].append(num)
        else:
            st.toast("⚠️ Chỉ được chọn tối đa 20 số!", icon="🚫")

def clear_selection():
    """Xóa các số đang chọn."""
    st.session_state['selected_nums'] = []

def parse_multi_draws(text, selected_date):
    """Hàm tách số thông minh từ văn bản copy."""
    results = []
    draw_pattern = r'\b114\d{6}\b'
    draw_matches = list(re.finditer(draw_pattern, text))
    
    for i in range(len(draw_matches)):
        try:
            draw_id = int(draw_matches[i].group())
            
            start_pos = draw_matches[i].end()
            if i + 1 < len(draw_matches):
                end_pos = draw_matches[i+1].start()
                segment = text[start_pos:end_pos]
            else:
                segment = text[start_pos:]
            
            all_digits = re.findall(r'\d{2}', segment)
            valid_numbers = []
            for n in all_digits:
                val = int(n)
                if 1 <= val <= 80:
                    valid_numbers.append(val)
            
            unique_nums = []
            for n in valid_numbers:
                if n not in unique_nums:
                    unique_nums.append(n)
                if len(unique_nums) == 20:
                    break
            
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
# 4. HỆ THỐNG THUẬT TOÁN (CHỈ PHÂN TÍCH 10 KỲ GẦN NHẤT)
# ==============================================================================
def run_prediction(df, strategy):
    if df.empty: return []
    
    # --- QUAN TRỌNG: CẮT LẤY 10 KỲ GẦN NHẤT ---
    # df đã được sắp xếp giảm dần (mới nhất ở trên), nên head(10) là 10 kỳ mới nhất
    recent_df = df.head(10)
    
    # Tính toán tần suất CHỈ TRONG 10 kỳ này
    all_numbers_history = []
    for i in range(1, 21):
        all_numbers_history.extend(recent_df[f'num_{i}'].tolist())
    
    # Tần suất (Số lần xuất hiện trong 10 kỳ)
    freq = pd.Series(all_numbers_history).value_counts()
    
    # Lấy kỳ vừa quay xong (dòng đầu tiên) để bắt cầu bệt
    last_draw = [df.iloc[0][f'num_{i}'] for i in range(1, 21)]
    
    scores = {}
    
    # Tổng số kỳ đang xét (Tối đa là 10)
    window_size = len(recent_df) 
    
    # 1. AI MASTER
    if strategy == "🔮 AI Master (Tổng Hợp)":
        for n in range(1, 81):
            # Điểm cơ bản = Số lần xuất hiện trong 10 kỳ
            score = freq.get(n, 0) * 1.5
            
            # Nếu vừa ra ở kỳ trước -> Cộng điểm lớn (Xu hướng bệt ngắn hạn)
            if n in last_draw: score += 3.0
            
            # Cầu hàng xóm
            if (n-1) in last_draw or (n+1) in last_draw: score += 1.0
            
            # Ngẫu nhiên nhẹ
            score += random.uniform(0, 1.0)
            scores[n] = score

    # 2. SOI CẦU NÓNG (HOT TREND 10 KỲ)
    elif strategy == "🔥 Soi Cầu Nóng (Hot)":
        for n in range(1, 81):
            # Chỉ quan tâm con nào ra nhiều nhất trong 10 kỳ qua
            scores[n] = freq.get(n, 0) + (random.random() * 0.1)

    # 3. SOI CẦU LẠNH (NUÔI SỐ TRONG 10 KỲ)
    elif strategy == "❄️ Soi Cầu Lạnh (Nuôi)":
        # Tìm con nào xuất hiện ít hoặc chưa xuất hiện trong 10 kỳ này
        max_f = freq.max() if not freq.empty else 0
        for n in range(1, 81):
            f = freq.get(n, 0)
            # Tần suất càng thấp điểm càng cao
            scores[n] = (max_f - f) + random.uniform(0, 1.5)

    # 4. SOI CẦU BỆT (LẠI)
    elif strategy == "♻️ Soi Cầu Bệt (Lại)":
        for n in range(1, 81):
            # Điểm chủ yếu dựa vào việc nó có nằm trong kỳ trước hay không
            score = freq.get(n, 0) * 0.1
            if n in last_draw: score += 1000
            scores[n] = score

    # 5. THẦN SỐ HỌC
    elif strategy == "✨ Thần Số Học":
        now = datetime.now()
        seed_val = sum(int(d) for d in str(now.day)+str(now.month)) + now.hour
        random.seed(seed_val)
        for n in range(1, 81):
            # Kết hợp tâm linh + tần suất ngắn hạn (10 kỳ)
            scores[n] = random.randint(1, 100) + (freq.get(n, 0) * 1.0)
        random.seed(None)

    return sorted(scores, key=scores.get, reverse=True)

# ==============================================================================
# 5. GIAO DIỆN NGƯỜI DÙNG (UI CHÍNH)
# ==============================================================================

st.title("🎲 BINGO TAIWAN - 10 DRAWS SYSTEM")

# Tải dữ liệu lịch sử
df_history = load_data()

# ==============================================================================
# KHU VỰC NHẬP LIỆU (SỬ DỤNG TABS)
# ==============================================================================
with st.container(border=True):
    # Tạo 2 Tab riêng biệt
    tab_manual, tab_paste = st.tabs(["🖱️ BÀN PHÍM SỐ (THỦ CÔNG)", "📋 DÁN TỪ WEB (COPY)"])

    # --------------------------------------------------------------------------
    # TAB 1: NHẬP THỦ CÔNG (GRID)
    # --------------------------------------------------------------------------
    with tab_manual:
        st.caption("Chế độ nhập từng số trực tiếp.")
        
        # Hàng nhập thông tin kỳ quay
        tm1, tm2, tm3 = st.columns([2, 2, 1])
        with tm1:
            # Gợi ý mã kỳ tiếp theo
            next_id = ""
            if not df_history.empty:
                next_id = str(int(df_history.iloc[0]['draw_id']) + 1)
            manual_draw_id = st.text_input("Nhập Mã Kỳ Mới:", value=next_id, key="manual_id")
        with tm2:
            manual_date = st.date_input("Ngày quay:", datetime.now(), key="manual_date")
        with tm3:
            st.write("")
            st.write("")
            if st.button("Xóa chọn", key="btn_clear_manual", use_container_width=True):
                clear_selection()
                st.rerun()

        st.markdown("---")
        
        # Hiển thị số lượng đã chọn
        cnt = len(st.session_state['selected_nums'])
        st.markdown(f"**🔢 Đã chọn: <span style='color:red; font-size:1.2em'>{cnt}/20</span> số**", unsafe_allow_html=True)

        # LƯỚI 80 SỐ
        for row in range(8):
            cols = st.columns(10)
            for col in range(10):
                num = row * 10 + col + 1
                if num > 80: break
                
                with cols[col]:
                    is_sel = num in st.session_state['selected_nums']
                    # Nút đỏ nếu chọn, xám nếu không
                    b_type = "primary" if is_sel else "secondary"
                    if st.button(f"{num:02d}", key=f"grid_{num}", type=b_type, use_container_width=True):
                        toggle_number(num)
                        st.rerun()
        
        # Chọn số siêu cấp
        st.markdown("---")
        valid_supers = sorted(st.session_state['selected_nums']) if st.session_state['selected_nums'] else range(1, 81)
        manual_super = st.selectbox("🔥 Số Siêu Cấp:", valid_supers, index=len(valid_supers)-1 if valid_supers else 0)
        
        # Nút Lưu Thủ Công
        if st.button("💾 LƯU KỲ THỦ CÔNG", type="primary", use_container_width=True, key="save_manual"):
            if not manual_draw_id:
                st.error("Thiếu mã kỳ!")
            elif len(st.session_state['selected_nums']) != 20:
                st.error("Chưa chọn đủ 20 số!")
            else:
                if not df_history.empty and int(manual_draw_id) in df_history['draw_id'].values:
                    st.warning("Mã kỳ này đã tồn tại!")
                else:
                    new_row = {
                        'draw_id': int(manual_draw_id),
                        'time': datetime.combine(manual_date, datetime.now().time()),
                        'super_num': manual_super
                    }
                    sorted_final = sorted(st.session_state['selected_nums'])
                    for i, val in enumerate(sorted_final):
                        new_row[f'num_{i+1}'] = val
                    
                    df_history = pd.concat([pd.DataFrame([new_row]), df_history], ignore_index=True)
                    save_data(df_history)
                    st.success(f"Đã lưu kỳ {manual_draw_id}!")
                    clear_selection()
                    st.rerun()

    # --------------------------------------------------------------------------
    # TAB 2: DÁN TỪ WEB (COPY-PASTE)
    # --------------------------------------------------------------------------
    with tab_paste:
        st.caption("Chế độ dán hàng loạt bảng kết quả.")
        
        tp1, tp2 = st.columns([3, 1])
        with tp1:
            paste_date = st.date_input("Ngày dữ liệu:", datetime.now(), key="paste_date")
        with tp2:
            if st.button("🗑 Xóa ô dán", key="clear_paste", use_container_width=True):
                st.rerun()
                
        paste_text = st.text_area("Dán kết quả vào đây:", height=200, placeholder="114072xxx ...", key="paste_area")
        
        if st.button("💾 XỬ LÝ & LƯU HÀNG LOẠT", type="primary", use_container_width=True, key="save_paste"):
            if paste_text.strip():
                extracted = parse_multi_draws(paste_text, paste_date)
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
                        st.success(f"Đã thêm {added} kỳ mới!")
                        st.rerun()
                    else:
                        st.warning("Dữ liệu đã có sẵn!")
                else:
                    st.error("Không đọc được dữ liệu nào.")
            else:
                st.warning("Ô nhập liệu trống!")

# ==============================================================================
# KHU VỰC PHÂN TÍCH (CHUNG CHO CẢ 2 TAB)
# ==============================================================================
st.write("")
st.markdown("### 📊 PHÂN TÍCH (Dựa trên 10 kỳ gần nhất)")

if st.button("🚀 CHẠY PHÂN TÍCH NGAY", type="primary", use_container_width=True):
    if not df_history.empty:
        st.session_state['predict_data'] = run_prediction(df_history, st.session_state['selected_algo'])
        st.toast(f"Đã phân tích 10 kỳ gần nhất theo: {st.session_state['selected_algo']}", icon="✅")
    else:
        st.error("Chưa có lịch sử để phân tích.")

# ==============================================================================
# HIỂN THỊ KẾT QUẢ & CẤU HÌNH
# ==============================================================================
if st.session_state['predict_data'] or not df_history.empty:
    st.markdown("---")
    st.subheader("🎯 KẾT QUẢ DỰ ĐOÁN")
    
    col_conf1, col_conf2 = st.columns(2)
    
    # Cấu hình Thuật toán
    with col_conf1:
        algos = [
            "🔮 AI Master (Tổng Hợp)",
            "🔥 Soi Cầu Nóng (Hot)",
            "❄️ Soi Cầu Lạnh (Nuôi)",
            "♻️ Soi Cầu Bệt (Lại)",
            "✨ Thần Số Học"
        ]
        algo_idx = 0
        if st.session_state['selected_algo'] in algos:
            algo_idx = algos.index(st.session_state['selected_algo'])
            
        new_algo = st.selectbox("🧠 Chọn Thuật Toán:", algos, index=algo_idx)
        if new_algo != st.session_state['selected_algo']:
            st.session_state['selected_algo'] = new_algo
            if not df_history.empty:
                st.session_state['predict_data'] = run_prediction(df_history, new_algo)
                st.rerun()

    # Cấu hình Cách chơi
    with col_conf2:
        modes = {
            "10 Tinh": 10, "9 Tinh": 9, "8 Tinh": 8, "7 Tinh": 7,
            "6 Tinh": 6, "5 Tinh": 5, "4 Tinh": 4, "3 Tinh": 3,
            "2 Tinh": 2, "1 Tinh": 1, "Dàn 20 số": 20
        }
        mode_key = st.selectbox("🎯 Chọn Dàn Đánh:", list(modes.keys()), index=4)
        pick_n = modes[mode_key]

    # HIỂN THỊ DÀN SỐ
    if st.session_state['predict_data']:
        st.markdown(f"##### Kết quả từ: **{st.session_state['selected_algo']}**")
        final_nums = sorted(st.session_state['predict_data'][:pick_n])
        
        # Grid hiển thị số
        cols = st.columns(5)
        for idx, n in enumerate(final_nums):
            with cols[idx % 5]:
                color = "#E74C3C" if n > 40 else "#3498DB"
                st.markdown(
                    f"<div style='background-color:{color}; color:white; padding:12px; border-radius:8px; text-align:center; font-weight:bold; font-size:20px; margin-bottom:8px;'>{n:02d}</div>",
                    unsafe_allow_html=True
                )
        
        # Thống kê chi tiết
        st.markdown("###### 📊 Thống kê dàn số:")
        t = len([x for x in final_nums if x > 40])
        x = len([x for x in final_nums if x <= 40])
        l = len([x for x in final_nums if x % 2 != 0])
        c = len([x for x in final_nums if x % 2 == 0])
        
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("🔴 Tài (>40)", t)
        sc2.metric("🔵 Xỉu (<=40)", x)
        sc3.metric("⚡ Lẻ", l)
        sc4.metric("📦 Chẵn", c)

# ==============================================================================
# QUẢN LÝ LỊCH SỬ
# ==============================================================================
st.markdown("---")
with st.expander("📋 LỊCH SỬ KỲ QUAY (MỚI NHẤT TRÊN CÙNG)", expanded=True):
    cd1, cd2 = st.columns(2)
    with cd1:
        if st.button("↩️ Xóa kỳ mới nhất"):
            if delete_last_row(): st.rerun()
    with cd2:
        if st.button("🧨 Xóa tất cả"):
            if delete_all_data(): st.rerun()
            
    if not df_history.empty:
        st.dataframe(
            df_history, 
            use_container_width=True, 
            hide_index=True,
            column_config={"draw_id": st.column_config.NumberColumn("Mã Kỳ", format="%d")}
        )
    else:
        st.info("Chưa có dữ liệu.")
