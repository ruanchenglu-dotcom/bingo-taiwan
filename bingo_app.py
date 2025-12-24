import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import re
from collections import Counter
from datetime import datetime

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG & CSS GIAO DIỆN
# ==============================================================================
st.set_page_config(
    page_title="Bingo Taiwan Pro Max", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# CSS Tùy chỉnh: Nút bấm cao (Piano keys) và Giao diện Tab
st.markdown("""
<style>
    /* Style cho nút bấm số trong lưới: Cao và Hẹp */
    div.stButton > button:first-child {
        min-height: 75px;       
        width: 100%;            
        margin: 0px 2px;        
        font-weight: bold;
        border-radius: 8px;     
        font-size: 20px;        
        padding: 0px;           
        line-height: 75px;      
    }
    
    /* Màu đỏ nổi bật cho Tab đang chọn */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
        font-weight: bold;
    }
    
    /* Thu hẹp khoảng cách giữa các cột */
    [data-testid="column"] {
        padding: 0px 2px;
    }
    
    /* Style cho bảng thống kê hàng xóm */
    .neighbor-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 5px;
        border-left: 5px solid #ff4b4b;
        font-weight: bold;
        color: #31333F;
    }
</style>
""", unsafe_allow_html=True)

# Tên file cơ sở dữ liệu
DATA_FILE = 'bingo_history.csv'

# ==============================================================================
# 2. QUẢN LÝ TRẠNG THÁI (SESSION STATE)
# ==============================================================================
# Khởi tạo các biến nhớ
if 'selected_nums' not in st.session_state:
    st.session_state['selected_nums'] = [] 
if 'predict_data' not in st.session_state:
    st.session_state['predict_data'] = None 
if 'selected_algo' not in st.session_state:
    st.session_state['selected_algo'] = "🔮 AI Master (Tổng Hợp)"
if 'neighbor_stats' not in st.session_state:
    st.session_state['neighbor_stats'] = []

# Key động cho ô nhập liệu (Để fix lỗi xóa không được)
if 'paste_key_id' not in st.session_state:
    st.session_state['paste_key_id'] = 0

# ==============================================================================
# 3. CÁC HÀM XỬ LÝ DỮ LIỆU (DATABASE)
# ==============================================================================
def load_data():
    """Tải dữ liệu và đảm bảo định dạng đúng."""
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
    
    if 'draw_id' in df.columns:
        df['draw_id'] = pd.to_numeric(df['draw_id'], errors='coerce').fillna(0).astype(int)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
    
    df = df[df['draw_id'] > 0]
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
# 4. HÀM HỖ TRỢ GIAO DIỆN & LOGIC
# ==============================================================================
def toggle_number(num):
    """Bật/tắt chọn số trên bàn phím."""
    if num in st.session_state['selected_nums']:
        st.session_state['selected_nums'].remove(num)
    else:
        if len(st.session_state['selected_nums']) < 20:
            st.session_state['selected_nums'].append(num)
        else:
            st.toast("⚠️ Chỉ được chọn tối đa 20 số!", icon="🚫")

def clear_selection():
    """Xóa các số đang chọn trên bàn phím."""
    st.session_state['selected_nums'] = []

def clear_paste_box():
    """Hàm callback để xóa ô dán (Fix lỗi không xóa được)."""
    st.session_state['paste_key_id'] += 1

def parse_multi_draws(text, selected_date):
    """Tách số thông minh từ văn bản copy."""
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
# 5. CÁC MODULE PHÂN TÍCH & THỐNG KÊ (NÂNG CẤP MỚI)
# ==============================================================================

def analyze_neighbors(df):
    """
    Thống kê các cặp số hàng xóm (Liên tiếp) hay đi cùng nhau.
    Ví dụ: 09-10, 45-46.
    Dựa trên 50 kỳ gần nhất.
    """
    if df.empty: return []
    
    # Lấy 50 kỳ gần nhất để thống kê cho nóng
    recent_df = df.head(50)
    consecutive_pairs = []
    
    for idx, row in recent_df.iterrows():
        # Lấy 20 số của kỳ đó
        nums = sorted([row[f'num_{i}'] for i in range(1, 21)])
        
        # Tìm các cặp liên tiếp (n, n+1)
        for i in range(len(nums) - 1):
            if nums[i+1] == nums[i] + 1:
                consecutive_pairs.append(f"{nums[i]:02d}-{nums[i+1]:02d}")
                
    # Đếm tần suất
    counts = Counter(consecutive_pairs)
    # Lấy top 10 cặp hay ra nhất
    return counts.most_common(10)

def run_prediction(df, strategy):
    """Hệ thống dự đoán đa chiến thuật (Dựa trên 10 kỳ gần nhất)."""
    if df.empty: return []
    
    # Lấy 10 kỳ gần nhất
    recent_df = df.head(10)
    
    all_numbers_history = []
    for i in range(1, 21):
        all_numbers_history.extend(recent_df[f'num_{i}'].tolist())
    
    freq = pd.Series(all_numbers_history).value_counts()
    last_draw = [df.iloc[0][f'num_{i}'] for i in range(1, 21)]
    
    scores = {}
    
    # 1. AI MASTER
    if strategy == "🔮 AI Master (Tổng Hợp)":
        for n in range(1, 81):
            score = freq.get(n, 0) * 1.5
            if n in last_draw: score += 3.0 
            if (n-1) in last_draw or (n+1) in last_draw: score += 1.0 
            score += random.uniform(0, 1.0)
            scores[n] = score

    # 2. SOI CẦU NÓNG
    elif strategy == "🔥 Soi Cầu Nóng (Hot)":
        for n in range(1, 81):
            scores[n] = freq.get(n, 0) + (random.random() * 0.1)

    # 3. SOI CẦU LẠNH
    elif strategy == "❄️ Soi Cầu Lạnh (Nuôi)":
        max_f = freq.max() if not freq.empty else 0
        for n in range(1, 81):
            f = freq.get(n, 0)
            scores[n] = (max_f - f) + random.uniform(0, 1.5)

    # 4. SOI CẦU BỆT
    elif strategy == "♻️ Soi Cầu Bệt (Lại)":
        for n in range(1, 81):
            score = freq.get(n, 0) * 0.1
            if n in last_draw: score += 1000
            scores[n] = score

    # 5. THẦN SỐ HỌC
    elif strategy == "✨ Thần Số Học":
        now = datetime.now()
        seed_val = sum(int(d) for d in str(now.day)+str(now.month)) + now.hour
        random.seed(seed_val)
        for n in range(1, 81):
            scores[n] = random.randint(1, 100) + (freq.get(n, 0) * 1.0)
        random.seed(None)

    return sorted(scores, key=scores.get, reverse=True)

# ==============================================================================
# 6. GIAO DIỆN NGƯỜI DÙNG (UI CHÍNH)
# ==============================================================================

st.title("🎲 BINGO TAIWAN PRO MAX")

df_history = load_data()

# ==============================================================================
# KHU VỰC NHẬP LIỆU (TABS)
# ==============================================================================
with st.container(border=True):
    tab_manual, tab_paste = st.tabs(["🖱️ BÀN PHÍM SỐ", "📋 DÁN (COPY)"])

    # --- TAB 1: BÀN PHÍM SỐ ---
    with tab_manual:
        st.caption("Chế độ nhập tay.")
        
        tm1, tm2, tm3 = st.columns([2, 2, 1])
        with tm1:
            next_id = ""
            if not df_history.empty:
                try:
                    max_id = df_history['draw_id'].max()
                    next_id = str(int(max_id) + 1)
                except: pass
            manual_draw_id = st.text_input("Mã Kỳ Mới:", value=next_id, key="manual_id_input")
        with tm2:
            manual_date = st.date_input("Ngày:", datetime.now(), key="manual_date_input")
        with tm3:
            st.write("")
            st.write("")
            if st.button("Xóa chọn", key="btn_clr_manual", use_container_width=True):
                clear_selection()
                st.rerun()

        st.markdown("---")
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
                    b_type = "primary" if is_sel else "secondary"
                    if st.button(f"{num:02d}", key=f"btn_grid_{num}", type=b_type, use_container_width=True):
                        toggle_number(num)
                        st.rerun()
        
        st.markdown("---")
        valid_supers = sorted(st.session_state['selected_nums']) if st.session_state['selected_nums'] else range(1, 81)
        manual_super = st.selectbox("🔥 Số Siêu Cấp:", valid_supers, key="super_sel_manual")
        
        if st.button("💾 LƯU KỲ THỦ CÔNG", type="primary", use_container_width=True, key="btn_save_manual"):
            if not manual_draw_id:
                st.error("Chưa nhập mã kỳ!")
            elif len(st.session_state['selected_nums']) != 20:
                st.error(f"Mới chọn {len(st.session_state['selected_nums'])} số. Cần đủ 20 số!")
            else:
                try:
                    check_id = int(manual_draw_id)
                    if not df_history.empty and check_id in df_history['draw_id'].values:
                        st.warning("Mã kỳ này đã tồn tại!")
                    else:
                        new_row = {
                            'draw_id': check_id,
                            'time': datetime.combine(manual_date, datetime.now().time()),
                            'super_num': manual_super
                        }
                        sorted_final = sorted(st.session_state['selected_nums'])
                        for i, val in enumerate(sorted_final):
                            new_row[f'num_{i+1}'] = val
                        
                        df_history = pd.concat([pd.DataFrame([new_row]), df_history], ignore_index=True)
                        save_data(df_history)
                        st.success(f"Đã lưu kỳ {check_id}!")
                        clear_selection()
                        st.rerun()
                except ValueError:
                    st.error("Mã kỳ phải là số!")

    # --- TAB 2: DÁN COPY (ĐÃ FIX LỖI XÓA) ---
    with tab_paste:
        st.caption("Dán bảng kết quả từ web.")
        
        tp1, tp2 = st.columns([3, 1])
        with tp1:
            paste_date = st.date_input("Ngày:", datetime.now(), key="paste_date_input")
        with tp2:
            # Sử dụng callback để xóa chắc chắn 100%
            st.button("🗑 Xóa ô dán", key="btn_clr_paste", on_click=clear_paste_box, use_container_width=True)
                
        # Key động để reset widget
        dynamic_key = f"paste_area_{st.session_state['paste_key_id']}"
        paste_text = st.text_area("Dán dữ liệu:", height=200, key=dynamic_key)
        
        if st.button("💾 XỬ LÝ & LƯU HÀNG LOẠT", type="primary", use_container_width=True, key="btn_save_paste"):
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
                    st.error("Không tìm thấy dữ liệu hợp lệ.")
            else:
                st.warning("Ô nhập trống!")

# ==============================================================================
# KHU VỰC PHÂN TÍCH & THỐNG KÊ
# ==============================================================================
st.write("")
st.markdown("### 📊 TRUNG TÂM PHÂN TÍCH")

if st.button("🚀 CHẠY PHÂN TÍCH NGAY", type="primary", use_container_width=True, key="btn_run_analyze"):
    if not df_history.empty:
        # Chạy dự đoán
        st.session_state['predict_data'] = run_prediction(df_history, st.session_state['selected_algo'])
        # Chạy thống kê hàng xóm
        st.session_state['neighbor_stats'] = analyze_neighbors(df_history)
        st.toast(f"Đã phân tích xong!", icon="✅")
    else:
        st.error("Chưa có lịch sử để phân tích.")

# ==============================================================================
# HIỂN THỊ KẾT QUẢ
# ==============================================================================
if st.session_state['predict_data'] or not df_history.empty:
    st.markdown("---")
    
    # Chia cột: Bên trái là Kết quả Dự đoán, Bên phải là Thống kê Hàng xóm
    res_col1, res_col2 = st.columns([2, 1])
    
    # --- CỘT 1: DỰ ĐOÁN SỐ ---
    with res_col1:
        st.subheader("🎯 KẾT QUẢ DỰ ĐOÁN")
        
        c_algo, c_mode = st.columns(2)
        with c_algo:
            algos = [
                "🔮 AI Master (Tổng Hợp)", "🔥 Soi Cầu Nóng (Hot)", 
                "❄️ Soi Cầu Lạnh (Nuôi)", "♻️ Soi Cầu Bệt (Lại)", "✨ Thần Số Học"
            ]
            curr_algo = st.session_state['selected_algo']
            idx_algo = algos.index(curr_algo) if curr_algo in algos else 0
            new_algo = st.selectbox("Thuật Toán:", algos, index=idx_algo, key="sel_algo")
            
            if new_algo != st.session_state['selected_algo']:
                st.session_state['selected_algo'] = new_algo
                if not df_history.empty:
                    st.session_state['predict_data'] = run_prediction(df_history, new_algo)
                    st.rerun()

        with c_mode:
            modes = {
                "10 Tinh": 10, "9 Tinh": 9, "8 Tinh": 8, "7 Tinh": 7,
                "6 Tinh": 6, "5 Tinh": 5, "4 Tinh": 4, "3 Tinh": 3,
                "2 Tinh": 2, "1 Tinh": 1, "Dàn 20 số": 20
            }
            mode_key = st.selectbox("Dàn Đánh:", list(modes.keys()), index=4, key="sel_mode")
            pick_n = modes[mode_key]

        # Hiển thị số
        if st.session_state['predict_data']:
            st.markdown(f"**Gợi ý từ: {st.session_state['selected_algo']}**")
            final_nums = sorted(st.session_state['predict_data'][:pick_n])
            
            cols = st.columns(5)
            for idx, n in enumerate(final_nums):
                with cols[idx % 5]:
                    color = "#E74C3C" if n > 40 else "#3498DB"
                    st.markdown(
                        f"<div style='background-color:{color}; color:white; padding:12px; border-radius:8px; text-align:center; font-weight:bold; font-size:20px; margin-bottom:8px;'>{n:02d}</div>",
                        unsafe_allow_html=True
                    )
            
            # Thống kê nhanh
            t = len([x for x in final_nums if x > 40])
            x = len([x for x in final_nums if x <= 40])
            l = len([x for x in final_nums if x % 2 != 0])
            c = len([x for x in final_nums if x % 2 == 0])
            
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("🔴 Tài", t)
            sc2.metric("🔵 Xỉu", x)
            sc3.metric("⚡ Lẻ", l)
            sc4.metric("📦 Chẵn", c)

    # --- CỘT 2: THỐNG KÊ HÀNG XÓM (TÍNH NĂNG MỚI) ---
    with res_col2:
        st.subheader("🔗 CẶP HÀNG XÓM HAY VỀ")
        st.caption("Top cặp số liền kề (50 kỳ gần nhất):")
        
        if st.session_state['neighbor_stats']:
            for pair, count in st.session_state['neighbor_stats']:
                st.markdown(
                    f"<div class='neighbor-box'>⚡ Cặp <b>{pair}</b> - Về {count} lần</div>", 
                    unsafe_allow_html=True
                )
        else:
            st.info("Chưa có dữ liệu thống kê. Hãy bấm 'Chạy Phân Tích'.")

# ==============================================================================
# QUẢN LÝ LỊCH SỬ
# ==============================================================================
st.markdown("---")
with st.expander("📋 LỊCH SỬ KỲ QUAY", expanded=True):
    cd1, cd2 = st.columns(2)
    with cd1:
        if st.button("↩️ Xóa kỳ mới nhất", key="btn_del_last"):
            if delete_last_row(): st.rerun()
    with cd2:
        if st.button("🧨 Xóa tất cả", key="btn_del_all"):
            if delete_all_data(): st.rerun()
            
    if not df_history.empty:
        st.dataframe(
            df_history, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "draw_id": st.column_config.NumberColumn("Mã Kỳ", format="%d"),
                "super_num": st.column_config.NumberColumn("Siêu Cấp", format="%d")
            }
        )
    else:
        st.info("Chưa có dữ liệu.")
