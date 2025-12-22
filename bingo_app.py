import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import re
from datetime import datetime

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Bingo AI Bulk Import", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

DATA_FILE = 'bingo_history.csv'

# --- QUẢN LÝ DỮ LIỆU ---
def load_data():
    columns = ['draw_id', 'time'] + [f'num_{i}' for i in range(1, 21)] + ['super_num']
    df = pd.DataFrame(columns=columns)
    if os.path.exists(DATA_FILE):
        try:
            loaded_df = pd.read_csv(DATA_FILE)
            if not loaded_df.empty: df = loaded_df
        except: pass
    
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
    
    df = df.dropna(subset=['time'])
    df = df.sort_values(by='time', ascending=False)
    df = df.drop_duplicates(subset=['draw_id'], keep='first')
    return df

def save_data(df):
    df.to_csv(DATA_FILE, index=False)

def delete_last_row():
    df = load_data()
    if not df.empty:
        deleted_id = df.iloc[0]['draw_id']
        df = df.iloc[1:]
        save_data(df)
        return True, deleted_id
    return False, None

def delete_all_data():
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
        return True
    return False

# ========================================================
# 🚀 CẢI TIẾN: HÀM ĐỌC ĐA LUỒNG (ĐỌC CẢ BẢNG)
# ========================================================
def parse_bulk_text(text, selected_date):
    """
    Hàm này sẽ quét từng dòng một để tìm tất cả các kỳ có trong văn bản copy
    """
    found_draws = []
    
    # 1. Tách văn bản thành từng dòng (dựa vào dấu xuống dòng)
    lines = text.strip().split('\n')
    
    for line in lines:
        try:
            # Làm sạch dòng
            clean_line = re.sub(r'\D', ' ', line)
            numbers = [int(n) for n in clean_line.split() if n.strip()]
            
            # Bỏ qua dòng quá ngắn (không đủ số liệu)
            if len(numbers) < 15: continue
            
            draw_id = None
            balls = []
            super_n = 0
            
            # Tìm Mã kỳ (> 100 triệu)
            potential_ids = [n for n in numbers if n > 100000000]
            if potential_ids: 
                draw_id = str(potential_ids[0])
            else:
                # Nếu dòng có nhiều số nhưng không có ID, bỏ qua để tránh rác
                continue
            
            # Tìm 20 số (1-80)
            potential_balls = [n for n in numbers if 1 <= n <= 80]
            
            # Lọc trùng giữ thứ tự
            seen = set()
            unique_balls = []
            for x in potential_balls:
                if x not in seen:
                    unique_balls.append(x)
                    seen.add(x)
                    if len(unique_balls) == 20: break
            
            balls = sorted(unique_balls)
            
            # Nếu đủ 20 số -> Ghi nhận là 1 kỳ hợp lệ
            if len(balls) >= 15:
                super_n = balls[-1] if balls else 0
                # Giả lập thời gian (vì copy bảng không có giờ cụ thể, ta lấy giờ hiện tại)
                final_time = datetime.combine(selected_date, datetime.now().time())
                
                found_draws.append({
                    'draw_id': draw_id, 
                    'time': final_time, 
                    'nums': balls, 
                    'super_num': super_n
                })
        except:
            continue
            
    # Trả về danh sách các kỳ tìm được (Sắp xếp từ cũ đến mới để lưu cho đúng)
    # Nhưng khi hiển thị ta cần kỳ mới nhất để phân tích
    return found_draws

# --- THUẬT TOÁN AI 2.0 (GIỮ NGUYÊN) ---
def advanced_prediction_v2(df):
    if df.empty: return [], "Chưa có dữ liệu"
    
    # Lấy dữ liệu để phân tích (Ưu tiên các kỳ mới nhất vừa nhập)
    short_term_df = df.head(15) 
    last_draw = [df.iloc[0][f'num_{i}'] for i in range(1, 21)]
    
    all_short_nums = [n for i in range(1, 21) for n in short_term_df[f'num_{i}']]
    freq_short = pd.Series(all_short_nums).value_counts()
    
    scores = {}
    for n in range(1, 81):
        score = 0
        count = freq_short.get(n, 0)
        score += count * 2.0 
        if n in last_draw: score += 4.0 # Bệt
        if (n-1) in last_draw or (n+1) in last_draw: score += 1.5 # Hàng xóm
        score += random.uniform(0, 1.0)
        scores[n] = score

    ranked_nums = sorted(scores, key=scores.get, reverse=True)
    
    # Bộ lọc cân bằng
    candidates = ranked_nums[:25]
    final_picks = []
    odd_count, even_count = 0, 0
    
    for num in candidates:
        if len(final_picks) == 20: break
        is_odd = (num % 2 != 0)
        if is_odd and odd_count < 12:
            final_picks.append(num)
            odd_count += 1
        elif not is_odd and even_count < 12:
            final_picks.append(num)
            even_count += 1
            
    if len(final_picks) < 20:
        remain = [x for x in candidates if x not in final_picks]
        final_picks.extend(remain[:20-len(final_picks)])
        
    return final_picks, "AI 2.0 Bulk"

# =================================================
# GIAO DIỆN CHÍNH
# =================================================

st.title("📥 BINGO NHẬP LIỆU HÀNG LOẠT")

if 'analysis_result' not in st.session_state: st.session_state['analysis_result'] = None
if 'text_input_key' not in st.session_state: st.session_state['text_input_key'] = 0

df = load_data()

# --- INPUT ---
with st.container(border=True):
    col_date, col_clear = st.columns([2, 1])
    with col_date:
        input_date = st.date_input("Ngày:", datetime.now(), label_visibility="collapsed")
    with col_clear:
        if st.button("🗑 Xóa ô", use_container_width=True):
            st.session_state['text_input_key'] += 1
            st.rerun()

    st.caption("💡 Mẹo: Bạn có thể copy CẢ BẢNG (nhiều dòng) dán vào đây, máy sẽ tự tách.")
    text_paste = st.text_area(
        "", 
        height=150, 
        placeholder="Dán cả bảng kết quả vào đây...",
        key=f"input_{st.session_state['text_input_key']}"
    )

    if st.button("🔥 LƯU & PHÂN TÍCH TẤT CẢ", type="primary", use_container_width=True):
        if text_paste.strip():
            # Dùng hàm xử lý đa luồng mới
            draws_list = parse_bulk_text(text_paste, input_date)
            
            if len(draws_list) > 0:
                count_new = 0
                latest_draw_id = None
                
                # Duyệt qua từng kỳ tìm được và lưu
                for draw in draws_list:
                    # Kiểm tra trùng
                    if not df.empty and str(draw['draw_id']) in df['draw_id'].astype(str).values:
                        continue # Đã có thì bỏ qua
                    
                    # Lưu kỳ mới
                    new_row = {'draw_id': draw['draw_id'], 'time': draw['time']}
                    for i, n in enumerate(draw['nums']): new_row[f'num_{i+1}'] = n
                    new_row['super_num'] = draw['super_num']
                    
                    # Thêm vào dataframe tạm
                    df = pd.concat([pd.DataFrame([new_row]), df], ignore_index=True)
                    count_new += 1
                    
                    # Cập nhật ID mới nhất để hiển thị phân tích
                    if latest_draw_id is None or int(draw['draw_id']) > int(latest_draw_id):
                        latest_draw_id = draw['draw_id']
                
                # Lưu file
                if count_new > 0:
                    save_data(df)
                    st.success(f"✅ Đã thêm thành công {count_new} kỳ mới vào dữ liệu!")
                else:
                    st.warning("⚠️ Các kỳ này đã có trong máy rồi, không cần lưu lại.")
                    # Vẫn lấy ID mới nhất trong đám vừa paste để phân tích
                    latest_draw_id = draws_list[0]['draw_id']

                # CHẠY PHÂN TÍCH (Dựa trên dữ liệu vừa cập nhật)
                p_nums, method = advanced_prediction_v2(df)
                st.session_state['analysis_result'] = {'nums': p_nums, 'ref_id': latest_draw_id}
                
                # Xóa ô nhập liệu
                st.session_state['text_input_key'] += 1
                st.rerun()
            else:
                st.error("❌ Không đọc được dữ liệu nào. Hãy chắc chắn bạn copy đúng bảng số.")
        else:
            st.warning("Hãy dán dữ liệu vào trước!")

# --- OUTPUT ---
if st.session_state['analysis_result']:
    res = st.session_state['analysis_result']
    st.markdown("---")
    st.header(f"🎯 DỰ ĐOÁN (Sau kỳ {res['ref_id']})")
    
    # MENU CHỌN CÁCH CHƠI
    game_modes = {
        "10 Tinh (10 Số)": 10, "9 Tinh (9 Số)": 9, "8 Tinh (8 Số)": 8,
        "7 Tinh (7 Số)": 7, "6 Tinh (6 Số)": 6, "5 Tinh (5 Số)": 5, 
        "4 Tinh (4 Số)": 4, "3 Tinh (3 Số)": 3, "2 Tinh (2 Số)": 2, 
        "1 Tinh (1 Số)": 1, "Full 20 Số": 20
    }
    
    st.write("Chọn dàn đánh:")
    mode = st.selectbox("", list(game_modes.keys()), index=4, label_visibility="collapsed")
    pick_n = game_modes[mode]
    
    best_picks = res['nums'][:pick_n]
    final_display = sorted(best_picks)
    
    st.info(f"⚡ Dàn **{pick_n} số** (AI 2.0):")
    
    cols = st.columns(4)
    for idx, n in enumerate(final_display):
        color = "#d63031" if n > 40 else "#0984e3"
        with cols[idx % 4]:
             st.markdown(f"<div style='text-align: center; font-size: 20px; font-weight: bold; color: white; background-color: {color}; border-radius: 10px; padding: 10px; margin-bottom: 8px;'>{n:02d}</div>", unsafe_allow_html=True)
    
    if pick_n >= 5:
        big = len([n for n in final_display if n > 40])
        st.caption(f"Tài: {big} | Xỉu: {pick_n-big}")

# --- TOOLS ---
st.markdown("---")
with st.expander("Lịch sử & Cài đặt"):
    c1, c2 = st.columns(2)
    with c1:
        if st.button("↩️ Xóa kỳ sai"):
            delete_last_row(); st.rerun()
    with c2:
        if st.button("🗑 Xóa HẾT"):
            delete_all_data(); st.rerun()
            
    if not df.empty:
        st.write("10 Kỳ gần nhất trong máy:")
        st.dataframe(df.head(10)[['draw_id', 'super_num']], use_container_width=True, hide_index=True)
