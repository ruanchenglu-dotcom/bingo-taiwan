import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import re
from datetime import datetime

# ==============================================================================
# 1. CẤU HÌNH TRANG WEB
# ==============================================================================
st.set_page_config(
    page_title="Bingo Mobile VIP Fixed", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# Tên file lưu trữ lịch sử
DATA_FILE = 'bingo_history.csv'

# ==============================================================================
# 2. CÁC HÀM QUẢN LÝ DỮ LIỆU
# ==============================================================================
def load_data():
    """
    Hàm đọc dữ liệu từ file CSV lên.
    Tạo đủ cột cho draw_id, time, super_num và 20 con số (num_1 -> num_20).
    """
    columns = ['draw_id', 'time'] + [f'num_{i}' for i in range(1, 21)] + ['super_num']
    df = pd.DataFrame(columns=columns)
    
    if os.path.exists(DATA_FILE):
        try:
            loaded_df = pd.read_csv(DATA_FILE)
            if not loaded_df.empty: 
                df = loaded_df
        except Exception: 
            pass
    
    # Chuyển đổi cột thời gian
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
    
    # Sắp xếp: Mới nhất lên đầu
    df = df.dropna(subset=['time'])
    df = df.sort_values(by='time', ascending=False)
    # Xóa trùng lặp
    df = df.drop_duplicates(subset=['draw_id'], keep='first')
    
    return df

def save_data(df):
    """Lưu dữ liệu xuống CSV"""
    df.to_csv(DATA_FILE, index=False)

def delete_last_row():
    """Xóa kỳ mới nhất"""
    df = load_data()
    if not df.empty:
        deleted_id = df.iloc[0]['draw_id']
        df = df.iloc[1:] 
        save_data(df)
        return True, deleted_id
    return False, None

def delete_all_data():
    """Xóa sạch dữ liệu"""
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
        return True
    return False

# ==============================================================================
# 3. HÀM XỬ LÝ VĂN BẢN (ĐÃ SỬA LỖI ĐỌC NHIỀU DÒNG)
# ==============================================================================
def parse_bulk_text(text, selected_date):
    """
    Hàm quét từng dòng văn bản để tách nhiều kỳ.
    Sử dụng splitlines() để đảm bảo tách dòng chính xác trên mọi thiết bị.
    """
    found_draws = []
    
    # Tách dòng an toàn
    lines = text.strip().splitlines()
    
    for line in lines:
        try:
            # Bỏ qua dòng trống
            if not line.strip(): 
                continue

            # Tách toàn bộ số trong dòng ra
            # Sử dụng regex tìm tất cả các chuỗi số liên tiếp
            numbers_str = re.findall(r'\d+', line)
            numbers = [int(n) for n in numbers_str]
            
            # Nếu dòng quá ngắn (ít hơn 15 số) thì bỏ qua
            if len(numbers) < 15:
                continue
            
            draw_id = None
            balls = []
            super_n = 0
            
            # 1. Tìm Mã Kỳ (> 100.000.000)
            potential_ids = [n for n in numbers if n > 100000000]
            if potential_ids:
                draw_id = str(max(potential_ids)) # Lấy số lớn nhất làm ID
            else:
                continue # Không có mã kỳ thì bỏ qua dòng này
            
            # 2. Tìm 20 Số Kết Quả (1 <= n <= 80)
            potential_balls = [n for n in numbers if 1 <= n <= 80]
            
            # Lọc trùng trong 1 dòng (giữ thứ tự)
            seen = set()
            unique_balls = []
            for x in potential_balls:
                if x not in seen:
                    unique_balls.append(x)
                    seen.add(x)
                    if len(unique_balls) == 20: 
                        break
            
            balls = sorted(unique_balls)
            
            # 3. Lưu kết quả nếu đủ số
            if len(balls) >= 15:
                # Lấy số siêu cấp (thường là số cuối cùng hoặc số thứ 20)
                super_n = balls[-1] if balls else 0
                
                # Thời gian giả lập
                final_time = datetime.combine(selected_date, datetime.now().time())
                
                found_draws.append({
                    'draw_id': draw_id,
                    'time': final_time,
                    'nums': balls,
                    'super_num': super_n
                })
        except Exception:
            continue
            
    return found_draws

# ==============================================================================
# 4. THUẬT TOÁN AI (PHÂN TÍCH ĐA CHIỀU)
# ==============================================================================
def advanced_prediction_v2(df):
    if df.empty: return [], "Chưa có dữ liệu"
    
    # Lấy 15 kỳ gần nhất
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
    
    # Lọc cân bằng
    candidates = ranked_nums[:25]
    final_picks = []
    odd_count = 0
    even_count = 0
    
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
        
    return final_picks, "AI 2.0 Multi-Factor"

# ==============================================================================
# 5. GIAO DIỆN CHÍNH
# ==============================================================================

st.title("📥 BINGO NHẬP LIỆU HÀNG LOẠT (BẢN FIX)")

if 'analysis_result' not in st.session_state: st.session_state['analysis_result'] = None
if 'text_input_key' not in st.session_state: st.session_state['text_input_key'] = 0

df = load_data()

# --- KHUNG NHẬP LIỆU ---
with st.container(border=True):
    col_date, col_clear = st.columns([2, 1])
    with col_date:
        input_date = st.date_input("Ngày:", datetime.now(), label_visibility="collapsed")
    with col_clear:
        if st.button("🗑 Xóa ô nhập", use_container_width=True):
            st.session_state['text_input_key'] += 1
            st.rerun()

    text_paste = st.text_area(
        "", 
        height=150, 
        placeholder="Dán toàn bộ bảng kết quả (10-20 dòng) vào đây...",
        key=f"input_{st.session_state['text_input_key']}"
    )

    if st.button("🔥 LƯU TẤT CẢ & PHÂN TÍCH", type="primary", use_container_width=True):
        if text_paste.strip():
            # Xử lý đa dòng
            draws_list = parse_bulk_text(text_paste, input_date)
            
            if len(draws_list) > 0:
                count_new = 0
                latest_draw_id = None
                
                # Sắp xếp các kỳ tìm được theo ID tăng dần
                draws_list_sorted = sorted(draws_list, key=lambda x: int(x['draw_id']))
                
                for draw in draws_list_sorted:
                    # Kiểm tra trùng
                    if not df.empty and str(draw['draw_id']) in df['draw_id'].astype(str).values:
                        continue 
                    
                    # Tạo dòng mới
                    new_row = {'draw_id': draw['draw_id'], 'time': draw['time']}
                    for i, n in enumerate(draw['nums']): 
                        new_row[f'num_{i+1}'] = n
                    new_row['super_num'] = draw['super_num']
                    
                    df = pd.concat([pd.DataFrame([new_row]), df], ignore_index=True)
                    count_new += 1
                    
                    # Tìm ID lớn nhất để phân tích
                    if latest_draw_id is None or int(draw['draw_id']) > int(latest_draw_id):
                        latest_draw_id = draw['draw_id']
                
                # Nếu không có kỳ mới (do trùng hết), lấy kỳ lớn nhất trong đám vừa paste
                if latest_draw_id is None:
                     latest_draw_id = max([d['draw_id'] for d in draws_list], key=lambda x: int(x))

                # Lưu và thông báo
                if count_new > 0:
                    df = df.sort_values(by='time', ascending=False)
                    save_data(df)
                    st.success(f"✅ Đã thêm {count_new} kỳ mới! Tổng cộng đã tìm thấy {len(draws_list)} dòng.")
                else:
                    st.warning("⚠️ Dữ liệu đã có sẵn. Đang phân tích kỳ mới nhất...")

                # Phân tích
                p_nums, method = advanced_prediction_v2(df)
                st.session_state['analysis_result'] = {'nums': p_nums, 'ref_id': latest_draw_id}
                st.session_state['text_input_key'] += 1
                st.rerun()
            else:
                st.error("❌ Không tìm thấy kỳ nào hợp lệ. Hãy kiểm tra lại định dạng copy.")
        else:
            st.warning("Bạn chưa dán dữ liệu!")

# --- KHUNG KẾT QUẢ ---
if st.session_state['analysis_result']:
    res = st.session_state['analysis_result']
    st.markdown("---")
    st.header(f"🎯 DỰ ĐOÁN (Sau kỳ {res['ref_id']})")
    
    # --- DANH SÁCH GAME MODES ĐẦY ĐỦ (1-10 SAO) ---
    # Đã sửa lại theo yêu cầu: Đủ 7, 8, 9 Tinh và sắp xếp từ 10 xuống 1
    game_modes = {
        "10 Tinh (10 Số)": 10, 
        "9 Tinh (9 Số)": 9, 
        "8 Tinh (8 Số)": 8,
        "7 Tinh (7 Số)": 7, 
        "6 Tinh (6 Số)": 6, 
        "5 Tinh (5 Số)": 5, 
        "4 Tinh (4 Số)": 4, 
        "3 Tinh (3 Số)": 3, 
        "2 Tinh (2 Số)": 2, 
        "1 Tinh (1 Số)": 1, 
        "Full 20 Số": 20
    }
    
    st.write("🎯 **Chọn dàn đánh:**")
    mode = st.selectbox("", list(game_modes.keys()), index=4, label_visibility="collapsed")
    pick_n = game_modes[mode]
    
    best_picks = res['nums'][:pick_n]
    final_display = sorted(best_picks)
    
    st.info(f"⚡ Dàn **{pick_n} số** xác suất cao nhất:")
    
    cols = st.columns(4)
    for idx, n in enumerate(final_display):
        color = "#d63031" if n > 40 else "#0984e3"
        with cols[idx % 4]:
             st.markdown(
                 f"<div style='text-align: center; font-size: 20px; font-weight: bold; color: white; background-color: {color}; border-radius: 10px; padding: 10px; margin-bottom: 8px; box-shadow: 2px 2px 5px rgba(0,0,0,0.2);'>"
                 f"{n:02d}"
                 f"</div>", 
                 unsafe_allow_html=True
             )
    
    if pick_n >= 5:
        big = len([n for n in final_display if n > 40])
        st.caption(f"📊 Thống kê: {big} Tài - {pick_n-big} Xỉu")

# --- KHUNG LỊCH SỬ CHI TIẾT ---
st.markdown("---")
with st.expander("🛠 Lịch sử & Dữ liệu", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        if st.button("↩️ Xóa kỳ mới nhất"):
            delete_last_row()
            st.rerun()
    with c2:
        if st.button("🗑 Xóa TẤT CẢ"):
            delete_all_data()
            st.rerun()
            
    if not df.empty:
        st.write("📋 **Chi tiết các kỳ đã nhập:**")
        
        # Chọn các cột hiển thị: ID, 20 số, Super Num
        # Tạo danh sách tên cột rõ ràng
        display_cols = ['draw_id'] + [f'num_{i}' for i in range(1, 21)] + ['super_num']
        
        # Hiển thị bảng
        st.dataframe(
            df[display_cols].head(50), # Hiện 50 dòng
            use_container_width=True, 
            hide_index=True,
            column_config={
                "draw_id": "Mã Kỳ",
                "super_num": "Siêu Cấp"
            }
        )
    else:
        st.caption("Chưa có dữ liệu.")
