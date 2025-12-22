import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import re
from datetime import datetime

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Bingo Mobile AI 2.0", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

DATA_FILE = 'bingo_history.csv'

# --- KHỐI QUẢN LÝ DỮ LIỆU ---
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

# --- KHỐI TÁCH SỐ THÔNG MINH ---
def smart_parse_text(text, selected_date):
    try:
        clean_text = re.sub(r'\D', ' ', text)
        numbers = [int(n) for n in clean_text.split() if n.strip()]
        
        draw_id = None
        balls = []
        super_n = 0
        
        potential_ids = [n for n in numbers if n > 100000000]
        if potential_ids: draw_id = str(potential_ids[0])
        
        potential_balls = [n for n in numbers if 1 <= n <= 80]
        
        if not draw_id: draw_id = f"Manual-{int(datetime.now().timestamp())}"
        
        seen = set()
        unique_balls = []
        for x in potential_balls:
            if x not in seen:
                unique_balls.append(x)
                seen.add(x)
                if len(unique_balls) == 20: break
        
        balls = sorted(unique_balls)

        if len(balls) >= 15:
            super_n = balls[-1] if balls else 0
            final_time = datetime.combine(selected_date, datetime.now().time())
            return {'draw_id': draw_id, 'time': final_time, 'nums': balls, 'super_num': super_n}, "OK"
        else:
            return None, f"Lỗi: Tìm thấy {len(balls)} số. Hãy Copy lại."
            
    except Exception as e: return None, str(e)

# ========================================================
# 🚀 NÂNG CẤP: THUẬT TOÁN AI 2.0 (ĐA CHIỀU)
# ========================================================
def advanced_prediction_v2(df):
    if df.empty: return [], "Chưa có dữ liệu"
    
    # 1. Chuẩn bị dữ liệu
    # Lấy 10 kỳ gần nhất (Trend ngắn hạn quan trọng hơn dài hạn)
    short_term_df = df.head(10)
    # Lấy kỳ vừa quay xong
    last_draw = [df.iloc[0][f'num_{i}'] for i in range(1, 21)]
    
    # Tính tần suất trong 10 kỳ gần nhất
    all_short_nums = [n for i in range(1, 21) for n in short_term_df[f'num_{i}']]
    freq_short = pd.Series(all_short_nums).value_counts()
    
    scores = {}
    
    # 2. CHẤM ĐIỂM TỪNG SỐ (1-80)
    for n in range(1, 81):
        score = 0
        
        # --- TIÊU CHÍ 1: HOT TREND (Số đang vào cầu) ---
        # Nếu số này xuất hiện nhiều trong 10 kỳ gần đây -> Cộng điểm lớn
        count = freq_short.get(n, 0)
        score += count * 2.0 
        
        # --- TIÊU CHÍ 2: CẦU BỆT (Số rơi lại) ---
        # Nếu số này vừa ra ở kỳ trước -> Cộng điểm cực lớn (Bingo hay bệt)
        if n in last_draw:
            score += 4.0
            
        # --- TIÊU CHÍ 3: CẦU HÀNG XÓM (Neighbor) ---
        # Nếu số bên cạnh (n-1 hoặc n+1) vừa ra kỳ trước -> Cộng điểm nhẹ
        # Ví dụ: Kỳ trước ra 15, thì 14 và 16 có khả năng ra theo
        if (n-1) in last_draw or (n+1) in last_draw:
            score += 1.5
            
        # --- TIÊU CHÍ 4: NGẪU NHIÊN (Yếu tố may mắn) ---
        # Cộng thêm một chút random để tránh AI bị cứng nhắc
        score += random.uniform(0, 1.0)
        
        scores[n] = score

    # 3. Sắp xếp theo điểm số
    ranked_nums = sorted(scores, key=scores.get, reverse=True)
    
    # 4. BỘ LỌC CÂN BẰNG (Balance Filter)
    # Lấy 25 số điểm cao nhất để lọc lại lần cuối lấy 20 số
    candidates = ranked_nums[:25]
    final_picks = []
    
    odd_count = 0  # Đếm số lẻ
    even_count = 0 # Đếm số chẵn
    
    for num in candidates:
        if len(final_picks) == 20: break
        
        # Kiểm tra cân bằng chẵn lẻ (Không cho phép quá lệch)
        is_odd = (num % 2 != 0)
        
        if is_odd and odd_count < 12: # Không quá 12 số lẻ
            final_picks.append(num)
            odd_count += 1
        elif not is_odd and even_count < 12: # Không quá 12 số chẵn
            final_picks.append(num)
            even_count += 1
            
    # Nếu lọc xong mà vẫn thiếu (do điều kiện chặt quá), bốc thêm cho đủ 20
    if len(final_picks) < 20:
        remain = [x for x in candidates if x not in final_picks]
        final_picks.extend(remain[:20-len(final_picks)])
        
    return final_picks, "AI 2.0 Multi-Factor"

# =================================================
# GIAO DIỆN CHÍNH
# =================================================

st.title("🚀 BINGO AI 2.0")

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

    text_paste = st.text_area(
        "👇 Dán kết quả vào đây:", 
        height=150, 
        placeholder="Chạm vào đây -> Chọn 'Dán'...",
        key=f"input_{st.session_state['text_input_key']}"
    )

    if st.button("🔥 PHÂN TÍCH (THUẬT TOÁN MỚI)", type="primary", use_container_width=True):
        if text_paste.strip():
            res, msg = smart_parse_text(text_paste, input_date)
            if res:
                is_duplicate = False
                if not df.empty and str(res['draw_id']) in df['draw_id'].astype(str).values:
                    is_duplicate = True
                    st.toast(f"Kỳ {res['draw_id']} đã có. Đang tính toán lại...", icon="⚠️")
                
                if not is_duplicate:
                    new_row = {'draw_id': res['draw_id'], 'time': res['time']}
                    for i, n in enumerate(res['nums']): new_row[f'num_{i+1}'] = n
                    new_row['super_num'] = res['super_num']
                    df = pd.concat([pd.DataFrame([new_row]), df], ignore_index=True)
                    save_data(df)
                    st.success(f"✅ Đã lưu kỳ {res['draw_id']}")
                
                # DÙNG THUẬT TOÁN V2 MỚI
                p_nums, method = advanced_prediction_v2(df)
                st.session_state['analysis_result'] = {'nums': p_nums, 'ref_id': res['draw_id']}
                
                if not is_duplicate:
                    st.session_state['text_input_key'] += 1
                    st.rerun()
            else:
                st.error(f"❌ {msg}")
        else:
            st.warning("Hãy dán số vào trước!")

# --- OUTPUT ---
if st.session_state['analysis_result']:
    res = st.session_state['analysis_result']
    st.markdown("---")
    st.header(f"🎯 GỢI Ý (AI 2.0)")
    
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
    
    # Lấy số từ kết quả AI V2
    best_picks = res['nums'][:pick_n]
    final_display = sorted(best_picks)
    
    st.info(f"⚡ Dàn **{pick_n} số** xác suất cao nhất:")
    
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
        st.dataframe(df.head(10)[['draw_id', 'super_num']], use_container_width=True, hide_index=True)
