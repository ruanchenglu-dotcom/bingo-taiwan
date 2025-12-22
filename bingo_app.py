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
    page_title="Bingo Mobile VIP Final Fixed", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# Tên file lưu trữ lịch sử
DATA_FILE = 'bingo_history.csv'

# ==============================================================================
# 2. CÁC HÀM QUẢN LÝ DỮ LIỆU
# ==============================================================================
def load_data():
    """Đọc dữ liệu từ file CSV."""
    columns = ['draw_id', 'time'] + [f'num_{i}' for i in range(1, 21)] + ['super_num']
    df = pd.DataFrame(columns=columns)
    
    if os.path.exists(DATA_FILE):
        try:
            loaded_df = pd.read_csv(DATA_FILE)
            if not loaded_df.empty: 
                df = loaded_df
        except Exception: 
            pass
    
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
    
    df = df.dropna(subset=['time'])
    df = df.sort_values(by='time', ascending=False)
    df = df.drop_duplicates(subset=['draw_id'], keep='first')
    return df

def save_data(df):
    """Lưu dữ liệu."""
    df.to_csv(DATA_FILE, index=False)

def delete_last_row():
    """Xóa kỳ mới nhất."""
    df = load_data()
    if not df.empty:
        deleted_id = df.iloc[0]['draw_id']
        df = df.iloc[1:] 
        save_data(df)
        return True, deleted_id
    return False, None

def delete_all_data():
    """Reset toàn bộ."""
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
        return True
    return False

# ==============================================================================
# 3. HÀM XỬ LÝ VĂN BẢN (FIX LỖI SỐ DÍNH LIỀN)
# ==============================================================================
def parse_bulk_text(text, selected_date):
    """
    Hàm quét đa dòng, xử lý cả trường hợp số bị dính liền (010203...).
    """
    found_draws = []
    lines = text.strip().splitlines()
    
    for line in lines:
        try:
            # Bỏ qua dòng trống
            if not line.strip(): continue

            # --- BƯỚC 1: TÌM MÃ KỲ (9 CHỮ SỐ) ---
            # Tìm chuỗi 9 chữ số liên tiếp (Ví dụ: 114072268)
            id_match = re.search(r'\b\d{9}\b', line)
            
            draw_id = None
            if id_match:
                draw_id = id_match.group(0)
            else:
                # Nếu không tìm thấy bằng regex, thử tìm số lớn nhất trong dòng
                nums_in_line = [int(n) for n in re.findall(r'\d+', line)]
                big_nums = [n for n in nums_in_line if n > 100000000]
                if big_nums:
                    draw_id = str(max(big_nums))
                else:
                    continue # Không có ID -> Bỏ qua dòng này

            # --- BƯỚC 2: TÌM 20 SỐ KẾT QUẢ ---
            # Xóa mã kỳ ra khỏi dòng để tránh nhầm lẫn
            line_without_id = line.replace(draw_id, "")
            
            # Lọc lấy tất cả các chữ số còn lại
            digits_only = re.sub(r'\D', '', line_without_id)
            
            potential_balls = []
            
            # Logic xử lý thông minh:
            # Nếu copy dính liền (VD: 010415...), ta cắt từng cặp 2 số
            if len(digits_only) >= 30: # Nếu chuỗi số dài, khả năng cao là dính liền
                # Cắt từng cặp: 01, 04, 15...
                pairs = [digits_only[i:i+2] for i in range(0, len(digits_only), 2)]
                for p in pairs:
                    if len(p) == 2:
                        val = int(p)
                        if 1 <= val <= 80:
                            potential_balls.append(val)
            else:
                # Nếu copy có dấu cách (VD: 01 04 15...), dùng cách tách thông thường
                temp_nums = [int(n) for n in re.findall(r'\d+', line_without_id)]
                potential_balls = [n for n in temp_nums if 1 <= n <= 80]

            # --- BƯỚC 3: LỌC TRÙNG & KIỂM TRA ---
            seen = set()
            unique_balls = []
            for x in potential_balls:
                if x not in seen:
                    unique_balls.append(x)
                    seen.add(x)
                    if len(unique_balls) == 20: break
            
            balls = sorted(unique_balls)
            
            # Phải có ít nhất 15 số mới nhận
            if len(balls) >= 15:
                super_n = balls[-1] if balls else 0
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
# 4. THUẬT TOÁN AI
# ==============================================================================
def advanced_prediction_v2(df):
    if df.empty: return [], "Chưa có dữ liệu"
    
    short_term_df = df.head(15)
    last_draw = [df.iloc[0][f'num_{i}'] for i in range(1, 21)]
    
    all_short_nums = [n for i in range(1, 21) for n in short_term_df[f'num_{i}']]
    freq_short = pd.Series(all_short_nums).value_counts()
    
    scores = {}
    for n in range(1, 81):
        score = 0
        count = freq_short.get(n, 0)
        score += count * 2.0 
        if n in last_draw: score += 4.0 
        if (n-1) in last_draw or (n+1) in last_draw: score += 1.5 
        score += random.uniform(0, 1.0)
        scores[n] = score

    ranked_nums = sorted(scores, key=scores.get, reverse=True)
    
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
# 5. GIAO DIỆN CHÍNH (ĐÃ CẬP NHẬT ĐẦY ĐỦ MENU)
# ==============================================================================

st.title("📱 BINGO VIP PRO FIXED")

if 'analysis_result' not in st.session_state: st.session_state['analysis_result'] = None
if 'text_input_key' not in st.session_state: st.session_state['text_input_key'] = 0

df = load_data()

# --- KHUNG NHẬP LIỆU ---
with st.container(border=True):
    col_date, col_clear = st.columns([2, 1])
    with col_date:
        input_date = st.date_input("Ngày:", datetime.now(), label_visibility="collapsed")
    with col_clear:
        if st.button("🗑 Xóa ô", use_container_width=True):
            st.session_state['text_input_key'] += 1
            st.rerun()

    text_paste = st.text_area(
        "", 
        height=150, 
        placeholder="Dán toàn bộ bảng kết quả vào đây (Máy sẽ tự tách số dính liền)...",
        key=f"input_{st.session_state['text_input_key']}"
    )

    if st.button("🔥 LƯU TẤT CẢ & PHÂN TÍCH", type="primary", use_container_width=True):
        if text_paste.strip():
            # Xử lý đa dòng + fix lỗi dính số
            draws_list = parse_bulk_text(text_paste, input_date)
            
            if len(draws_list) > 0:
                count_new = 0
                latest_draw_id = None
                
                # Sắp xếp ID tăng dần để lưu
                draws_list_sorted = sorted(draws_list, key=lambda x: int(x['draw_id']))
                
                for draw in draws_list_sorted:
                    if not df.empty and str(draw['draw_id']) in df['draw_id'].astype(str).values:
                        continue 
                    
                    new_row = {'draw_id': draw['draw_id'], 'time': draw['time']}
                    for i, n in enumerate(draw['nums']): 
                        new_row[f'num_{i+1}'] = n
                    new_row['super_num'] = draw['super_num']
                    
                    df = pd.concat([pd.DataFrame([new_row]), df], ignore_index=True)
                    count_new += 1
                    
                    if latest_draw_id is None or int(draw['draw_id']) > int(latest_draw_id):
                        latest_draw_id = draw['draw_id']
                
                if latest_draw_id is None:
                     latest_draw_id = max([d['draw_id'] for d in draws_list], key=lambda x: int(x))

                if count_new > 0:
                    df = df.sort_values(by='time', ascending=False)
                    save_data(df)
                    st.success(f"✅ Đã thêm {count_new} kỳ mới! Tìm thấy tổng cộng {len(draws_list)} dòng.")
                else:
                    st.warning("⚠️ Dữ liệu đã có sẵn. Đang phân tích kỳ mới nhất...")

                p_nums, method = advanced_prediction_v2(df)
                st.session_state['analysis_result'] = {'nums': p_nums, 'ref_id': latest_draw_id}
                st.session_state['text_input_key'] += 1
                st.rerun()
            else:
                st.error("❌ Không đọc được dữ liệu. Có thể do copy thiếu hoặc lỗi định dạng.")
        else:
            st.warning("Bạn chưa dán dữ liệu!")

# --- KHUNG KẾT QUẢ & MENU CHỌN CÁCH CHƠI ---
if st.session_state['analysis_result']:
    res = st.session_state['analysis_result']
    st.markdown("---")
    st.header(f"🎯 DỰ ĐOÁN (Sau kỳ {res['ref_id']})")
    
    # --- MENU ĐẦY ĐỦ TỪ 1 ĐẾN 10 ---
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
    
    st.write("🎯 **Chọn cách đánh:**")
    # Mặc định chọn 6 Tinh (Index 4)
    mode_name = st.selectbox("", list(game_modes.keys()), index=4, label_visibility="collapsed")
    pick_n = game_modes[mode_name]
    
    best_picks = res['nums'][:pick_n]
    final_display = sorted(best_picks)
    
    st.info(f"⚡ Gợi ý dàn **{pick_n} số**:")
    
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
        st.caption(f"📊 {big} Tài - {pick_n-big} Xỉu")

# --- KHUNG LỊCH SỬ ---
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
        st.write("📋 **Lịch sử nhập (Hiện đủ 20 số):**")
        display_cols = ['draw_id', 'super_num'] + [f'num_{i}' for i in range(1, 21)]
        st.dataframe(
            df[display_cols].head(50), 
            use_container_width=True, 
            hide_index=True,
            column_config={"draw_id": "Mã Kỳ", "super_num": "Siêu"}
        )
    else:
        st.caption("Chưa có dữ liệu.")
