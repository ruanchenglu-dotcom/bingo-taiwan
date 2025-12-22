import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import re
from datetime import datetime

# --- CẤU HÌNH TRANG (MOBILE) ---
st.set_page_config(
    page_title="Bingo Mobile VIP", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

DATA_FILE = 'bingo_history.csv'

# --- KHỐI XỬ LÝ DỮ LIỆU ---
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

# --- KHỐI TÁCH SỐ TỪ VĂN BẢN COPY ---
def smart_parse_text(text, selected_date):
    try:
        clean_text = re.sub(r'\D', ' ', text)
        numbers = [int(n) for n in clean_text.split() if n.strip()]
        
        draw_id = None
        balls = []
        super_n = 0
        
        # Tìm mã kỳ
        potential_ids = [n for n in numbers if n > 100000000]
        if potential_ids: draw_id = str(potential_ids[0])
        
        # Tìm số kết quả
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
            return None, f"Lỗi: Chỉ tìm thấy {len(balls)} số. Copy chưa chuẩn."
            
    except Exception as e: return None, str(e)

# --- KHỐI THUẬT TOÁN AI (XẾP HẠNG 80 SỐ) ---
def advanced_prediction(df):
    if df.empty: return [], "Chưa có dữ liệu"
    
    recent_df = df.head(50)
    all_nums = [n for i in range(1, 21) for n in recent_df[f'num_{i}']]
    freq = pd.Series(all_nums).value_counts()
    
    scores = {}
    last_res = [df.iloc[0][f'num_{i}'] for i in range(1, 21)]
    
    for n in range(1, 81):
        score = freq.get(n, 0) * 1.5 
        if n in last_res: score += 5 
        scores[n] = score + random.random()
        
    # Trả về Top 20 số điểm cao nhất
    top_20_ranked = sorted(scores, key=scores.get, reverse=True)[:20]
    return top_20_ranked, "AI Ranking"

# =================================================
# GIAO DIỆN CHÍNH
# =================================================

st.title("📱 BINGO VIP FULL")

if 'analysis_result' not in st.session_state: st.session_state['analysis_result'] = None
if 'text_input_key' not in st.session_state: st.session_state['text_input_key'] = 0

df = load_data()

# --- KHU VỰC 1: NHẬP LIỆU ---
with st.container(border=True):
    st.write("### 1. Nhập Số")
    
    col_date, col_clear = st.columns([2, 1])
    with col_date:
        input_date = st.date_input("Ngày:", datetime.now(), label_visibility="collapsed")
    with col_clear:
        if st.button("🗑 Xóa ô", use_container_width=True):
            st.session_state['text_input_key'] += 1
            st.rerun()

    text_paste = st.text_area(
        "👇 CHẠM VÀO ĐÂY ĐỂ DÁN 👇", 
        height=200, 
        placeholder="Chạm vào đây -> Chọn 'Dán' (hoặc bấm gợi ý trên bàn phím)",
        key=f"input_{st.session_state['text_input_key']}"
    )

    if st.button("🚀 PHÂN TÍCH NGAY", type="primary", use_container_width=True):
        if text_paste.strip():
            res, msg = smart_parse_text(text_paste, input_date)
            if res:
                is_duplicate = False
                if not df.empty and str(res['draw_id']) in df['draw_id'].astype(str).values:
                    is_duplicate = True
                    st.toast(f"Kỳ {res['draw_id']} đã có! Đang phân tích lại...", icon="⚠️")
                
                if not is_duplicate:
                    new_row = {'draw_id': res['draw_id'], 'time': res['time']}
                    for i, n in enumerate(res['nums']): new_row[f'num_{i+1}'] = n
                    new_row['super_num'] = res['super_num']
                    df = pd.concat([pd.DataFrame([new_row]), df], ignore_index=True)
                    save_data(df)
                    st.success(f"✅ Đã lưu kỳ {res['draw_id']}")
                
                p_nums, method = advanced_prediction(df)
                st.session_state['analysis_result'] = {'nums': p_nums, 'ref_id': res['draw_id']}
                
                if not is_duplicate:
                    st.session_state['text_input_key'] += 1
                    st.rerun()
            else:
                st.error(f"❌ {msg}")
        else:
            st.warning("Bạn chưa dán số nào cả!")

# --- KHU VỰC 2: KẾT QUẢ & CHỌN CÁCH CHƠI ---
if st.session_state['analysis_result']:
    res = st.session_state['analysis_result']
    st.markdown("---")
    
    st.header(f"🔮 KẾT QUẢ (Sau kỳ {res['ref_id']})")
    
    # --- MENU CHỌN ĐẦY ĐỦ TỪ 1 ĐẾN 10 ---
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
        "Xem Full 20 số": 20
    }
    
    st.write("🎯 **Chọn cách đánh:**")
    # Mặc định chọn 6 Tinh (index=4)
    mode = st.selectbox("", list(game_modes.keys()), index=4, label_visibility="collapsed")
    
    pick_n = game_modes[mode]
    
    # Lấy Top N số tốt nhất
    best_picks = res['nums'][:pick_n]
    
    # Sắp xếp từ bé đến lớn để dễ dò
    final_display = sorted(best_picks)
    
    st.info(f"🔥 Dàn **{pick_n} số** sáng nhất:")
    
    # Hiển thị số
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

# --- KHU VỰC 3: CÔNG CỤ QUẢN LÝ ---
st.markdown("---")
with st.expander("🛠 Công cụ & Lịch sử"):
    c1, c2 = st.columns(2)
    with c1:
        if st.button("↩️ Xóa kỳ sai"):
            ok, del_id = delete_last_row()
            if ok: st.success(f"Đã xóa {del_id}"); st.rerun()
    with c2:
        if st.button("🗑 Xóa TẤT CẢ"):
            delete_all_data(); st.success("Đã xóa sạch!"); st.rerun()
            
    st.write(f"**Dữ liệu hôm nay:**")
    if not df.empty:
        st.dataframe(df.head(10)[['draw_id', 'super_num']], use_container_width=True, hide_index=True)
    else:
        st.caption("Chưa có dữ liệu.")
