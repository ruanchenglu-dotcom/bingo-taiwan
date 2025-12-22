import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import re
from datetime import datetime

# --- CẤU HÌNH TRANG (GIAO DIỆN MOBILE) ---
st.set_page_config(
    page_title="Bingo Mobile VIP", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# --- TÊN FILE DỮ LIỆU ---
DATA_FILE = 'bingo_history.csv'

# --- KHỐI XỬ LÝ DỮ LIỆU (GIỮ NGUYÊN) ---
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

# --- KHỐI XỬ LÝ TÁCH SỐ THÔNG MINH (GIỮ NGUYÊN) ---
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
            return None, f"Lỗi: Chỉ tìm thấy {len(balls)} số. Hãy copy lại."
            
    except Exception as e: return None, str(e)

# --- KHỐI THUẬT TOÁN AI (GIỮ NGUYÊN) ---
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
        
    top_20_ranked = sorted(scores, key=scores.get, reverse=True)[:20]
    return top_20_ranked, "AI Ranking"

# =================================================
# GIAO DIỆN CHÍNH (ĐƯỢC LÀM TO DỄ BẤM)
# =================================================

st.title("📱 BINGO VIP")

if 'analysis_result' not in st.session_state: st.session_state['analysis_result'] = None
if 'text_input_key' not in st.session_state: st.session_state['text_input_key'] = 0

df = load_data()

# --- KHU VỰC 1: NHẬP LIỆU TO RÕ ---
with st.container(border=True):
    st.write("### 1. Nhập Số")
    
    # Nút xóa nhanh để nhập lại
    col_date, col_clear = st.columns([2, 1])
    with col_date:
        input_date = st.date_input("Ngày:", datetime.now(), label_visibility="collapsed")
    with col_clear:
        if st.button("🗑 Xóa ô", use_container_width=True):
            st.session_state['text_input_key'] += 1
            st.rerun()

    # Ô NHẬP LIỆU SIÊU TO (Height 200)
    text_paste = st.text_area(
        "👇 CHẠM VÀO KHOẢNG TRẮNG NÀY ĐỂ DÁN 👇", 
        height=200, 
        placeholder="1. Copy kết quả trên web xổ số\n2. Chạm vào đây\n3. Chọn 'Dán' (hoặc bấm vào số gợi ý trên bàn phím)",
        key=f"input_{st.session_state['text_input_key']}"
    )

    # NÚT PHÂN TÍCH (Màu đỏ, To hết cỡ)
    if st.button("🚀 PHÂN TÍCH NGAY", type="primary", use_container_width=True):
        if text_paste.strip():
            res, msg = smart_parse_text(text_paste, input_date)
            if res:
                # Kiểm tra trùng lặp
                is_duplicate = False
                if not df.empty and str(res['draw_id']) in df['draw_id'].astype(str).values:
                    is_duplicate = True
                    st.toast(f"Kỳ {res['draw_id']} đã có! Đang phân tích lại...", icon="⚠️")
                
                # Lưu nếu chưa có
                if not is_duplicate:
                    new_row = {'draw_id': res['draw_id'], 'time': res['time']}
                    for i, n in enumerate(res['nums']): new_row[f'num_{i+1}'] = n
                    new_row['super_num'] = res['super_num']
                    df = pd.concat([pd.DataFrame([new_row]), df], ignore_index=True)
                    save_data(df)
                    st.success(f"✅ Đã lưu kỳ {res['draw_id']}")
                
                # CHẠY PHÂN TÍCH
                p_nums, method = advanced_prediction(df)
                st.session_state['analysis_result'] = {'nums': p_nums, 'ref_id': res['draw_id']}
                
                # Nếu không phải trùng lặp thì reload để xóa ô nhập cho sạch
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
    
    # --- MENU CHỌN CÁCH CHƠI (GIỮ NGUYÊN) ---
    game_modes = {
        "6 Số (6 Tinh)": 6, 
        "10 Số (10 Tinh)": 10,
        "5 Số (5 Tinh)": 5, 
        "4 Số (4 Tinh)": 4, 
        "3 Số (3 Tinh)": 3, 
        "2 Số (2 Tinh)": 2, 
        "1 Số (1 Tinh)": 1,
        "Dàn Đầy Đủ (20 số)": 20
    }
    
    # Selectbox chọn cách chơi
    st.write("🎯 **Chọn cách đánh:**")
    mode = st.selectbox("", list(game_modes.keys()), index=0, label_visibility="collapsed")
    
    pick_n = game_modes[mode]
    best_picks = res['nums'][:pick_n]
    final_display = sorted(best_picks)
    
    st.info(f"🔥 Dàn **{pick_n} số** đẹp nhất:")
    
    # Hiển thị số (Chia cột đẹp)
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
with st.expander("🛠 Công cụ sửa lỗi & Lịch sử"):
    c1, c2 = st.columns(2)
    with c1:
        if st.button("↩️ Xóa kỳ sai gần nhất"):
            ok, del_id = delete_last_row()
            if ok: st.success(f"Đã xóa {del_id}"); st.rerun()
    with c2:
        if st.button("🗑 Xóa TẤT CẢ"):
            delete_all_data(); st.success("Đã xóa sạch!"); st.rerun()
            
    st.write(f"**Dữ liệu hôm nay:**")
    # Hiện bảng
    if not df.empty:
        # Chỉ hiện 10 kỳ mới nhất cho đỡ rối
        st.dataframe(df.head(10)[['draw_id', 'super_num']], use_container_width=True, hide_index=True)
    else:
        st.caption("Chưa có dữ liệu.")
