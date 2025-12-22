import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import re
from datetime import datetime
import webbrowser # Thư viện mở trình duyệt web

# --- CẤU HÌNH ---
DATA_FILE = 'bingo_history.csv'
ST_PAGE_TITLE = "Bingo Master - Đa Chiến Thuật"
TARGET_URL = "https://www.taiwanlottery.com.tw/Lotto/BINGOBINGO/drawing.aspx"

# --- HÀM QUẢN LÝ DỮ LIỆU ---
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

# --- HÀM XỬ LÝ TEXT THÔNG MINH ---
def smart_parse_text(text, selected_date):
    try:
        # Làm sạch: Chỉ giữ lại số
        clean_text = re.sub(r'\D', ' ', text)
        numbers = [int(n) for n in clean_text.split() if n.strip()]
        
        draw_id = None
        balls = []
        super_n = 0
        
        # Tìm Mã kỳ quay (> 100.000.000)
        potential_ids = [n for n in numbers if n > 100000000]
        if potential_ids:
            draw_id = str(potential_ids[0])
        
        # Lấy số kết quả (<= 80)
        potential_balls = [n for n in numbers if 1 <= n <= 80]
        
        if not draw_id: 
            draw_id = f"Manual-{int(datetime.now().timestamp())}"
        
        # Lọc trùng giữ thứ tự
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
            return None, f"Chỉ tìm thấy {len(balls)} số. Hãy copy đầy đủ."
            
    except Exception as e: return None, str(e)

# --- THUẬT TOÁN PHÂN TÍCH (NÂNG CẤP) ---
def advanced_prediction(df):
    """
    Trả về danh sách 20 số tốt nhất được sắp xếp theo điểm số (Score)
    để sau này cắt ra theo cách chơi (Ví dụ chọn 6 số thì lấy Top 6).
    """
    if df.empty: return [], "Chưa có dữ liệu"
    
    # Xét 50 kỳ gần nhất
    recent_df = df.head(50)
    all_nums = [n for i in range(1, 21) for n in recent_df[f'num_{i}']]
    freq = pd.Series(all_nums).value_counts()
    
    scores = {}
    last_res = [df.iloc[0][f'num_{i}'] for i in range(1, 21)]
    
    for n in range(1, 81):
        # Công thức điểm: Tần suất * 1.5 + Điểm bệt (nếu vừa ra) + Random nhẹ
        score = freq.get(n, 0) * 1.5 
        if n in last_res: score += 5
        scores[n] = score + random.random()
    
    # Sắp xếp các số từ điểm cao nhất đến thấp nhất (Quan trọng!)
    ranked_nums = sorted(scores, key=scores.get, reverse=True)
    
    # Lấy 20 số tốt nhất
    top_20_best = ranked_nums[:20]
    
    return top_20_best, "AI Ranking"

# --- GIAO DIỆN CHÍNH ---
st.set_page_config(page_title=ST_PAGE_TITLE, layout="wide")
st.title("🎲 BINGO MASTER - CHỌN CÁCH CHƠI")
st.markdown("---")

if 'analysis_result' not in st.session_state: st.session_state['analysis_result'] = None

df = load_data()

# --- SIDEBAR: CÔNG CỤ ---
st.sidebar.header("⚙️ Công Cụ")
if st.sidebar.button("↩️ Xóa dòng sai gần nhất"):
    ok, del_id = delete_last_row()
    if ok: st.sidebar.success(f"Đã xóa {del_id}"); st.rerun()
    
if st.sidebar.checkbox("Xóa tất cả dữ liệu"):
    if st.sidebar.button("🔥 XÁC NHẬN XÓA"):
        delete_all_data(); st.sidebar.success("Đã xóa sạch!"); st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("📅 Lịch Sử")
filter_date = st.sidebar.date_input("Ngày:", datetime.now())
view_all = st.sidebar.checkbox("Xem tất cả", value=False)

# --- MAIN LAYOUT ---
col1, col2 = st.columns([1, 1.3])

with col1:
    st.subheader("🚀 BƯỚC 1: LẤY DỮ LIỆU")
    st.info("💡 Mẹo: Bấm nút mở web -> Ctrl+A -> Ctrl+C -> Dán vào dưới.")
    
    if st.button("🌏 MỞ WEB BINGO (TAB MỚI)", type="primary", use_container_width=True):
        webbrowser.open_new_tab(TARGET_URL)
        st.toast("Đã mở web! Hãy copy và quay lại đây.", icon="🚀")

    input_date = st.date_input("Ngày kết quả:", datetime.now())
    text_paste = st.text_area("Dán kết quả vào đây:", height=100, placeholder="Dán nội dung copy từ web...")
    
    if st.button("📥 LƯU & PHÂN TÍCH", use_container_width=True):
        if text_paste.strip():
            res, msg = smart_parse_text(text_paste, input_date)
            if res:
                if not df.empty and str(res['draw_id']) in df['draw_id'].astype(str).values:
                    st.warning(f"Kỳ {res['draw_id']} đã lưu rồi!")
                else:
                    new_row = {'draw_id': res['draw_id'], 'time': res['time']}
                    for i, n in enumerate(res['nums']): new_row[f'num_{i+1}'] = n
                    new_row['super_num'] = res['super_num']
                    df = pd.concat([pd.DataFrame([new_row]), df], ignore_index=True)
                    save_data(df)
                    st.success(f"✅ Đã lưu kỳ {res['draw_id']}")
                    
                    # Tự động chạy phân tích
                    p_nums, method = advanced_prediction(df)
                    st.session_state['analysis_result'] = {'nums': p_nums, 'ref_id': res['draw_id']}
                    st.rerun()
            else:
                st.error(f"❌ {msg}")
        else:
            st.warning("Bạn chưa dán gì cả!")

with col2:
    st.subheader("🔮 BƯỚC 2: GỢI Ý SỐ")
    
    # --- TÍNH NĂNG MỚI: CHỌN CÁCH CHƠI ---
    # Tạo danh sách lựa chọn
    game_modes = {
        "Dàn Đầy Đủ (20 Số)": 20,
        "10 Tinh (Đánh 10 số)": 10,
        "9 Tinh (Đánh 9 số)": 9,
        "8 Tinh (Đánh 8 số)": 8,
        "7 Tinh (Đánh 7 số)": 7,
        "6 Tinh (Đánh 6 số)": 6,
        "5 Tinh (Đánh 5 số)": 5,
        "4 Tinh (Đánh 4 số)": 4,
        "3 Tinh (Đánh 3 số)": 3,
        "2 Tinh (Đánh 2 số)": 2,
        "1 Tinh (Đánh 1 số)": 1
    }
    
    selected_mode = st.selectbox("🎯 Bạn muốn chơi kiểu nào?", list(game_modes.keys()), index=5) # Mặc định để 6 Tinh
    num_to_pick = game_modes[selected_mode]

    st.markdown("---")

    if st.session_state['analysis_result']:
        res = st.session_state['analysis_result']
        full_prediction = res['nums'] # Đây là 20 số tốt nhất đã xếp hạng
        
        # Lấy đúng số lượng cần thiết (Top N số tốt nhất)
        final_suggestion = full_prediction[:num_to_pick]
        
        # Sắp xếp lại theo thứ tự nhỏ đến lớn để dễ dò (sau khi đã lọc được Top N)
        final_suggestion_display = sorted(final_suggestion)
        
        st.success(f"🔥 GỢI Ý {num_to_pick} SỐ NGON NHẤT (Sau kỳ {res['ref_id']})")
        
        # Hiển thị số đẹp
        cols = st.columns(min(num_to_pick, 5)) # Tối đa 5 cột
        for idx, n in enumerate(final_suggestion_display):
            color = "#d63031" if n > 40 else "#0984e3"
            # Tính toán vị trí cột để hiển thị đẹp
            col_idx = idx % 5
            if num_to_pick <= 5: 
                col_idx = idx # Nếu ít số thì dàn đều ra
            
            with cols[col_idx]:
                 st.markdown(f"<div style='text-align: center; font-size: 24px; font-weight: bold; color: {color}; border: 2px solid #ddd; border-radius: 10px; padding: 10px; margin-bottom: 10px; background-color: white;'>{n:02d}</div>", unsafe_allow_html=True)
        
        # Thống kê phụ cho dàn số gợi ý
        if num_to_pick >= 5:
            big = len([n for n in final_suggestion_display if n > 40])
            st.info(f"Phân tích dàn {num_to_pick} số này: 🔴 {big} Tài | 🔵 {num_to_pick-big} Xỉu")
        
    else:
        st.info("👈 Hãy dán kết quả và bấm nút bên trái để xem gợi ý.")

    st.markdown("---")
    # Bảng dữ liệu
    st.write(f"**Dữ liệu ngày {filter_date.strftime('%d/%m/%Y')}**")
    if view_all: d_show = df
    else: d_show = df[df['time'].dt.date == filter_date] if not df.empty else pd.DataFrame()
    
    if not d_show.empty:
        st.dataframe(d_show[['draw_id', 'time', 'super_num'] + [f'num_{i}' for i in range(1, 6)]], height=300, use_container_width=True)
    else:
        st.caption("Chưa có dữ liệu.")