import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import re
from datetime import datetime

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Bingo Mobile VIP", 
    layout="wide", 
    initial_sidebar_state="collapsed" # Tự thu gọn menu để rộng màn hình điện thoại
)

# --- TÊN FILE DỮ LIỆU ---
DATA_FILE = 'bingo_history.csv'

# --- HÀM 1: QUẢN LÝ DỮ LIỆU (LƯU/XÓA) ---
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

# --- HÀM 2: BÓC TÁCH SỐ TỪ VĂN BẢN COPY ---
def smart_parse_text(text, selected_date):
    try:
        # Xóa hết chữ, chỉ giữ số
        clean_text = re.sub(r'\D', ' ', text)
        numbers = [int(n) for n in clean_text.split() if n.strip()]
        
        draw_id = None
        balls = []
        super_n = 0
        
        # Tìm Mã kỳ quay (là số lớn > 100 triệu)
        potential_ids = [n for n in numbers if n > 100000000]
        if potential_ids: draw_id = str(potential_ids[0])
        
        # Tìm các số kết quả (từ 1 đến 80)
        potential_balls = [n for n in numbers if 1 <= n <= 80]
        
        # Nếu không tìm thấy mã kỳ, tự tạo mã giả
        if not draw_id: draw_id = f"Manual-{int(datetime.now().timestamp())}"
        
        # Lọc trùng nhưng giữ nguyên thứ tự xuất hiện
        seen = set()
        unique_balls = []
        for x in potential_balls:
            if x not in seen:
                unique_balls.append(x)
                seen.add(x)
                if len(unique_balls) == 20: break
        
        balls = sorted(unique_balls)

        # Kiểm tra đủ số lượng tối thiểu
        if len(balls) >= 15:
            # Số siêu cấp tạm lấy là số cuối cùng
            super_n = balls[-1] if balls else 0
            final_time = datetime.combine(selected_date, datetime.now().time())
            return {'draw_id': draw_id, 'time': final_time, 'nums': balls, 'super_num': super_n}, "OK"
        else:
            return None, f"Lỗi: Chỉ tìm thấy {len(balls)} số (Cần 20). Hãy copy lại."
            
    except Exception as e: return None, str(e)

# --- HÀM 3: THUẬT TOÁN PHÂN TÍCH (AI RANKING) ---
def advanced_prediction(df):
    if df.empty: return [], "Chưa có dữ liệu"
    
    # 1. Lấy dữ liệu 50 kỳ gần nhất
    recent_df = df.head(50)
    all_nums = [n for i in range(1, 21) for n in recent_df[f'num_{i}']]
    freq = pd.Series(all_nums).value_counts()
    
    scores = {}
    last_res = [df.iloc[0][f'num_{i}'] for i in range(1, 21)]
    
    # 2. Tính điểm cho từng số (1-80)
    for n in range(1, 81):
        # Điểm = Tần suất + Điểm bệt (nếu vừa ra kỳ trước)
        score = freq.get(n, 0) * 1.5 
        if n in last_res: score += 5 
        # Thêm chút ngẫu nhiên để không bị cứng nhắc
        scores[n] = score + random.random()
        
    # 3. Xếp hạng 20 số có điểm cao nhất
    top_20_ranked = sorted(scores, key=scores.get, reverse=True)[:20]
    return top_20_ranked, "AI Ranking"

# =================================================
# GIAO DIỆN CHÍNH (ĐƯỢC TỐI ƯU CHO ĐIỆN THOẠI)
# =================================================

st.title("📱 BINGO MOBILE VIP")

# Khởi tạo kho lưu trữ kết quả phân tích
if 'analysis_result' not in st.session_state: st.session_state['analysis_result'] = None

df = load_data()

# --- KHU VỰC 1: NHẬP LIỆU ---
with st.container(border=True):
    st.write("### 1. Nhập Kết Quả")
    
    # Chọn ngày
    input_date = st.date_input("Ngày:", datetime.now(), label_visibility="collapsed")
    
    # Ô dán to đùng cho dễ bấm
    text_paste = st.text_area(
        "", 
        height=120, 
        placeholder="Chạm vào đây -> Dán kết quả copy từ Web..."
    )

    # Nút bấm ĐỎ to hết cỡ
    if st.button("🚀 LƯU & PHÂN TÍCH NGAY", type="primary", use_container_width=True):
        if text_paste.strip():
            res, msg = smart_parse_text(text_paste, input_date)
            if res:
                # Kiểm tra xem kỳ này đã lưu chưa
                if not df.empty and str(res['draw_id']) in df['draw_id'].astype(str).values:
                    st.toast(f"Kỳ {res['draw_id']} đã có rồi!", icon="⚠️")
                    # Vẫn chạy phân tích lại cho người dùng xem
                    p_nums, method = advanced_prediction(df)
                    st.session_state['analysis_result'] = {'nums': p_nums, 'ref_id': res['draw_id']}
                else:
                    # Lưu dữ liệu mới
                    new_row = {'draw_id': res['draw_id'], 'time': res['time']}
                    for i, n in enumerate(res['nums']): new_row[f'num_{i+1}'] = n
                    new_row['super_num'] = res['super_num']
                    df = pd.concat([pd.DataFrame([new_row]), df], ignore_index=True)
                    save_data(df)
                    st.success(f"✅ Đã lưu kỳ {res['draw_id']}")
                    
                    # Chạy phân tích
                    p_nums, method = advanced_prediction(df)
                    st.session_state['analysis_result'] = {'nums': p_nums, 'ref_id': res['draw_id']}
                    st.rerun()
            else:
                st.error(f"❌ {msg}")
        else:
            st.warning("Bạn chưa dán dữ liệu nào cả!")

# --- KHU VỰC 2: KẾT QUẢ DỰ ĐOÁN (QUAN TRỌNG NHẤT) ---
if st.session_state['analysis_result']:
    res = st.session_state['analysis_result']
    st.markdown("---")
    st.subheader(f"🔮 DỰ ĐOÁN (Sau kỳ {res['ref_id']})")
    
    # --- MENU CHỌN CÁCH CHƠI ---
    # Danh sách các kiểu chơi
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
    
    # Selectbox chọn kiểu chơi
    st.write("🎯 **Bạn muốn lấy bao nhiêu số?**")
    mode = st.selectbox("", list(game_modes.keys()), index=0, label_visibility="collapsed")
    
    # Xử lý logic lấy số
    pick_n = game_modes[mode]
    
    # Lấy Top N số tốt nhất từ kết quả AI
    best_picks = res['nums'][:pick_n]
    
    # Sắp xếp lại từ bé đến lớn để bạn dễ dò vé
    final_display = sorted(best_picks)
    
    st.info(f"🔥 Đây là **{pick_n} số sáng nhất** cho bạn:")
    
    # Hiển thị dạng ô vuông đẹp mắt trên điện thoại
    cols = st.columns(4) # Chia 4 cột để không bị bé quá
    for idx, n in enumerate(final_display):
        # Tô đỏ nếu > 40 (Tài), Xanh nếu <= 40 (Xỉu)
        color = "#d63031" if n > 40 else "#0984e3"
        with cols[idx % 4]:
             st.markdown(
                 f"<div style='text-align: center; font-size: 20px; font-weight: bold; color: white; background-color: {color}; border-radius: 10px; padding: 10px; margin-bottom: 8px; box-shadow: 2px 2px 5px rgba(0,0,0,0.2);'>"
                 f"{n:02d}"
                 f"</div>", 
                 unsafe_allow_html=True
             )
    
    # Thống kê nhanh Tài/Xỉu cho dàn số gợi ý
    if pick_n >= 5:
        big = len([n for n in final_display if n > 40])
        st.caption(f"📊 Phân tích dàn này: {big} Tài - {pick_n-big} Xỉu")

# --- KHU VỰC 3: CÔNG CỤ QUẢN LÝ (ẨN CHO GỌN) ---
st.markdown("---")
with st.expander("🛠 Công cụ sửa lỗi & Lịch sử"):
    c1, c2 = st.columns(2)
    with c1:
        if st.button("↩️ Xóa dòng sai gần nhất"):
            ok, del_id = delete_last_row()
            if ok: st.success(f"Đã xóa {del_id}"); st.rerun()
    with c2:
        if st.button("🗑 Reset (Xóa tất cả)"):
            delete_all_data(); st.success("Sạch sẽ!"); st.rerun()
            
    st.write(f"**Dữ liệu ngày {input_date.strftime('%d/%m')}**")
    
    # Hiển thị bảng dữ liệu
    d_show = df[df['time'].dt.date == input_date] if not df.empty else pd.DataFrame()
    if not d_show.empty:
        st.dataframe(d_show[['draw_id', 'super_num']], use_container_width=True, hide_index=True)
    else:
        st.caption("Chưa có dữ liệu hôm nay.")
