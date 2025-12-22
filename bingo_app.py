import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import re
from datetime import datetime

# ==============================================================================
# 1. CẤU HÌNH TRANG WEB & FILE DỮ LIỆU
# ==============================================================================
st.set_page_config(
    page_title="Bingo Mobile VIP", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# Tên file lưu trữ lịch sử
DATA_FILE = 'bingo_history.csv'

# ==============================================================================
# 2. CÁC HÀM QUẢN LÝ DỮ LIỆU (ĐỌC, GHI, XÓA)
# ==============================================================================
def load_data():
    """
    Hàm đọc dữ liệu từ file CSV lên.
    Nếu file chưa có thì tạo bảng mới.
    """
    # Tạo đầy đủ cột cho 20 số
    columns = ['draw_id', 'time'] + [f'num_{i}' for i in range(1, 21)] + ['super_num']
    df = pd.DataFrame(columns=columns)
    
    if os.path.exists(DATA_FILE):
        try:
            loaded_df = pd.read_csv(DATA_FILE)
            if not loaded_df.empty: 
                df = loaded_df
        except Exception: 
            pass
    
    # Chuyển đổi cột thời gian sang định dạng ngày tháng
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
    
    # Sắp xếp dữ liệu: Mới nhất lên đầu
    df = df.dropna(subset=['time'])
    df = df.sort_values(by='time', ascending=False)
    # Loại bỏ các dòng trùng lặp mã kỳ
    df = df.drop_duplicates(subset=['draw_id'], keep='first')
    
    return df

def save_data(df):
    """
    Hàm lưu dữ liệu xuống file CSV.
    """
    df.to_csv(DATA_FILE, index=False)

def delete_last_row():
    """
    Hàm xóa kỳ mới nhất (dòng đầu tiên) nếu nhập sai.
    """
    df = load_data()
    if not df.empty:
        deleted_id = df.iloc[0]['draw_id']
        df = df.iloc[1:] # Bỏ dòng đầu, giữ lại phần còn lại
        save_data(df)
        return True, deleted_id
    return False, None

def delete_all_data():
    """
    Hàm xóa toàn bộ dữ liệu (Reset máy).
    """
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
        return True
    return False

# ==============================================================================
# 3. HÀM XỬ LÝ VĂN BẢN ĐA LUỒNG (SỬA LỖI CHỈ ĐỌC 1 KỲ)
# ==============================================================================
def parse_bulk_text(text, selected_date):
    """
    Hàm này quét từng dòng trong văn bản copy để tìm NHIỀU kỳ quay.
    Trả về một danh sách (List) chứa tất cả các kỳ tìm được.
    """
    found_draws = []
    
    # Tách văn bản thành từng dòng (dựa vào dấu xuống dòng)
    lines = text.strip().split('\n')
    
    for line in lines:
        try:
            # 1. Làm sạch dòng: Thay thế chữ cái bằng khoảng trắng, chỉ giữ số
            clean_line = re.sub(r'\D', ' ', line)
            
            # 2. Tách thành danh sách các con số
            numbers = [int(n) for n in clean_line.split() if n.strip()]
            
            # Nếu dòng quá ngắn (ít hơn 15 số) thì bỏ qua, chắc là dòng rác
            if len(numbers) < 15:
                continue
            
            draw_id = None
            balls = []
            super_n = 0
            
            # 3. Tìm Mã Kỳ Quay (Số lớn > 100.000.000, ví dụ 114072268)
            # Chúng ta lấy số lớn nhất trong dòng làm mã kỳ
            potential_ids = [n for n in numbers if n > 100000000]
            if potential_ids:
                draw_id = str(max(potential_ids)) # Lấy ID lớn nhất cho chắc
            else:
                # Nếu dòng này không có mã kỳ > 100tr, bỏ qua
                continue
            
            # 4. Tìm 20 Số Kết Quả (Các số từ 1 đến 80)
            potential_balls = [n for n in numbers if 1 <= n <= 80]
            
            # Lọc trùng số trong cùng 1 dòng nhưng giữ thứ tự
            seen = set()
            unique_balls = []
            for x in potential_balls:
                if x not in seen:
                    unique_balls.append(x)
                    seen.add(x)
                    # Chỉ lấy đủ 20 số đầu tiên
                    if len(unique_balls) == 20: 
                        break
            
            balls = sorted(unique_balls)
            
            # 5. Kiểm tra tính hợp lệ (Phải có đủ 20 số hoặc ít nhất 15 số)
            if len(balls) >= 15:
                # Số siêu cấp tạm lấy là số cuối cùng (hoặc logic khác tùy bạn)
                super_n = balls[-1] if balls else 0
                
                # Tạo thời gian giả lập
                final_time = datetime.combine(selected_date, datetime.now().time())
                
                # Thêm vào danh sách kết quả
                found_draws.append({
                    'draw_id': draw_id,
                    'time': final_time,
                    'nums': balls,
                    'super_num': super_n
                })
        except Exception:
            # Nếu dòng nào lỗi thì bỏ qua dòng đó, chạy tiếp dòng sau
            continue
            
    return found_draws

# ==============================================================================
# 4. THUẬT TOÁN AI 2.0 (PHÂN TÍCH ĐA CHIỀU)
# ==============================================================================
def advanced_prediction_v2(df):
    """
    Thuật toán dự đoán dựa trên: Hot Trend, Cầu Bệt, Hàng Xóm và Cân Bằng Chẵn Lẻ.
    """
    if df.empty: 
        return [], "Chưa có dữ liệu"
    
    # Lấy 15 kỳ gần nhất để phân tích xu hướng
    short_term_df = df.head(15)
    
    # Lấy danh sách số của kỳ vừa quay nhất (để bắt cầu bệt)
    last_draw = [df.iloc[0][f'num_{i}'] for i in range(1, 21)]
    
    # Tính tần suất xuất hiện
    all_short_nums = [n for i in range(1, 21) for n in short_term_df[f'num_{i}']]
    freq_short = pd.Series(all_short_nums).value_counts()
    
    scores = {}
    
    # Chấm điểm cho từng số từ 01 đến 80
    for n in range(1, 81):
        score = 0
        
        # Tiêu chí 1: Tần suất (Ra càng nhiều điểm càng cao)
        count = freq_short.get(n, 0)
        score += count * 2.0 
        
        # Tiêu chí 2: Cầu Bệt (Vừa ra kỳ trước thì dễ ra lại)
        if n in last_draw:
            score += 4.0
            
        # Tiêu chí 3: Cầu Hàng Xóm (Ra 10 thì dễ kéo theo 09, 11)
        if (n-1) in last_draw or (n+1) in last_draw:
            score += 1.5
            
        # Tiêu chí 4: Ngẫu nhiên (Để dàn số tự nhiên hơn)
        score += random.uniform(0, 1.0)
        
        scores[n] = score

    # Sắp xếp các số theo điểm từ cao xuống thấp
    ranked_nums = sorted(scores, key=scores.get, reverse=True)
    
    # Lọc cân bằng Chẵn/Lẻ (Không để quá lệch)
    candidates = ranked_nums[:25] # Lấy 25 ứng viên sáng giá nhất
    final_picks = []
    odd_count = 0
    even_count = 0
    
    for num in candidates:
        if len(final_picks) == 20: 
            break
            
        is_odd = (num % 2 != 0)
        
        # Logic: Không cho phép quá 12 số Lẻ hoặc 12 số Chẵn trong dàn 20 số
        if is_odd and odd_count < 12:
            final_picks.append(num)
            odd_count += 1
        elif not is_odd and even_count < 12:
            final_picks.append(num)
            even_count += 1
            
    # Nếu lọc xong mà vẫn chưa đủ 20 số, lấy thêm từ danh sách dự bị
    if len(final_picks) < 20:
        remain = [x for x in candidates if x not in final_picks]
        final_picks.extend(remain[:20-len(final_picks)])
        
    return final_picks, "AI 2.0 Multi-Factor"

# ==============================================================================
# 5. GIAO DIỆN NGƯỜI DÙNG (STREAMLIT UI)
# ==============================================================================

st.title("📥 BINGO NHẬP LIỆU HÀNG LOẠT")

# Khởi tạo các biến trong phiên làm việc (Session State)
if 'analysis_result' not in st.session_state:
    st.session_state['analysis_result'] = None
if 'text_input_key' not in st.session_state:
    st.session_state['text_input_key'] = 0

# Tải dữ liệu từ file
df = load_data()

# --- KHUNG NHẬP LIỆU ---
with st.container(border=True):
    # Hàng 1: Chọn ngày và Nút Xóa Ô
    col_date, col_clear = st.columns([2, 1])
    with col_date:
        input_date = st.date_input("Ngày:", datetime.now(), label_visibility="collapsed")
    with col_clear:
        if st.button("🗑 Xóa ô nhập", use_container_width=True):
            st.session_state['text_input_key'] += 1
            st.rerun()

    st.caption("💡 Mẹo: Bạn có thể copy CẢ BẢNG (10-20 dòng) dán vào đây, máy sẽ tự tách từng kỳ.")
    
    # Ô nhập liệu văn bản (Text Area)
    text_paste = st.text_area(
        "", 
        height=150, 
        placeholder="Dán toàn bộ bảng kết quả copy từ web vào đây...",
        key=f"input_{st.session_state['text_input_key']}"
    )

    # Nút bấm Phân Tích
    if st.button("🔥 LƯU TẤT CẢ & PHÂN TÍCH", type="primary", use_container_width=True):
        if text_paste.strip():
            # Gọi hàm xử lý đa luồng mới (parse_bulk_text)
            draws_list = parse_bulk_text(text_paste, input_date)
            
            if len(draws_list) > 0:
                count_new = 0
                latest_draw_id = None
                
                # Duyệt qua danh sách các kỳ tìm được (Đảo ngược để lưu kỳ cũ trước)
                # Sắp xếp draws_list theo ID tăng dần để lưu vào DB cho đúng thứ tự thời gian
                draws_list_sorted = sorted(draws_list, key=lambda x: int(x['draw_id']))
                
                for draw in draws_list_sorted:
                    
                    # Kiểm tra xem kỳ này đã có trong máy chưa
                    if not df.empty and str(draw['draw_id']) in df['draw_id'].astype(str).values:
                        continue # Nếu có rồi thì bỏ qua
                    
                    # Tạo dòng dữ liệu mới
                    new_row = {'draw_id': draw['draw_id'], 'time': draw['time']}
                    for i, n in enumerate(draw['nums']): 
                        new_row[f'num_{i+1}'] = n
                    new_row['super_num'] = draw['super_num']
                    
                    # Thêm vào DataFrame chính
                    df = pd.concat([pd.DataFrame([new_row]), df], ignore_index=True)
                    count_new += 1
                    
                    # Cập nhật ID mới nhất để hiển thị phân tích
                    if latest_draw_id is None or int(draw['draw_id']) > int(latest_draw_id):
                        latest_draw_id = draw['draw_id']
                
                # Nếu không có kỳ mới nào được thêm (do đã có hết rồi), lấy kỳ mới nhất trong đám vừa paste
                if latest_draw_id is None:
                     # Lấy ID lớn nhất trong danh sách vừa paste
                     latest_draw_id = max([d['draw_id'] for d in draws_list], key=lambda x: int(x))

                # Lưu dữ liệu xuống file
                if count_new > 0:
                    # Sắp xếp lại lần nữa cho chắc chắn (Mới nhất lên đầu)
                    df = df.sort_values(by='time', ascending=False)
                    save_data(df)
                    st.success(f"✅ Đã thêm thành công {count_new} kỳ mới vào lịch sử!")
                else:
                    st.warning("⚠️ Các kỳ này đã có trong máy rồi. Đang phân tích kỳ mới nhất...")

                # CHẠY PHÂN TÍCH AI (Dựa trên dữ liệu mới nhất)
                p_nums, method = advanced_prediction_v2(df)
                st.session_state['analysis_result'] = {'nums': p_nums, 'ref_id': latest_draw_id}
                
                # Xóa sạch ô nhập liệu để nhập tiếp
                st.session_state['text_input_key'] += 1
                st.rerun()
            else:
                st.error("❌ Không đọc được dữ liệu nào. Hãy chắc chắn bạn copy đúng bảng số.")
        else:
            st.warning("Hãy dán dữ liệu vào ô trống trước!")

# --- KHUNG HIỂN THỊ KẾT QUẢ ---
if st.session_state['analysis_result']:
    res = st.session_state['analysis_result']
    st.markdown("---")
    st.header(f"🎯 DỰ ĐOÁN (Sau kỳ {res['ref_id']})")
    
    # --- MENU CHỌN CÁCH CHƠI (ĐÃ BỔ SUNG ĐẦY ĐỦ 7, 8, 9 SAO) ---
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
    
    st.write("🎯 **Chọn dàn đánh (Đã sắp xếp đầy đủ):**")
    # Mặc định chọn 6 Tinh (index=4 trong danh sách trên)
    mode = st.selectbox("", list(game_modes.keys()), index=4, label_visibility="collapsed")
    pick_n = game_modes[mode]
    
    # Lấy Top N số tốt nhất từ kết quả AI
    best_picks = res['nums'][:pick_n]
    
    # Sắp xếp từ bé đến lớn để dễ dò vé
    final_display = sorted(best_picks)
    
    st.info(f"⚡ Dàn **{pick_n} số** xác suất cao nhất:")
    
    # Hiển thị số dạng ô vuông màu sắc
    cols = st.columns(4)
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
    
    # Thống kê nhanh Tài/Xỉu
    if pick_n >= 5:
        big = len([n for n in final_display if n > 40])
        st.caption(f"📊 Thống kê dàn này: {big} Tài - {pick_n-big} Xỉu")

# --- KHUNG CÔNG CỤ & LỊCH SỬ ---
st.markdown("---")
with st.expander("🛠 Lịch sử & Cài đặt"):
    c1, c2 = st.columns(2)
    with c1:
        if st.button("↩️ Xóa kỳ mới nhất"):
            delete_last_row()
            st.rerun()
    with c2:
        if st.button("🗑 Xóa TẤT CẢ"):
            delete_all_data()
            st.rerun()
            
    # Hiển thị bảng lịch sử (Hiện rõ 20 số)
    if not df.empty:
        st.write("📋 **Lịch sử các kỳ đã nhập:**")
        
        # Chọn các cột cần hiển thị: ID, Super Num và 20 số
        display_cols = ['draw_id', 'super_num'] + [f'num_{i}' for i in range(1, 21)]
        
        # Hiển thị bảng dữ liệu
        st.dataframe(
            df[display_cols].head(20), # Hiện 20 kỳ gần nhất
            use_container_width=True, 
            hide_index=True
        )
    else:
        st.caption("Chưa có dữ liệu.")
