import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import re
from collections import Counter
from datetime import datetime
import plotly.express as px
from PIL import Image, ImageEnhance, ImageOps
import pytesseract
import cv2

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG
# ==============================================================================
st.set_page_config(
    page_title="Bingo Quantum AI - Platinum Fix V3", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# CSS Tùy chỉnh
st.markdown("""
<style>
    div.stButton > button:first-child {
        min-height: 65px; width: 100%; margin: 0px 1px;
        font-weight: bold; border-radius: 6px; font-size: 18px;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem; font-weight: bold;
    }
    [data-testid="column"] { padding: 0px 2px; }
    .anomaly-box-hot { background-color: #ffe6e6; padding: 10px; border-radius: 5px; border-left: 5px solid #ff4b4b; color: #c0392b;}
    .anomaly-box-cold { background-color: #e8f8f5; padding: 10px; border-radius: 5px; border-left: 5px solid #1abc9c; color: #16a085;}
    .kelly-box { background-color: #fff8e1; padding: 15px; border-radius: 8px; border: 2px solid #f1c40f; text-align: center; font-weight: bold; font-size: 18px; }
</style>
""", unsafe_allow_html=True)

DATA_FILE = 'bingo_history.csv'

# ==============================================================================
# 2. STATE & DATA
# ==============================================================================
if 'selected_nums' not in st.session_state: st.session_state['selected_nums'] = [] 
if 'predict_data' not in st.session_state: st.session_state['predict_data'] = None 
if 'z_score_data' not in st.session_state: st.session_state['z_score_data'] = None 
if 'selected_algo' not in st.session_state: st.session_state['selected_algo'] = "🔮 AI Master (Tổng Hợp)"
if 'paste_key_id' not in st.session_state: st.session_state['paste_key_id'] = 0
if 'ocr_result' not in st.session_state: st.session_state['ocr_result'] = [] 

def load_data():
    num_cols = [f'num_{i}' for i in range(1, 21)]
    columns = ['draw_id', 'time'] + num_cols + ['super_num']
    df = pd.DataFrame(columns=columns)
    if os.path.exists(DATA_FILE):
        try:
            loaded_df = pd.read_csv(DATA_FILE)
            if not loaded_df.empty: df = loaded_df
        except: pass
    if 'draw_id' in df.columns:
        df['draw_id'] = pd.to_numeric(df['draw_id'], errors='coerce').fillna(0).astype(int)
    if 'time' in df.columns: df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df[df['draw_id'] > 0].sort_values(by='draw_id', ascending=False).drop_duplicates(subset=['draw_id'], keep='first')
    return df

def save_data(df):
    df.sort_values(by='draw_id', ascending=False).to_csv(DATA_FILE, index=False)

def delete_last_row():
    df = load_data()
    if not df.empty: df = df.iloc[1:]; save_data(df); return True
    return False

def toggle_number(num):
    if num in st.session_state['selected_nums']: st.session_state['selected_nums'].remove(num)
    else:
        if len(st.session_state['selected_nums']) < 20: st.session_state['selected_nums'].append(num)
        else: st.toast("⚠️ Tối đa 20 số!", icon="🚫")

def clear_selection(): st.session_state['selected_nums'] = []
def clear_paste_box(): st.session_state['paste_key_id'] += 1

# ==============================================================================
# 3. OCR & PARSER ENGINE (NÂNG CẤP V3 - XỬ LÝ ẢNH TỐI)
# ==============================================================================
def preprocess_image(image):
    """
    Xử lý ảnh chuyên sâu cho Bingo Đài Loan:
    - Xử lý nền tối (Dark mode web)
    - Tách số khỏi bóng màu
    """
    # 1. Chuyển sang ảnh xám
    img_cv = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    
    # 2. Thresholding: Tự động phân ngưỡng để tách chữ
    # Dùng OTSU để tìm ngưỡng tốt nhất
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Kiểm tra xem ảnh là "Nền đen chữ trắng" hay ngược lại
    # Đếm số điểm ảnh trắng. Nếu > 50% là trắng -> Nền trắng. Ngược lại là nền đen.
    # Bingo web thường là nền tối, chữ trắng.
    white_pixels = np.sum(thresh == 255)
    total_pixels = thresh.shape[0] * thresh.shape[1]
    
    # Nếu ảnh là nền tối (ít điểm trắng), ta cần đảo ngược để thành Nền Trắng - Chữ Đen
    # Vì Tesseract đọc chữ đen trên nền trắng tốt nhất.
    if white_pixels < total_pixels * 0.5:
        thresh = cv2.bitwise_not(thresh)
        
    return thresh

def extract_text_from_image(image):
    try:
        processed_img = preprocess_image(image)
        # psm 6: Đọc theo khối văn bản thống nhất (quan trọng cho bảng số)
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789: ' 
        text = pytesseract.image_to_string(processed_img, config=custom_config)
        return text
    except Exception as e:
        return f"Error OCR: {e}"

def parse_multi_draws(text, selected_date):
    """
    Logic phân tích mới:
    - Giữ nguyên thứ tự đọc (Trái -> Phải).
    - Tách số Siêu Cấp dựa trên vị trí cuối cùng.
    """
    results = []
    
    # Clean text (Thay thế các ký tự dễ nhầm lẫn)
    text = text.replace('O', '0').replace('o', '0').replace('l', '1').replace('I', '1').replace('|', '1').replace('S', '5')
    
    # Tìm các Draw ID (114xxxxxx)
    matches = list(re.finditer(r'\b114\d{6}\b', text))
    if not matches: matches = list(re.finditer(r'114\d{6}', text))

    for i in range(len(matches)):
        try:
            did_str = matches[i].group()
            did = int(did_str)
            
            # Vùng text của kỳ này (từ ID này đến ID kia)
            s = matches[i].end()
            e = matches[i+1].start() if i + 1 < len(matches) else len(text)
            seg = text[s:e]
            
            # Lấy tất cả các con số tìm được trong vùng này, GIỮ NGUYÊN THỨ TỰ
            # Regex \d{1,2} bắt số từ 1 đến 99
            raw_nums_str = re.findall(r'\b\d{1,2}\b', seg)
            
            valid_nums_ordered = []
            for n_str in raw_nums_str:
                try:
                    val = int(n_str)
                    if 1 <= val <= 80:
                        valid_nums_ordered.append(val)
                except: continue
            
            # --- LOGIC TÁCH SỐ SIÊU CẤP ---
            # Với ảnh bảng kết quả, số siêu cấp luôn nằm cuối cùng bên phải
            
            main_20 = []
            super_n = 0
            
            # Nếu đọc được từ 20 số trở lên
            if len(valid_nums_ordered) >= 20:
                # 20 số đầu là dãy chính
                main_20 = valid_nums_ordered[:20]
                
                # Nếu có số thứ 21, đó chắc chắn là số siêu cấp
                if len(valid_nums_ordered) > 20:
                    super_n = valid_nums_ordered[20]
                else:
                    # Nếu chỉ đọc được đúng 20 số (có thể sót siêu cấp),
                    # Tạm thời để siêu cấp là 0 để user tự điền
                    super_n = 0
            
            # Chấp nhận kết quả nếu đọc được ít nhất 15 số (để user sửa)
            if len(main_20) >= 15:
                # Sắp xếp lại dãy số chính cho đúng chuẩn Bingo
                main_20 = sorted(list(set(main_20)))
                
                # Bù số 0 nếu thiếu (do OCR sót)
                while len(main_20) < 20:
                    main_20.append(0)
                    
                results.append({
                    'draw_id': did, 
                    'time': datetime.combine(selected_date, datetime.now().time()), 
                    'nums': main_20, 
                    'super_num': super_n
                })
        except: continue
        
    return results

# ==============================================================================
# 4. MODULE PHÂN TÍCH
# ==============================================================================
def calculate_z_scores(df):
    if df.empty: return None, [], []
    recent = df.head(30)
    all_nums = []
    for i in range(1, 21): all_nums.extend(recent[f'num_{i}'].tolist())
    counts = pd.Series(all_nums).value_counts().reindex(range(1, 81), fill_value=0)
    mean = counts.mean(); std = counts.std()
    z_scores = (counts - mean) / std
    return z_scores, z_scores[z_scores > 1.5].sort_values(ascending=False), z_scores[z_scores < -1.5].sort_values(ascending=True)

def kelly_criterion_suggestion(win_prob, odds, bankroll):
    b = odds - 1; p = win_prob; q = 1 - p
    f = (b * p - q) / b
    safe_f = max(0, f * 0.5)
    return safe_f * 100, bankroll * safe_f

def run_prediction(df, strategy):
    if df.empty: return []
    recent = df.head(10)
    all_nums = []
    for i in range(1, 21): all_nums.extend(recent[f'num_{i}'].tolist())
    freq = pd.Series(all_nums).value_counts()
    last = [df.iloc[0][f'num_{i}'] for i in range(1, 21)]
    scores = {}
    for n in range(1, 81):
        if strategy == "🔮 AI Master (Tổng Hợp)":
            s = freq.get(n, 0) * 1.5
            if n in last: s += 3.0
            if (n-1) in last or (n+1) in last: s += 1.0
            s += random.uniform(0, 1.0)
            scores[n] = s
        elif strategy == "🔥 Soi Cầu Nóng (Hot)": scores[n] = freq.get(n, 0) + (random.random() * 0.1)
        elif strategy == "❄️ Soi Cầu Lạnh (Nuôi)": scores[n] = (freq.max() if not freq.empty else 0 - freq.get(n, 0)) + random.uniform(0, 1.5)
        elif strategy == "♻️ Soi Cầu Bệt (Lại)": scores[n] = (1000 if n in last else 0) + freq.get(n, 0)*0.1
    return sorted(scores, key=scores.get, reverse=True)

# ==============================================================================
# 5. GIAO DIỆN CHÍNH
# ==============================================================================
st.title("🎲 BINGO QUANTUM - PLATINUM EDITION")
df_history = load_data()

# --- KHU VỰC NHẬP LIỆU ---
with st.container(border=True):
    t1, t2, t3 = st.tabs(["📸 QUÉT ẢNH (SCAN)", "🖱️ BÀN PHÍM SỐ", "📋 DÁN (COPY)"])
    
    # --- TAB 1: SCAN ẢNH (V3) ---
    with t1:
        st.caption("Upload ảnh chụp bảng kết quả. Hệ thống sẽ tự động tách số siêu cấp.")
        col_up1, col_up2 = st.columns([2, 1])
        with col_up1:
            uploaded_file = st.file_uploader("Chọn ảnh:", type=['png', 'jpg', 'jpeg'])
            scan_date = st.date_input("Ngày trên ảnh:", datetime.now(), key="scan_date")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Ảnh gốc', use_container_width=True)
            
            if st.button("🔍 BẮT ĐẦU QUÉT SỐ", type="primary"):
                with st.spinner("AI đang xử lý ảnh nền tối & đọc số..."):
                    raw_text = extract_text_from_image(image)
                    extracted_data = parse_multi_draws(raw_text, scan_date)
                    
                    if extracted_data:
                        st.session_state['ocr_result'] = extracted_data
                        st.success(f"Đã tìm thấy {len(extracted_data)} kỳ quay!")
                    else:
                        st.error("Không tìm thấy dữ liệu. Hãy thử chụp gần hơn hoặc cắt bớt phần thừa.")

        if st.session_state['ocr_result']:
            st.markdown("---")
            st.write("### 📝 Kết quả đọc được (Hãy kiểm tra lại):")
            
            for i, item in enumerate(st.session_state['ocr_result']):
                with st.expander(f"Kỳ {item['draw_id']} - Siêu cấp: {item['super_num']}", expanded=True):
                    c_edit1, c_edit2 = st.columns([3, 1])
                    with c_edit1:
                        # Hiển thị input để sửa dãy số
                        nums_str = ", ".join([str(n) for n in item['nums']])
                        new_nums_str = st.text_area(f"Dãy số chính (Kỳ {item['draw_id']}):", value=nums_str, key=f"edit_ocr_nums_{i}", height=68)
                        try:
                            # Cập nhật lại số khi user sửa
                            new_nums = sorted([int(n.strip()) for n in new_nums_str.split(',') if n.strip().isdigit()])
                            st.session_state['ocr_result'][i]['nums'] = new_nums
                        except: pass
                    with c_edit2:
                        # Input sửa số siêu cấp
                        new_super = st.number_input(f"Số Siêu Cấp:", value=int(item['super_num']), min_value=0, max_value=80, key=f"edit_ocr_super_{i}")
                        st.session_state['ocr_result'][i]['super_num'] = new_super

            if st.button("💾 LƯU TẤT CẢ VÀO LỊCH SỬ", type="primary", key="save_ocr"):
                added = 0
                for item in st.session_state['ocr_result']:
                    if df_history.empty or item['draw_id'] not in df_history['draw_id'].values:
                        r = {'draw_id': item['draw_id'], 'time': item['time'], 'super_num': item['super_num']}
                        for i, v in enumerate(item['nums']): 
                            if i < 20: r[f'num_{i+1}'] = v
                        for k in range(len(item['nums']) + 1, 21): r[f'num_{k}'] = 0
                        df_history = pd.concat([pd.DataFrame([r]), df_history], ignore_index=True)
                        added += 1
                
                if added > 0:
                    save_data(df_history)
                    st.success(f"Đã lưu thành công {added} kỳ mới!")
                    st.session_state['ocr_result'] = []
                    st.rerun()
                else:
                    st.warning("Các kỳ này đã có trong lịch sử!")

    # --- TAB 2 & 3 GIỮ NGUYÊN ---
    with t2:
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1: nid = str(int(df_history['draw_id'].max()) + 1) if not df_history.empty else ""; mid = st.text_input("Mã Kỳ:", value=nid, key="mid")
        with c2: mdate = st.date_input("Ngày:", datetime.now(), key="mdate")
        with c3: st.write(""); st.write(""); st.button("Xóa chọn", key="b_clr", on_click=clear_selection)
        st.markdown(f"**🔢 Đã chọn: <span style='color:red'>{len(st.session_state['selected_nums'])}/20</span>**", unsafe_allow_html=True)
        for r in range(8):
            cols = st.columns(10)
            for c in range(10):
                n = r*10 + c + 1
                with cols[c]:
                    sel = n in st.session_state['selected_nums']
                    if st.button(f"{n:02d}", key=f"g_{n}", type="primary" if sel else "secondary"): toggle_number(n); st.rerun()
        st.markdown("---")
        v_supers = sorted(st.session_state['selected_nums']) if st.session_state['selected_nums'] else range(1, 81)
        msuper = st.selectbox("🔥 Siêu Cấp:", v_supers, key="msup")
        if st.button("💾 LƯU THỦ CÔNG", type="primary"):
            if not mid or len(st.session_state['selected_nums']) != 20: st.error("Lỗi nhập liệu!")
            elif not df_history.empty and int(mid) in df_history['draw_id'].values: st.warning("Đã tồn tại!")
            else:
                row = {'draw_id': int(mid), 'time': datetime.combine(mdate, datetime.now().time()), 'super_num': msuper}
                for i, v in enumerate(sorted(st.session_state['selected_nums'])): row[f'num_{i+1}'] = v
                save_data(pd.concat([pd.DataFrame([row]), df_history], ignore_index=True)); st.success("Đã lưu!"); clear_selection(); st.rerun()

    with t3:
        c1, c2 = st.columns([3, 1])
        with c1: pdate = st.date_input("Ngày:", datetime.now(), key="pdate")
        with c2: st.button("🗑 Xóa ô dán", on_click=clear_paste_box, use_container_width=True)
        ptext = st.text_area("Dán dữ liệu:", height=150, key=f"parea_{st.session_state['paste_key_id']}")
        if st.button("💾 XỬ LÝ & LƯU", type="primary"):
            results = []
            matches = list(re.finditer(r'\b114\d{6}\b', ptext))
            for i in range(len(matches)):
                try:
                    did = int(matches[i].group()); s = matches[i].end(); e = matches[i+1].start() if i + 1 < len(matches) else len(ptext)
                    nums = sorted(list(set([int(n) for n in re.findall(r'\d{2}', ptext[s:e]) if 1 <= int(n) <= 80]))[:20])
                    if len(nums) >= 15: results.append({'draw_id': did, 'time': datetime.combine(pdate, datetime.now().time()), 'nums': nums, 'super_num': nums[-1]})
                except: continue
            if results:
                added = 0
                for it in results:
                    if df_history.empty or it['draw_id'] not in df_history['draw_id'].values:
                        r = {'draw_id': it['draw_id'], 'time': it['time'], 'super_num': it['super_num']}
                        for i, v in enumerate(it['nums']): r[f'num_{i+1}'] = v
                        df_history = pd.concat([pd.DataFrame([r]), df_history], ignore_index=True); added += 1
                if added: save_data(df_history); st.success(f"Thêm {added} kỳ!"); st.rerun()
                else: st.warning("Dữ liệu cũ!")
            else: st.error("Lỗi dữ liệu!")

# --- KHU VỰC PHÂN TÍCH ---
st.write(""); st.markdown("### 📊 PHÂN TÍCH ĐỊNH LƯỢNG (QUANTITATIVE)")
if st.button("🚀 CHẠY PHÂN TÍCH TOÀN DIỆN", type="primary"):
    if not df_history.empty:
        st.session_state['predict_data'] = run_prediction(df_history, st.session_state['selected_algo'])
        st.session_state['z_score_data'] = calculate_z_scores(df_history)
        st.toast("Phân tích hoàn tất!", icon="✅")
    else: st.error("Chưa có dữ liệu.")

if st.session_state['predict_data'] or not df_history.empty:
    st.markdown("---")
    rt1, rt2 = st.tabs(["📉 PHÂN TÍCH Z-SCORE", "🎯 DỰ ĐOÁN & KELLY"])
    with rt1:
        st.subheader("🔍 Tìm Kiếm Sự Dị Biệt")
        if st.session_state['z_score_data']:
            z_all, hots, colds = st.session_state['z_score_data']
            c_hot, c_cold = st.columns(2)
            with c_hot:
                st.markdown("#### 🔥 SỐ 'NÓNG' (Z > 1.5)")
                if not hots.empty:
                    for n, score in hots.items(): st.markdown(f"<div class='anomaly-box-hot'>🔴 Số <b>{n:02d}</b> (Z: {score:.2f})</div>", unsafe_allow_html=True)
                else: st.info("Không có.")
            with c_cold:
                st.markdown("#### ❄️ SỐ 'LẠNH' (Z < -1.5)")
                if not colds.empty:
                    for n, score in colds.items(): st.markdown(f"<div class='anomaly-box-cold'>🔵 Số <b>{n:02d}</b> (Z: {score:.2f})</div>", unsafe_allow_html=True)
                else: st.info("Không có.")
            st.plotly_chart(px.bar(x=z_all.index, y=z_all.values, labels={'x': 'Số', 'y': 'Z-Score'}, color=z_all.values, color_continuous_scale='RdBu_r'), use_container_width=True)
        else: st.info("Chưa chạy phân tích.")

    with rt2:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("KẾT QUẢ DỰ ĐOÁN")
            salgo = st.selectbox("Thuật toán:", ["🔮 AI Master (Tổng Hợp)", "🔥 Soi Cầu Nóng (Hot)", "❄️ Soi Cầu Lạnh (Nuôi)", "♻️ Soi Cầu Bệt (Lại)"])
            if salgo != st.session_state['selected_algo']: st.session_state['selected_algo'] = salgo; st.session_state['predict_data'] = run_prediction(df_history, salgo); st.rerun()
            smode = st.selectbox("Dàn:", {"10 Tinh": 10, "6 Tinh": 6, "1 Tinh": 1}.keys(), index=1)
            if st.session_state['predict_data']:
                fnums = sorted(st.session_state['predict_data'][:{"10 Tinh": 10, "6 Tinh": 6, "1 Tinh": 1}[smode]])
                cols = st.columns(5)
                for i, n in enumerate(fnums): cols[i%5].markdown(f"<div style='background-color:{'#E74C3C' if n>40 else '#3498DB'}; color:white; padding:10px; border-radius:5px; text-align:center; font-weight:bold; font-size:20px; margin-bottom:5px'>{n:02d}</div>", unsafe_allow_html=True)
        with c2:
            st.subheader("💰 QUẢN LÝ VỐN (KELLY)")
            my_money = st.number_input("Vốn (Đài tệ):", value=10000, step=1000)
            win_rate = 0.35 if smode == "6 Tinh" else 0.55; odds = 4.0 if smode == "6 Tinh" else 2.0
            k_pct, k_mon = kelly_criterion_suggestion(win_rate, odds, my_money)
            if k_pct > 0: st.markdown(f"<div class='kelly-box'>💡 GỢI Ý:<br><span style='color:#e67e22'>{k_pct:.1f}% Vốn</span><br><span style='color:#27ae60'>${k_mon:,.0f} TWD</span></div>", unsafe_allow_html=True)
            else: st.warning("Bảo toàn vốn.")

st.markdown("---")
with st.expander("LỊCH SỬ"):
    if st.button("Xóa kỳ cuối"): delete_last_row(); st.rerun()
    if not df_history.empty: st.dataframe(df_history, use_container_width=True, hide_index=True)
