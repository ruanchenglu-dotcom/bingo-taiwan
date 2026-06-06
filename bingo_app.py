import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import re
from collections import Counter
from datetime import datetime
import plotly.express as px

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG
# ==============================================================================
st.set_page_config(
    page_title="Bingo Quantum AI - Z-Score Edition", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# CSS Tùy chỉnh (Giao diện Chuyên gia)
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
    .anomaly-box-hot { background-color: #ffe6e6; padding: 10px; border-radius: 5px; border-left: 5px solid #ff4b4b; margin-bottom: 5px; color: #c0392b;}
    .anomaly-box-cold { background-color: #e8f8f5; padding: 10px; border-radius: 5px; border-left: 5px solid #1abc9c; margin-bottom: 5px; color: #16a085;}
    .kelly-box { background-color: #fff8e1; padding: 15px; border-radius: 8px; border: 2px solid #f1c40f; text-align: center; font-weight: bold; font-size: 18px; }
</style>
""", unsafe_allow_html=True)

DATA_FILE = 'bingo_history.csv'

# ==============================================================================
# 2. STATE & DATA
# ==============================================================================
if 'selected_nums' not in st.session_state: st.session_state['selected_nums'] = [] 
if 'predict_data' not in st.session_state: st.session_state['predict_data'] = None 
if 'z_score_data' not in st.session_state: st.session_state['z_score_data'] = None # New State
if 'selected_algo' not in st.session_state: st.session_state['selected_algo'] = "🔮 Ensemble AI (Đa Mô Hình)"
if 'paste_key_id' not in st.session_state: st.session_state['paste_key_id'] = 0

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

def delete_all_data():
    if os.path.exists(DATA_FILE): os.remove(DATA_FILE); return True
    return False

# ==============================================================================
# 3. LOGIC & PARSER
# ==============================================================================
def toggle_number(num):
    if num in st.session_state['selected_nums']: st.session_state['selected_nums'].remove(num)
    else:
        if len(st.session_state['selected_nums']) < 20: st.session_state['selected_nums'].append(num)
        else: st.toast("⚠️ Max 20 số!", icon="🚫")

def clear_selection(): st.session_state['selected_nums'] = []
def clear_paste_box(): st.session_state['paste_key_id'] += 1

def parse_multi_draws(text, selected_date):
    results = []
    matches = list(re.finditer(r'\b114\d{6}\b', text))
    for i in range(len(matches)):
        try:
            did = int(matches[i].group())
            s = matches[i].end()
            e = matches[i+1].start() if i + 1 < len(matches) else len(text)
            seg = text[s:e]
            nums = sorted(list(set([int(n) for n in re.findall(r'\b\d{1,2}\b', seg) if 1 <= int(n) <= 80]))[:20])
            if len(nums) >= 15:
                results.append({'draw_id': did, 'time': datetime.combine(selected_date, datetime.now().time()), 'nums': nums, 'super_num': nums[-1]})
        except: continue
    return results

def process_ocr_image(image_bytes):
    try:
        import sys
        import pytesseract
        from PIL import Image
        import io
        import os
        if sys.platform == "win32":
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        img = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(img)
        return text
    except ImportError:
        return "ERROR_NO_LIB"
    except Exception as e:
        return f"ERROR: {str(e)}"

def fetch_auzo_bingo():
    try:
        import requests
        import re
        from bs4 import BeautifulSoup
        from datetime import timedelta
        
        now = datetime.now()
        results = []
        
        for d in [now, now - timedelta(days=1)]:
            date_str = d.strftime('%Y%m%d')
            url = f'https://lotto.auzo.tw/bingobingo/list_{date_str}.html'
            r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            soup = BeautifulSoup(r.text, 'html.parser')
            rows = soup.find_all('tr', class_='bingo_row')
            
            for row in rows:
                period_td = row.find('td', class_='BPeriod')
                if not period_td: continue
                b_tag = period_td.find('b')
                if not b_tag: continue
                draw_id_str = b_tag.text.strip()
                if not re.match(r'11[45]\d{6}', draw_id_str): continue
                draw_id = int(draw_id_str)
                
                tds = row.find_all('td')
                if len(tds) < 2: continue
                divs = tds[1].find_all('div')
                nums = []
                super_num = None
                for div in divs:
                    n = div.text.strip()
                    if n.isdigit():
                        val = int(n)
                        nums.append(val)
                        cls = div.get('class', [''])[0]
                        if 's' in cls:
                            super_num = val
                if len(nums) >= 15:
                    if super_num is None: super_num = nums[-1]
                    results.append({'draw_id': draw_id, 'time': datetime.now(), 'nums': nums, 'super_num': super_num})
            
            if len(results) > 0:
                break
                
        if len(results) > 0:
            results.sort(key=lambda x: x['draw_id'])
            return results, None
        else:
            return None, "Không tìm thấy mã kỳ mới nhất hôm nay"
    except Exception as e:
        return None, str(e)

# ==============================================================================
# 4. QUANTUM ANALYSIS (Z-SCORE & KELLY)
# ==============================================================================
def calculate_z_scores(df):
    """Tính toán Z-Score cho 80 số dựa trên 30 kỳ gần nhất"""
    if df.empty: return None, [], []
    
    recent = df.head(30)
    all_nums = []
    for i in range(1, 21): all_nums.extend(recent[f'num_{i}'].tolist())
    
    counts = pd.Series(all_nums).value_counts().reindex(range(1, 81), fill_value=0)
    
    # Thống kê cơ bản
    mean = counts.mean() # Trung bình
    std = counts.std()   # Độ lệch chuẩn
    
    # Tính Z-Score: (Giá trị - Trung bình) / Độ lệch chuẩn
    z_scores = (counts - mean) / std
    
    # Lọc Dị Biệt (Anomalies)
    # Z > 1.5: Nóng bất thường
    # Z < -1.5: Lạnh bất thường
    hot_anomalies = z_scores[z_scores > 1.5].sort_values(ascending=False)
    cold_anomalies = z_scores[z_scores < -1.5].sort_values(ascending=True)
    
    return z_scores, hot_anomalies, cold_anomalies

def kelly_criterion_suggestion(win_prob=0.25, odds=3.0, bankroll=10000):
    """Gợi ý đi tiền theo Kelly"""
    # f = (bp - q) / b
    # b = odds - 1 (Tỷ lệ cược ròng)
    # p = win_prob (Xác suất thắng)
    # q = 1 - p (Xác suất thua)
    b = odds - 1
    p = win_prob
    q = 1 - p
    f = (b * p - q) / b
    
    # Kelly an toàn (Half Kelly) để giảm rủi ro
    safe_f = f * 0.5 
    if safe_f < 0: safe_f = 0
    
    bet_amount = bankroll * safe_f
    return safe_f * 100, bet_amount

def backtest_accuracy(df, strategy, mode_count):
    """Kiểm tra tỷ lệ thắng của thuật toán trên 10 kỳ gần nhất"""
    if len(df) < 11: return 0.25 # Mặc định nếu không đủ data
    wins = 0
    test_draws = min(10, len(df) - 1)
    for i in range(test_draws):
        historical_df = df.iloc[i+1:]
        predictions = run_prediction(historical_df, strategy)
        predicted_nums = predictions[:mode_count]
        actual_nums = [df.iloc[i][f'num_{j}'] for j in range(1, 21)]
        hits = len(set(predicted_nums).intersection(set(actual_nums)))
        wins += hits / mode_count
    return wins / test_draws

def calculate_optimal_frame(df, fnums):
    if len(df) < 20: return "Cần thêm dữ liệu lịch sử (tối thiểu 20 kỳ) để tính toán nhịp gan."
    avg_gaps, current_gaps = [], []
    for n in fnums:
        apps = []
        for i in range(len(df)):
            row = df.iloc[i]
            if n in [row[f'num_{j}'] for j in range(1, 21)]: apps.append(i)
        if len(apps) >= 2:
            gaps = [apps[k] - apps[k+1] for k in range(len(apps)-1)]
            avg_gaps.append(np.mean(gaps))
        else:
            avg_gaps.append(4)
        current_gaps.append(apps[0] if apps else len(df))

    avg_cycle = np.mean(avg_gaps)
    avg_curr = np.mean(current_gaps)
    if avg_curr >= avg_cycle * 1.5:
        frame, status = 3, "Rất chín (Đã vượt chu kỳ max)"
    elif avg_curr >= avg_cycle:
        frame, status = 2, "Đang vào điểm rơi"
    else:
        frame = int(max(1, round(avg_cycle - avg_curr)))
        if frame > 5: frame = 5
        status = "Đang trong chu kỳ đợi"
    return f"📊 **Chiến thuật Nuôi Khung:** Trạng thái **{status}**. Khuyên bạn nên **vào tiền nuôi gấp thếp trong {frame} kỳ tiếp theo** để tối ưu xác suất nổ."

def train_and_predict_ml(df):
    if len(df) < 30: return []
    try:
        from sklearn.tree import DecisionTreeClassifier
        X, y = [], []
        for i in range(len(df) - 6):
            features = []
            for j in range(1, 6):
                nums = [df.iloc[i+j][f'num_{k}'] for k in range(1, 21)]
                for n in range(1, 81): features.append(1 if n in nums else 0)
            X.append(features)
            
            current_nums = [df.iloc[i][f'num_{k}'] for k in range(1, 21)]
            y_row = [1 if n in current_nums else 0 for n in range(1, 81)]
            y.append(y_row)
                
        next_features = []
        for j in range(0, 5):
            nums = [df.iloc[j][f'num_{k}'] for k in range(1, 21)]
            for n in range(1, 81): next_features.append(1 if n in nums else 0)
            
        rf = DecisionTreeClassifier(max_depth=5, random_state=42)
        rf.fit(X, y)
        probs = rf.predict_proba([next_features])
        
        scores = {}
        for n in range(1, 81):
            prob_n = probs[n-1][0]
            # sklearn predict_proba for multioutput returns a list of arrays (one for each output)
            if len(prob_n) > 1:
                scores[n] = prob_n[1]
            else:
                scores[n] = 0
                
        return sorted(scores, key=scores.get, reverse=True)
    except:
        return []

def run_ensemble_prediction(df):
    if len(df) < 30: return run_prediction(df, "🔮 AI Master")
    ai_scores = {n: 0 for n in range(1, 81)}
    for i, n in enumerate(run_prediction(df, "🔮 AI Master")): ai_scores[n] = 80 - i
    ml_scores = {n: 0 for n in range(1, 81)}
    ml_preds = train_and_predict_ml(df)
    if ml_preds:
        for i, n in enumerate(ml_preds): ml_scores[n] = 80 - i
    markov_scores = {n: 0 for n in range(1, 81)}
    last_nums = [df.iloc[0][f'num_{k}'] for k in range(1, 21)]
    transitions = {n: {m: 0 for m in range(1, 81)} for n in range(1, 81)}
    for i in range(min(100, len(df)-1)):
        curr = [df.iloc[i][f'num_{k}'] for k in range(1, 21)]
        prev = [df.iloc[i+1][f'num_{k}'] for k in range(1, 21)]
        for p in prev:
            for c in curr: transitions[p][c] += 1
    for n in range(1, 81): markov_scores[n] = sum([transitions[p][n] for p in last_nums])
    max_m = max(markov_scores.values()) if max(markov_scores.values()) > 0 else 1
    final_scores = {}
    for n in range(1, 81):
        final_scores[n] = (ai_scores[n]/80)*0.4 + (ml_scores[n]/80)*0.4 + (markov_scores[n]/max_m)*0.2
    return sorted(final_scores, key=final_scores.get, reverse=True)

def run_prediction(df, strategy):
    if df.empty: return []
    if strategy == "🤖 AI / Machine Learning":
        ml_pred = train_and_predict_ml(df)
        if ml_pred: return ml_pred
        strategy = "🔮 AI Master"
    elif strategy == "🔮 Ensemble AI (Đa Mô Hình)":
        return run_ensemble_prediction(df)
        
    recent = df.head(10)
    all_nums = []
    for i in range(1, 21): all_nums.extend(recent[f'num_{i}'].tolist())
    freq = pd.Series(all_nums).value_counts()
    last = [df.iloc[0][f'num_{i}'] for i in range(1, 21)]
    scores = {}
    for n in range(1, 81):
        tie_breaker = n * 0.0001
        if strategy == "🔮 AI Master":
            s = freq.get(n, 0) * 1.5
            if n in last: s += 3.0
            if (n-1) in last or (n+1) in last: s += 1.0
            scores[n] = s + tie_breaker
        elif strategy == "🔥 Soi Cầu Nóng": scores[n] = freq.get(n, 0) + tie_breaker
        elif strategy == "❄️ Soi Cầu Lạnh": scores[n] = (freq.max() if not freq.empty else 0 - freq.get(n, 0)) + tie_breaker
        elif strategy == "♻️ Soi Cầu Bệt": scores[n] = (1000 if n in last else 0) + freq.get(n, 0)*0.1 + tie_breaker
        else: scores[n] = tie_breaker
    return sorted(scores, key=scores.get, reverse=True)

# ==============================================================================
# 5. UI CHÍNH
# ==============================================================================
def load_line_token():
    import os
    if os.path.exists('line_token.txt'):
        with open('line_token.txt', 'r') as f: return f.read().strip()
    return ''

def save_line_token(t):
    with open('line_token.txt', 'w') as f: f.write(t)

def send_line_notification(token, message):
    if not token: return
    try:
        import requests
        url = 'https://notify-api.line.me/api/notify'
        headers = {'Authorization': f'Bearer {token}'}
        data = {'message': message}
        requests.post(url, headers=headers, data=data, timeout=5)
    except: pass

def process_new_draw_and_notify(df, added_count, latest_draw_id):
    if added_count > 0:
        save_data(df)
        token = load_line_token()
        if token:
            try:
                latest = df.iloc[0]
                nums = [str(latest[f'num_{i}']) for i in range(1, 21)]
                pred_scores = train_and_predict_ml(df)
                if not pred_scores: pred_scores = run_prediction(df, "🔮 AI Master")
                pred = pred_scores[:10]
                msg = f"\n🎲 KỲ {latest_draw_id} VỪA XỔ!\n👉 Số ra: {', '.join(nums)}\n🤖 AI Dự đoán kỳ tới: {', '.join([str(x) for x in pred])}"
                send_line_notification(token, msg)
            except Exception as e:
                print(f"LINE error: {e}")

st.title("🎲 BINGO QUANTUM - Z-SCORE EDITION")
df_history = load_data()

with st.sidebar:
    st.header("🔔 CẤU HÌNH THÔNG BÁO LINE")
    st.write("Nhận thông báo tự động về điện thoại.")
    curr_token = load_line_token()
    line_token = st.text_input("LINE Notify Token", type="password", value=curr_token)
    if st.button("Lưu Token"):
        save_line_token(line_token)
        st.success("Đã lưu cấu hình LINE!")

with st.container(border=True):
    t1, t2, t3, t4 = st.tabs(["🖱️ BÀN PHÍM SỐ", "📋 DÁN (COPY)", "📸 TẢI ẢNH (OCR)", "🔄 CÀO DỮ LIỆU"])
    with t1:
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1: nid = str(int(df_history['draw_id'].max()) + 1) if not df_history.empty else ""; mid = st.text_input("Mã Kỳ:", value=nid, key="mid")
        with c2: mdate = st.date_input("Ngày:", datetime.now(), key="mdate")
        with c3: st.write(""); st.write(""); st.button("Xóa chọn", key="b_clr", on_click=clear_selection)
        
        st.markdown(f"**🔢 Chọn: <span style='color:red'>{len(st.session_state['selected_nums'])}/20</span>**", unsafe_allow_html=True)
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
        
        if st.button("💾 LƯU THỦ CÔNG", type="primary", use_container_width=True):
            if not mid or len(st.session_state['selected_nums']) != 20: st.error("Lỗi nhập liệu!")
            elif not df_history.empty and int(mid) in df_history['draw_id'].values: st.warning("Đã tồn tại!")
            else:
                row = {'draw_id': int(mid), 'time': datetime.combine(mdate, datetime.now().time()), 'super_num': msuper}
                for i, v in enumerate(sorted(st.session_state['selected_nums'])): row[f'num_{i+1}'] = v
                df_history = pd.concat([pd.DataFrame([row]), df_history], ignore_index=True)
                process_new_draw_and_notify(df_history, 1, int(mid))
                st.success("Đã lưu!"); clear_selection(); st.rerun()

    with t2:
        c1, c2 = st.columns([3, 1])
        with c1: pdate = st.date_input("Ngày:", datetime.now(), key="pdate")
        with c2: st.button("🗑 Xóa ô dán", on_click=clear_paste_box, use_container_width=True)
        ptext = st.text_area("Dán dữ liệu:", height=150, key=f"parea_{st.session_state['paste_key_id']}")
        if st.button("💾 XỬ LÝ & LƯU", type="primary", use_container_width=True):
            ext = parse_multi_draws(ptext, pdate)
            if ext:
                added = 0
                for it in ext:
                    if df_history.empty or it['draw_id'] not in df_history['draw_id'].values:
                        r = {'draw_id': it['draw_id'], 'time': it['time'], 'super_num': it['super_num']}
                        for i, v in enumerate(it['nums']): r[f'num_{i+1}'] = v
                        df_history = pd.concat([pd.DataFrame([r]), df_history], ignore_index=True)
                        added += 1
                if added: 
                    process_new_draw_and_notify(df_history, added, ext[0]['draw_id'])
                    st.success(f"Thêm {added} kỳ!"); st.rerun()
                else: st.warning("Dữ liệu cũ!")
            else: st.error("Lỗi dữ liệu!")

    with t3:
        st.markdown("### 📸 Tự động đọc ảnh chụp màn hình (OCR)")
        st.write("Tải ảnh chụp kết quả lên, AI sẽ tự động đọc ra mã kỳ và các con số mà không cần nhập tay.")
        uploaded_file = st.file_uploader("Chọn ảnh (JPG/PNG)", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            st.image(uploaded_file, width=300)
            if st.button("🔍 XỬ LÝ ẢNH & LƯU", type="primary", use_container_width=True):
                with st.spinner("Đang nhận dạng ảnh..."):
                    text = process_ocr_image(uploaded_file.getvalue())
                    if text == "ERROR_NO_LIB":
                        st.error("⚠️ Bạn chưa cài đặt thư viện OCR. Hãy mở Terminal chạy: `pip install pytesseract pillow` và cài đặt phần mềm Tesseract-OCR trên Windows.")
                    elif text.startswith("ERROR:"):
                        st.error(f"Lỗi đọc ảnh: {text}")
                    else:
                        st.info(f"Văn bản nhận dạng được: {text[:200]}...")
                        ext = parse_multi_draws(text, datetime.now().date())
                        if ext:
                            added = 0
                            for it in ext:
                                if df_history.empty or it['draw_id'] not in df_history['draw_id'].values:
                                    r = {'draw_id': it['draw_id'], 'time': it['time'], 'super_num': it['super_num']}
                                    for i, v in enumerate(it['nums']): r[f'num_{i+1}'] = v
                                    df_history = pd.concat([pd.DataFrame([r]), df_history], ignore_index=True)
                                    added += 1
                            if added:
                                process_new_draw_and_notify(df_history, added, ext[0]['draw_id'])
                                st.success(f"Nhận dạng thành công & Thêm {added} kỳ!"); st.rerun()
                            else: st.warning("Dữ liệu trong ảnh đã tồn tại!")
                        else: 
                            st.error("Không tìm thấy mã kỳ hợp lệ trong ảnh (Cần có mã kỳ bắt đầu bằng 114...)")

    with t4:
        st.markdown("### 🔄 Cào dữ liệu tự động (Nguồn: lotto.auzo.tw)")
        st.write("Tự động lấy kết quả kỳ mới nhất từ web thống kê bên thứ ba.")
        if st.button("🌐 LẤY KẾT QUẢ MỚI NHẤT", type="primary", use_container_width=True):
            with st.spinner("Đang tải dữ liệu..."):
                res_list, err = fetch_auzo_bingo()
                if err:
                    st.error(f"Lỗi: {err}")
                elif res_list:
                    added = 0
                    last_id = None
                    for res in res_list:
                        if df_history.empty or res['draw_id'] not in df_history['draw_id'].values:
                            r = {'draw_id': res['draw_id'], 'time': res['time'], 'super_num': res['super_num']}
                            for i, v in enumerate(res['nums']): r[f'num_{i+1}'] = v
                            df_history = pd.concat([pd.DataFrame([r]), df_history], ignore_index=True)
                            added += 1
                            last_id = res['draw_id']
                    if added > 0:
                        process_new_draw_and_notify(df_history, added, last_id)
                        st.success(f"Đã cập nhật thành công {added} kỳ mới nhất!")
                        st.rerun()
                    else:
                        st.warning("Tất cả các kỳ hôm nay đều đã có trong hệ thống!")

st.write(""); st.markdown("### 📊 PHÂN TÍCH ĐỊNH LƯỢNG (QUANTITATIVE)")

if st.button("🚀 CHẠY PHÂN TÍCH TOÀN DIỆN", type="primary", use_container_width=True):
    if not df_history.empty:
        st.session_state['predict_data'] = run_prediction(df_history, st.session_state['selected_algo'])
        st.session_state['z_score_data'] = calculate_z_scores(df_history)
        st.toast("Phân tích hoàn tất!", icon="✅")
    else: st.error("Chưa có dữ liệu.")

if st.session_state['predict_data'] or not df_history.empty:
    st.markdown("---")
    rt1, rt2, rt3, rt4 = st.tabs(["📉 PHÂN TÍCH Z-SCORE", "🎯 DỰ ĐOÁN & QUẢN LÝ VỐN", "⚖️ TÀI XỈU / CHẴN LẺ", "📜 LỊCH SỬ"])
    
    # --- TAB Z-SCORE ---
    with rt1:
        st.subheader("🔍 Tìm Kiếm Sự Dị Biệt (Statistical Anomalies)")
        st.caption("Dựa trên 30 kỳ gần nhất. Nếu Z-Score > 1.5 là RẤT NÓNG. Z-Score < -1.5 là RẤT LẠNH (Sắp nổ).")
        
        if st.session_state['z_score_data']:
            z_all, hots, colds = st.session_state['z_score_data']
            
            c_hot, c_cold = st.columns(2)
            with c_hot:
                st.markdown("#### 🔥 CÁC SỐ 'NÓNG' BẤT THƯỜNG (Z > 1.5)")
                st.write("👉 *Chiến thuật: Bám theo dây đỏ (Đánh tiếp)*")
                if not hots.empty:
                    for n, score in hots.items():
                        st.markdown(f"<div class='anomaly-box-hot'>🔴 Số <b>{n:02d}</b> (Z-Score: {score:.2f}) - Siêu Hot</div>", unsafe_allow_html=True)
                else: st.info("Không có số nào nóng bất thường.")
                
            with c_cold:
                st.markdown("#### ❄️ CÁC SỐ 'LẠNH' BẤT THƯỜNG (Z < -1.5)")
                st.write("👉 *Chiến thuật: Nuôi gấp thếp (Sắp nổ)*")
                if not colds.empty:
                    for n, score in colds.items():
                        st.markdown(f"<div class='anomaly-box-cold'>🔵 Số <b>{n:02d}</b> (Z-Score: {score:.2f}) - Siêu Lạnh</div>", unsafe_allow_html=True)
                else: st.info("Không có số nào lạnh bất thường.")
                
            # Biểu đồ Z-Score
            st.markdown("---")
            st.markdown("##### 📈 Biểu đồ phân phối Z-Score toàn bộ 80 số")
            fig = px.bar(x=z_all.index, y=z_all.values, labels={'x': 'Số (1-80)', 'y': 'Z-Score (Độ lệch chuẩn)'}, color=z_all.values, color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Chưa chạy phân tích.")

    # --- TAB DỰ ĐOÁN & KELLY ---
    with rt2:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("KẾT QUẢ DỰ ĐOÁN")
            algos = ["🔮 Ensemble AI (Đa Mô Hình)", "🔮 AI Master", "🤖 AI / Machine Learning", "🔥 Soi Cầu Nóng", "❄️ Soi Cầu Lạnh", "♻️ Soi Cầu Bệt"]
            salgo = st.selectbox("Thuật toán:", algos, index=0)
            if salgo != st.session_state['selected_algo']:
                st.session_state['selected_algo'] = salgo
                if not df_history.empty: st.session_state['predict_data'] = run_prediction(df_history, salgo); st.rerun()
            
            modes = {f"{i} Tinh": i for i in range(1, 11)}
            smode = st.selectbox("Dàn:", list(modes.keys()), index=5)
            
            if st.session_state['predict_data']:
                fnums = sorted(st.session_state['predict_data'][:modes[smode]])
                cols = st.columns(5)
                for i, n in enumerate(fnums): 
                    cols[i%5].markdown(f"<div style='background-color:{'#E74C3C' if n>40 else '#3498DB'}; color:white; padding:10px; border-radius:5px; text-align:center; font-weight:bold; font-size:20px; margin-bottom:5px'>{n:02d}</div>", unsafe_allow_html=True)
                
                st.write("")
                st.info(calculate_optimal_frame(df_history, fnums))
            else:
                st.info("👆 Bấm nút '🚀 CHẠY PHÂN TÍCH TOÀN DIỆN' ở trên cùng để xem kết quả dự đoán.")

        # GỢI Ý ĐI TIỀN KELLY
        with c2:
            st.subheader("💰 QUẢN LÝ VỐN (KELLY)")
            st.caption("Công thức Kelly tính mức cược an toàn tối đa.")
            
            my_money = st.number_input("Vốn hiện có:", value=10000, step=1000)
            odds_input = st.number_input("Tỷ lệ Cược (Tổng thu/Vốn, ví dụ 1 ăn 2 điền 2.0):", value=2.0, step=0.1)
            
            st.markdown(f"**Thống kê hiệu suất ({salgo}):**")
            calculated_win_prob = backtest_accuracy(df_history, salgo, modes[smode]) if not df_history.empty else 0.25
            st.write(f"Tỷ lệ trúng thực tế (10 kỳ qua): **{calculated_win_prob:.1%}**")
            
            kelly_pct, kelly_money = kelly_criterion_suggestion(win_prob=calculated_win_prob, odds=odds_input, bankroll=my_money)
            
            if kelly_pct > 0:
                st.markdown(f"""
                <div class='kelly-box'>
                    💡 GỢI Ý VÀO TIỀN:<br>
                    <span style='color:#e67e22; font-size: 24px'>{kelly_pct:.1f}% Vốn</span><br>
                    Tương đương: <span style='color:#27ae60; font-size: 24px'>{kelly_money:,.0f} đ</span>
                </div>
                """, unsafe_allow_html=True)
                st.info("⚠️ Đây là mức cược tối ưu toán học (Kelly an toàn). Đừng đánh hơn số này.")
            else:
                st.markdown(f"""
                <div class='kelly-box' style='border-color:#e74c3c; background-color:#fadbd8;'>
                    ⚠️ Tỷ lệ thắng hiện tại quá thấp so với Odds!<br>
                    <span style='color:#c0392b; font-size: 18px'>Kelly khuyên TẠM NGƯNG VÀO TIỀN</span>
                </div>
                """, unsafe_allow_html=True)

    # --- TAB TÀI XỈU / CHẴN LẺ ---
    with rt3:
        st.subheader("⚖️ Thống Kê Tài/Xỉu - Chẵn/Lẻ (30 kỳ gần nhất)")
        if not df_history.empty:
            recent_30 = df_history.head(30)
            all_30_nums = []
            for i in range(1, 21): all_30_nums.extend(recent_30[f'num_{i}'].tolist())
            
            tai_count = sum(1 for x in all_30_nums if x > 40)
            xiu_count = sum(1 for x in all_30_nums if x <= 40)
            chan_count = sum(1 for x in all_30_nums if x % 2 == 0)
            le_count = sum(1 for x in all_30_nums if x % 2 != 0)
            
            col_tx, col_cl = st.columns(2)
            with col_tx:
                fig_tx = px.pie(values=[tai_count, xiu_count], names=["Tài (41-80)", "Xỉu (1-40)"], title="Tỉ lệ Tài / Xỉu", hole=0.4, color_discrete_sequence=["#E74C3C", "#3498DB"])
                st.plotly_chart(fig_tx, use_container_width=True)
            with col_cl:
                fig_cl = px.pie(values=[chan_count, le_count], names=["Chẵn (Even)", "Lẻ (Odd)"], title="Tỉ lệ Chẵn / Lẻ", hole=0.4, color_discrete_sequence=["#9B59B6", "#F1C40F"])
                st.plotly_chart(fig_cl, use_container_width=True)
        else:
            st.info("Chưa có đủ dữ liệu để thống kê.")

    # --- TAB LỊCH SỬ ---
    with rt4:
        st.subheader("📜 Dữ liệu Lịch sử các kỳ")
        col_del1, col_del2 = st.columns(2)
        with col_del1:
            if st.button("Xóa kỳ cuối", use_container_width=True): delete_last_row(); st.rerun()
        with col_del2:
            if st.button("Xóa tất cả", type="primary", use_container_width=True): 
                cols = ['draw_id', 'time', 'super_num'] + [f'num_{i}' for i in range(1, 21)]
                st.session_state['df_history'] = pd.DataFrame(columns=cols)
                save_history(st.session_state['df_history'])
                st.rerun()
        if not df_history.empty: st.dataframe(df_history, use_container_width=True, hide_index=True)