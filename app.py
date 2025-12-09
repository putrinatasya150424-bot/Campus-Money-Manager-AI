"""
Campus Money Manager AI
Personal / Household Finance untuk Mahasiswa (Weekly view)
- Multi-dashboard (Home, Weekly Expenses, Academic Spending, Lifestyle, Savings & Goals, AI Mentor)
- Upload Excel/CSV, atau gunakan contoh data otomatis
- Chat AI menggunakan Groq LLM (GROQ_API_KEY via .env or st.secrets)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from dotenv import load_dotenv

# Attempt to import Groq - wrap with friendly error if missing
try:
    from groq import Groq
except Exception as e:
    Groq = None

# --- Load env ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY") if "st.secrets" in globals() else None

# --- App config ---
st.set_page_config(page_title="Campus Money Manager AI", page_icon="🎓💸", layout="wide")
st.title("🎓💸 Campus Money Manager AI — Weekly Edition")
st.markdown("Aplikasi pengelolaan keuangan mingguan untuk mahasiswa. Upload datamu atau coba contoh data untuk eksplorasi cepat.")

# Sidebar - navigation
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Pilih Dashboard", [
    "Home Overview",
    "Weekly Expenses",
    "Academic Spending",
    "Lifestyle Spending",
    "Savings & Goals",
    "AI Financial Mentor",
    "Settings / Data"
])

st.sidebar.markdown("---")
st.sidebar.markdown("Tips: upload file Excel/CSV dengan kolom: `Week`, `Category`, `Detail`, `Amount`, `Type` (Income/Expense).")
st.sidebar.markdown("Contoh file tersedia jika belum punya data.")

# Helper: example dataset (weekly)
def generate_example_data():
    weeks = [f"W{k}" for k in range(1,9)]  # 8 weeks
    rows = []
    for w in weeks:
        # income once per two weeks typically
        if np.random.rand() > 0.5:
            rows.append([w, "Income", "Uang Kiriman", 500000, "Income"])
        # some part-time income occasionally
        if np.random.rand() > 0.7:
            rows.append([w, "Income", "Part-time", 150000, "Income"])
        # expenses - typical student categories
        rows.append([w, "Expense", "Kos & Listrik", 400000, "Expense"])
        rows.append([w, "Expense", "Makan", int(np.random.uniform(15000, 70000)*7), "Expense"])  # weekly eating
        rows.append([w, "Expense", "Transport", int(np.random.uniform(20000, 100000)), "Expense"])
        if np.random.rand() > 0.6:
            rows.append([w, "Expense", "Buku / Print", int(np.random.uniform(20000, 150000)), "Expense"])
        if np.random.rand() > 0.8:
            rows.append([w, "Expense", "Nongkrong", int(np.random.uniform(20000, 150000)), "Expense"])
    df = pd.DataFrame(rows, columns=["Week", "Category", "Detail", "Amount", "Type"])
    return df

# Data upload or sample
st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("Upload Excel/CSV (optional)", type=["xlsx", "csv"])
use_example = st.sidebar.checkbox("Gunakan contoh data otomatis", value=(uploaded is None))

if uploaded:
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
    except Exception as e:
        st.sidebar.error(f"File error: {e}")
        df = generate_example_data()
elif use_example:
    df = generate_example_data()
else:
    st.info("Silakan upload file atau centang 'Gunakan contoh data otomatis' di sidebar.")
    st.stop()

# Validate data columns and normalize
expected_cols = ["Week", "Category", "Detail", "Amount", "Type"]
for c in expected_cols:
    if c not in df.columns:
        # attempt to infer common alternatives
        if c.lower() in map(str.lower, df.columns):
            df.columns = [c if c.lower()==c2.lower() else c2 for c, c2 in zip(expected_cols, expected_cols)]
        else:
            st.error(f"Dataset wajib memiliki kolom: {', '.join(expected_cols)}. Kolom '{c}' tidak ditemukan.")
            st.stop()

# Clean Amount to numeric
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0)
df["Type"] = df["Type"].astype(str).str.capitalize()
df["Week"] = df["Week"].astype(str)

# Common aggregates
weekly_sum = df.groupby(["Week", "Type"])["Amount"].sum().unstack(fill_value=0).reset_index()
weekly_sum["Net"] = weekly_sum.get("Income", 0) - weekly_sum.get("Expense", 0)
total_income = df.loc[df["Type"] == "Income", "Amount"].sum()
total_expense = df.loc[df["Type"] == "Expense", "Amount"].sum()
net_total = total_income - total_expense

# --- PAGES ---
if page == "Home Overview":
    st.header("🏠 Home Overview (Weekly)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Income (All Weeks)", f"Rp {int(total_income):,}")
    col2.metric("Total Expense (All Weeks)", f"Rp {int(total_expense):,}")
    col3.metric("Net Total", f"Rp {int(net_total):,}")
    # Average weekly spend
    avg_week_expense = weekly_sum["Expense"].mean() if "Expense" in weekly_sum else 0
    col4.metric("Avg Weekly Expense", f"Rp {int(avg_week_expense):,}")
    
    st.markdown("### 📅 Tren Mingguan (Income vs Expense vs Net)")
    fig = px.line(weekly_sum, x="Week", y=[c for c in ["Income", "Expense", "Net"] if c in weekly_sum.columns],
                  markers=True, title="Trend Mingguan")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 🔍 Top 5 Pengeluaran (Total)")
    top_exp = df[df["Type"]=="Expense"].groupby("Detail")["Amount"].sum().sort_values(ascending=False).head(5).reset_index()
    st.table(top_exp.assign(Amount=lambda d: d["Amount"].map(lambda x: f"Rp {int(x):,}")))
    
    # Alerts - simple overspending detection
    st.markdown("### ⚠️ Alerts & Tips")
    # detect any week where expense > income
    deficit_weeks = weekly_sum[weekly_sum.get("Income",0) < weekly_sum.get("Expense",0)]
    if not deficit_weeks.empty:
        st.warning(f"Kamu mengalami defisit pada minggu: {', '.join(deficit_weeks['Week'].tolist())}. Pertimbangkan kurangi pengeluaran atau cari tambahan pemasukan.")
    else:
        st.success("Tidak ada minggu defisit — bagus! Jaga konsistensi ini 🙂")
    
    # Quick simulation: if daily coffee hábito change
    st.markdown("### ☕ Simulasi: Pengaruh Pengeluaran Kopi Harian")
    coffee_price = st.number_input("Harga kopi per hari (Rp)", min_value=0, value=15000, step=1000)
    days_per_week = st.slider("Berapa kali per minggu?", 0, 7, 5)
    extra_weekly = coffee_price * days_per_week
    st.info(f"Jika kamu minum kopi Rp{coffee_price:,} sebanyak {days_per_week}x/minggu → tambahan pengeluaran Rp{extra_weekly:,}/minggu → Rp{extra_weekly*4:,}/bulan.")

elif page == "Weekly Expenses":
    st.header("🧾 Weekly Expenses")
    st.markdown("Filter dan analisis pengeluaran per minggu.")
    weeks = sorted(df["Week"].unique().tolist())
    chosen_week = st.selectbox("Pilih Week", weeks, index=len(weeks)-1)
    df_week = df[df["Week"] == chosen_week]
    
    st.subheader(f"Detail transaksi — {chosen_week}")
    st.dataframe(df_week.reset_index(drop=True))
    
    st.markdown("### Pengeluaran menurut Detail (Pie)")
    df_exp = df_week[df_week["Type"]=="Expense"]
    if not df_exp.empty:
        fig = px.pie(df_exp, names="Detail", values="Amount", title=f"Pengeluaran {chosen_week}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Tidak ada pengeluaran pada minggu ini.")
    
    st.markdown("### Heatmap kategori per minggu (Expense totals across weeks)")
    pivot = df[df["Type"]=="Expense"].groupby(["Week","Detail"])["Amount"].sum().reset_index()
    if not pivot.empty:
        heat = pivot.pivot(index="Detail", columns="Week", values="Amount").fillna(0)
        st.dataframe(heat.style.format("{:,.0f}"))
    else:
        st.info("Tidak cukup data untuk heatmap.")

elif page == "Academic Spending":
    st.header("📚 Academic Spending")
    st.markdown("Fokus pada pengeluaran yang berhubungan langsung dengan studi.")
    academic_keywords = st.text_input("Masukkan kata kunci kategori (pisahkan koma), contoh: Buku, Print, Praktikum", value="Buku,Print,Praktikum")
    keywords = [k.strip().lower() for k in academic_keywords.split(",") if k.strip()]
    mask = df["Detail"].str.lower().apply(lambda s: any(k in s for k in keywords))
    academic_df = df[mask & (df["Type"]=="Expense")]
    st.subheader("Transaksi Akademik")
    if not academic_df.empty:
        st.dataframe(academic_df)
        tot_acad = academic_df["Amount"].sum()
        st.markdown(f"**Total Pengeluaran Akademik:** Rp {int(tot_acad):,}")
        fig = px.bar(academic_df.groupby("Week")["Amount"].sum().reset_index(), x="Week", y="Amount", title="Akademik: Pengeluaran per Minggu")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Tidak ditemukan transaksi akademik dengan kata kunci tersebut.")

elif page == "Lifestyle Spending":
    st.header("🎯 Lifestyle Spending")
    st.markdown("Analisis pengeluaran untuk gaya hidup (makan, hiburan, nongkrong).")
    lifestyle_keywords = ["makan", "nongkrong", "hiburan", "game", "fashion", "kopi"]
    st.markdown(f"Default keywords: {', '.join(lifestyle_keywords)}")
    mask = df["Detail"].str.lower().apply(lambda s: any(k in s for k in lifestyle_keywords))
    life_df = df[mask & (df["Type"]=="Expense")]
    st.subheader("Transaksi Lifestyle")
    if not life_df.empty:
        st.dataframe(life_df)
        group = life_df.groupby("Detail")["Amount"].sum().reset_index().sort_values("Amount", ascending=False)
        st.markdown("### Top lifestyle spends")
        st.table(group)
        fig = px.treemap(group, path=["Detail"], values="Amount", title="Lifestyle Spend Treemap")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Tidak ada transaksi lifestyle terdeteksi (sesuaikan kata kunci jika perlu).")

elif page == "Savings & Goals":
    st.header("💰 Savings & Goals")
    st.markdown("Buat target tabungan (goal) dan pantau progress berdasarkan transaksi weekly.")
    if "savings_goals" not in st.session_state:
        st.session_state.savings_goals = []
    # Input new goal
    with st.form("new_goal"):
        gname = st.text_input("Nama goal (contoh: Laptop, Dana Darurat)", key="gname")
        gtarget = st.number_input("Target amount (Rp)", min_value=0, step=10000, key="gtarget")
        gdeadline = st.text_input("Deadline (opsional)", value="3 bulan", key="gdeadline")
        submitted = st.form_submit_button("Tambah Goal")
        if submitted and gname and gtarget > 0:
            st.session_state.savings_goals.append({"name": gname, "target": gtarget, "deadline": gdeadline})
            st.success(f"Goal '{gname}' ditambahkan.")

    # Show goals and progress
    if st.session_state.savings_goals:
        for i, g in enumerate(st.session_state.savings_goals):
            # Calculate current saved for this goal based on transactions tagged as 'Tabungan' in Detail
            # (Simple heuristic: sum of Details that contain goal name or 'Tabungan')
            saved_mask = df["Detail"].str.contains(g["name"], case=False) | df["Detail"].str.contains("Tabungan", case=False)
            saved_amount = df[saved_mask & (df["Type"]=="Expense")]["Amount"].sum() * -1  # if savings recorded as negative expense
            # If user records savings as Income->Type Income we consider it
            saved_amount_income = df[saved_mask & (df["Type"]=="Income")]["Amount"].sum()
            saved = int(saved_amount_income - saved_amount)
            progress = min(1.0, saved / g["target"]) if g["target"] > 0 else 0
            st.markdown(f"**{g['name']}** — Target: Rp {int(g['target']):,} — Deadline: {g['deadline']}")
            st.progress(progress)
            st.write(f"Saved: Rp {saved:,} ({int(progress*100)}%)")
            if progress >= 1.0:
                st.balloons()
                st.success(f"Goal '{g['name']}' tercapai 🎉")
    else:
        st.info("Belum ada goal. Buat goal tabunganmu di form di atas.")

elif page == "AI Financial Mentor":
    st.header("🤖 AI Financial Mentor (Chat)")
    st.markdown("Tanya tentang budgeting mahasiswa, cara hemat, rekomendasi tabungan/investasi kecil, dsb.")
    if Groq is None or not GROQ_API_KEY:
        st.error("Groq SDK atau GROQ_API_KEY tidak ditemukan. Untuk mengaktifkan AI: install 'groq' dan set environment variable GROQ_API_KEY di .env atau st.secrets.")
        st.info("Meski AI tidak aktif, kamu tetap bisa pakai fitur-fitur analitis lainnya.")
    else:
        client = Groq(api_key=GROQ_API_KEY)
        # model selector (limited)
        model = st.selectbox("Pilih Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"], index=0)
        # chat UI
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        user_input = st.text_input("Tanyakan sesuatu ke AI (contoh: 'Bagaimana cara hemat 500k/bulan?')", key="mentor_input")
        col1, col2 = st.columns([4,1])
        send = col1.button("Send")
        reset = col2.button("Reset Chat")
        if reset:
            st.session_state.chat_history = []
            st.success("Chat cleared.")
        # Prepare a short preview of recent transactions to give AI context
        df_preview = df.tail(20).to_string(index=False)
        system_msg = {
            "role":"system",
            "content": "You are a helpful personal finance mentor specialized for university students. Provide practical, concise, step-by-step suggestions. Prioritize low-cost/high-impact tips."
        }
        if send and user_input:
            try:
                messages = [system_msg] + st.session_state.chat_history + [{"role":"user", "content": f"Recent transactions:\n{df_preview}\n\nUser question: {user_input}"}]
                resp = client.chat.completions.create(messages=messages, model=model)
                answer = resp.choices[0].message.content
                st.session_state.chat_history.append({"role":"user", "content": user_input})
                st.session_state.chat_history.append({"role":"assistant", "content": answer})
            except Exception as e:
                st.error(f"AI request failed: {e}")

        # render chat
        if st.session_state.get("chat_history"):
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"**👤 You:** {msg['content']}")
                else:
                    st.markdown(f"**🤖 Mentor:** {msg['content']}")

elif page == "Settings / Data":
    st.header("⚙️ Settings & Data")
    st.markdown("Preview raw data & export options.")
    st.subheader("Raw Data")
    st.dataframe(df)
    st.markdown("### Export")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="campus_money_data.csv", mime="text/csv")
    st.markdown("### Clear goals and chat (local session only)")
    if st.button("Clear session goals & chat"):
        st.session_state.pop("savings_goals", None)
        st.session_state.pop("chat_history", None)
        st.success("Session cleared.")

# End of app
st.markdown("---")
st.markdown("Made with ❤️ — Campus Money Manager AI. Kamu bisa fork repo ini dan sesuaikan kategori / kata kunci sesuai kebiasaanmu.")
