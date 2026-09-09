import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_config, bytes_to_image, bgr_to_rgb, draw_face_box, ensure_dirs
from src.face_engine import FaceEngine
from src.database import EmployeeDatabase
from src.liveness import check_liveness
from src.attendance import append_record, read_log

CONFIG = load_config("config.yaml")

ensure_dirs(
    CONFIG["paths"]["visualizations"],
    CONFIG["paths"]["models_dir"],
    os.path.dirname(CONFIG["paths"]["embeddings_db"]),
    os.path.dirname(CONFIG["paths"]["attendance_log"]),
)

st.set_page_config(
    page_title="FaceAttend | Face Recognition Attendance",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.metric-card {
    background: linear-gradient(145deg, #171B26, #12151F);
    border: 1px solid #2A2F3E;
    border-radius: 16px;
    padding: 22px 24px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.25);
}

.metric-label {
    font-size: 13px;
    color: #9AA1B4;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
}

.metric-value {
    font-size: 30px;
    font-weight: 700;
    color: #F2F3F7;
}

.metric-delta-positive {
    color: #33D399;
    font-size: 13px;
    font-weight: 600;
}

.metric-delta-negative {
    color: #FF6B6B;
    font-size: 13px;
    font-weight: 600;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #F2F3F7;
    margin-top: 6px;
    margin-bottom: 14px;
}

.section-subtitle {
    font-size: 14px;
    color: #9AA1B4;
    margin-bottom: 22px;
}

.status-pill-match {
    display: inline-block;
    background: rgba(51, 211, 153, 0.14);
    color: #33D399;
    border: 1px solid rgba(51, 211, 153, 0.35);
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 13px;
    font-weight: 600;
}

.status-pill-manual {
    display: inline-block;
    background: rgba(255, 190, 92, 0.14);
    color: #FFBE5C;
    border: 1px solid rgba(255, 190, 92, 0.35);
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 13px;
    font-weight: 600;
}

.status-pill-reject {
    display: inline-block;
    background: rgba(255, 107, 107, 0.14);
    color: #FF6B6B;
    border: 1px solid rgba(255, 107, 107, 0.35);
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 13px;
    font-weight: 600;
}

.identity-card {
    background: linear-gradient(145deg, #171B26, #12151F);
    border: 1px solid #2A2F3E;
    border-radius: 18px;
    padding: 26px;
    text-align: center;
}

.identity-name {
    font-size: 26px;
    font-weight: 800;
    color: #F2F3F7;
    margin-top: 8px;
}

.identity-id {
    font-size: 13px;
    color: #9AA1B4;
}

div[data-testid="stSidebar"] {
    background-color: #0B0D13;
    border-right: 1px solid #202433;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading face recognition model...")
def get_engine():
    model_cfg = CONFIG["model"]
    return FaceEngine(
        model_name=model_cfg["name"],
        detection_size=tuple(model_cfg["detection_size"]),
        ctx_id=model_cfg["ctx_id"],
    )


def get_database():
    return EmployeeDatabase(CONFIG["paths"]["embeddings_db"])


def metric_card(label, value, delta=None, positive=True):
    delta_html = ""
    if delta is not None:
        css_class = "metric-delta-positive" if positive else "metric-delta-negative"
        delta_html = f'<div class="{css_class}">{delta}</div>'
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def confidence_gauge(confidence, threshold):
    color = "#33D399" if confidence >= threshold else "#FF6B6B"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        number={"suffix": "%", "font": {"size": 40, "color": "#F2F3F7"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#9AA1B4"},
            "bar": {"color": color},
            "bgcolor": "#161A25",
            "borderwidth": 0,
            "steps": [
                {"range": [0, threshold], "color": "rgba(255,107,107,0.12)"},
                {"range": [threshold, 100], "color": "rgba(51,211,153,0.12)"},
            ],
            "threshold": {
                "line": {"color": "#F2F3F7", "width": 3},
                "thickness": 0.8,
                "value": threshold,
            },
        },
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E6E6E6"},
    )
    return fig


def sidebar_nav():
    with st.sidebar:
        st.markdown("### FaceAttend")
        st.caption("Face Recognition Attendance System")
        st.markdown("---")
        selected = option_menu(
            menu_title=None,
            options=["Dashboard", "Verify Attendance", "Enroll Employee", "Employee Database", "Analytics"],
            icons=["speedometer2", "camera", "person-plus", "people", "bar-chart"],
            default_index=0,
            styles={
                "container": {"padding": "0", "background-color": "transparent"},
                "icon": {"color": "#9AA1B4", "font-size": "16px"},
                "nav-link": {
                    "font-size": "14px",
                    "text-align": "left",
                    "margin": "3px 0",
                    "border-radius": "10px",
                    "color": "#C6CAD6",
                },
                "nav-link-selected": {"background-color": "#6C5CE7", "color": "#FFFFFF"},
            },
        )
        st.markdown("---")
        db = get_database()
        st.caption(f"Registered employees: {db.count()}")
        st.caption(f"Similarity threshold: {CONFIG['recognition']['similarity_threshold']}")
        st.caption(f"Confidence threshold: {CONFIG['recognition']['confidence_threshold_percent']}%")
    return selected


def page_dashboard():
    st.markdown('<div class="section-title">Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Overview of attendance activity and system health</div>', unsafe_allow_html=True)

    db = get_database()
    log_df = read_log(CONFIG["paths"]["attendance_log"])

    today = pd.Timestamp.now().normalize()
    today_df = pd.DataFrame()
    if not log_df.empty:
        log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])
        today_df = log_df[log_df["timestamp"].dt.normalize() == today]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Registered Employees", db.count())
    with col2:
        metric_card("Check-ins Today", len(today_df))
    with col3:
        avg_conf = f"{today_df['confidence_percent'].mean():.1f}%" if not today_df.empty else "-"
        metric_card("Avg Confidence Today", avg_conf)
    with col4:
        if not today_df.empty:
            auto_rate = (today_df["status"] == "auto_verified").mean() * 100
            metric_card("Auto-verified Rate", f"{auto_rate:.0f}%")
        else:
            metric_card("Auto-verified Rate", "-")

    st.write("")
    left, right = st.columns([2, 1])

    with left:
        st.markdown('<div class="section-title" style="font-size:18px;">Recent Activity</div>', unsafe_allow_html=True)
        if log_df.empty:
            st.info("No attendance records yet. Run a verification from the Verify Attendance page.")
        else:
            display_df = log_df.sort_values("timestamp", ascending=False).head(12)
            display_df = display_df[["timestamp", "name", "confidence_percent", "status", "verification_method"]]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    with right:
        st.markdown('<div class="section-title" style="font-size:18px;">Status Breakdown</div>', unsafe_allow_html=True)
        if log_df.empty:
            st.info("No data to summarize yet.")
        else:
            status_counts = log_df["status"].value_counts().reset_index()
            status_counts.columns = ["status", "count"]
            fig = px.pie(
                status_counts, names="status", values="count", hole=0.55,
                color_discrete_sequence=["#6C5CE7", "#33D399", "#FFBE5C", "#FF6B6B"],
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#E6E6E6"},
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig, use_container_width=True)

    if not log_df.empty:
        st.write("")
        st.markdown('<div class="section-title" style="font-size:18px;">Attendance Trend</div>', unsafe_allow_html=True)
        trend_df = log_df.copy()
        trend_df["date"] = trend_df["timestamp"].dt.date
        daily = trend_df.groupby("date").size().reset_index(name="check_ins")
        fig = px.area(daily, x="date", y="check_ins", color_discrete_sequence=["#6C5CE7"])
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#E6E6E6"},
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)


def page_verify():
    st.markdown('<div class="section-title">Verify Attendance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Scan employee face to record attendance</div>', unsafe_allow_html=True)

    db = get_database()
    if db.is_empty():
        st.warning("No employees enrolled yet. Go to Enroll Employee first.")
        return

    engine = get_engine()
    rec_cfg = CONFIG["recognition"]
    live_cfg = CONFIG["liveness"]

    col_cam, col_result = st.columns([1, 1.2])

    with col_cam:
        capture = st.camera_input("Scan face", label_visibility="collapsed")

    if capture is not None:
        image = bytes_to_image(capture.getvalue())
        embedding, face = engine.get_embedding(image)

        with col_result:
            if embedding is None:
                st.error("No face detected. Please try again with better lighting and framing.")
                return

            live_passed, live_reasons, texture_score = True, [], None
            if live_cfg["enabled"]:
                live_passed, live_reasons, texture_score = check_liveness(
                    image, face,
                    blur_threshold=live_cfg["blur_threshold"],
                    min_face_size=live_cfg["min_face_size"],
                )

            matches = engine.search(embedding, db, top_k=rec_cfg["top_k"])
            top_match = matches[0] if matches else None

            if top_match:
                eid, name, sim = top_match
                confidence = engine.similarity_to_confidence(
                    sim, rec_cfg["similarity_low_bound"], rec_cfg["similarity_high_bound"]
                )
            else:
                eid, name, sim, confidence = None, None, 0.0, 0.0

            threshold_pct = rec_cfg["confidence_threshold_percent"]
            auto_verified = confidence >= threshold_pct and live_passed

            st.plotly_chart(confidence_gauge(confidence, threshold_pct), use_container_width=True)

            if not live_passed:
                st.markdown('<span class="status-pill-reject">Liveness check failed</span>', unsafe_allow_html=True)
                st.caption(f"Reasons: {', '.join(live_reasons)}")

            if top_match:
                st.markdown(
                    f"""
                    <div class="identity-card">
                        <div class="identity-id">Best match</div>
                        <div class="identity-name">{name}</div>
                        <div class="identity-id">ID: {eid} | similarity: {sim:.3f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.write("")

            if auto_verified:
                st.markdown('<span class="status-pill-match">Identified above threshold</span>', unsafe_allow_html=True)
                st.write(f"Apakah ini {name}?")
                yc1, yc2 = st.columns(2)
                if yc1.button("Ya, benar", use_container_width=True, type="primary"):
                    append_record(
                        CONFIG["paths"]["attendance_log"], eid, name, sim, confidence,
                        "auto_verified", "face_recognition",
                    )
                    st.success(f"Attendance recorded for {name}.")
                if yc2.button("Bukan saya", use_container_width=True):
                    append_record(
                        CONFIG["paths"]["attendance_log"], eid, name, sim, confidence,
                        "manual_required", "face_recognition_rejected",
                    )
                    st.warning("Marked for manual verification.")
            else:
                if confidence < threshold_pct:
                    st.markdown('<span class="status-pill-manual">Below confidence threshold</span>', unsafe_allow_html=True)
                    st.write(f"Confidence {confidence:.1f}% is below the required {threshold_pct:.0f}%. Please verify manually.")
                else:
                    st.markdown('<span class="status-pill-manual">Liveness check did not pass</span>', unsafe_allow_html=True)
                    st.write("Confidence score is sufficient, but the liveness check failed. Please verify manually or retake the photo with better lighting/focus.")

                employees = db.list_employees()
                options = {f"{e['name']} ({e['id']})": e["id"] for e in employees}
                selected_label = st.selectbox("Select correct employee", list(options.keys()))
                if st.button("Confirm manual verification", type="primary"):
                    manual_eid = options[selected_label]
                    manual_name = selected_label.split(" (")[0]
                    append_record(
                        CONFIG["paths"]["attendance_log"], manual_eid, manual_name, sim, confidence,
                        "manual_verified", "manual",
                    )
                    st.success(f"Manual attendance recorded for {manual_name}.")


def page_enroll():
    st.markdown('<div class="section-title">Enroll Employee</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Register a new employee with multiple face samples</div>', unsafe_allow_html=True)

    engine = get_engine()
    db = get_database()
    min_samples = CONFIG["recognition"]["min_samples_per_employee"]

    with st.form("enroll_form"):
        name = st.text_input("Full name")
        employee_code = st.text_input("Employee ID (optional, auto-generated if empty)")
        uploaded_files = st.file_uploader(
            f"Upload at least {min_samples} face photos (different angles/lighting)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )
        submitted = st.form_submit_button("Process and enroll", type="primary")

    if submitted:
        if not name:
            st.error("Name is required.")
            return
        if not uploaded_files or len(uploaded_files) < min_samples:
            st.error(f"Please upload at least {min_samples} photos.")
            return

        images = [bytes_to_image(f.getvalue()) for f in uploaded_files]
        embeddings, faces = engine.enroll_from_images(images)

        if len(embeddings) < min_samples:
            st.error(
                f"Only detected {len(embeddings)} valid face(s) out of {len(images)} photos. "
                f"Please retake photos with clearer face visibility."
            )
            return

        preview_cols = st.columns(min(len(faces), 5))
        for i, (img, face) in enumerate(zip(images[:len(faces)], faces)):
            annotated = draw_face_box(img, face.bbox, label="face detected")
            with preview_cols[i % len(preview_cols)]:
                st.image(bgr_to_rgb(annotated), use_container_width=True)

        employee_id = db.enroll(name, embeddings, employee_id=employee_code or None)
        st.success(f"Enrolled {name} successfully with ID {employee_id} and {len(embeddings)} samples.")
        st.balloons()


def page_database():
    st.markdown('<div class="section-title">Employee Database</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Manage enrolled employees and their embeddings</div>', unsafe_allow_html=True)

    db = get_database()
    employees = db.list_employees()

    if not employees:
        st.info("No employees enrolled yet.")
        return

    df = pd.DataFrame(employees)
    search = st.text_input("Search by name")
    if search:
        df = df[df["name"].str.contains(search, case=False, na=False)]

    st.dataframe(df, use_container_width=True, hide_index=True)

    st.write("")
    st.markdown('<div class="section-title" style="font-size:18px;">Remove Employee</div>', unsafe_allow_html=True)
    options = {f"{e['name']} ({e['id']})": e["id"] for e in employees}
    selected_label = st.selectbox("Select employee to remove", list(options.keys()))
    if st.button("Remove employee", type="secondary"):
        db.remove(options[selected_label])
        st.success("Employee removed.")
        st.rerun()


def page_analytics():
    st.markdown('<div class="section-title">Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Insights into recognition performance and attendance patterns</div>', unsafe_allow_html=True)

    log_df = read_log(CONFIG["paths"]["attendance_log"])
    if log_df.empty:
        st.info("No attendance data yet to analyze.")
        return

    log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])
    threshold_pct = CONFIG["recognition"]["confidence_threshold_percent"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title" style="font-size:18px;">Confidence Score Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(log_df, x="confidence_percent", nbins=25, color_discrete_sequence=["#6C5CE7"])
        fig.add_vline(x=threshold_pct, line_dash="dash", line_color="#FF6B6B")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#E6E6E6"}, margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title" style="font-size:18px;">Check-ins per Employee</div>', unsafe_allow_html=True)
        by_employee = log_df.groupby("name").size().reset_index(name="check_ins").sort_values("check_ins", ascending=False)
        fig = px.bar(by_employee, x="name", y="check_ins", color_discrete_sequence=["#33D399"])
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#E6E6E6"}, margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.write("")
    st.markdown('<div class="section-title" style="font-size:18px;">Verification Method Over Time</div>', unsafe_allow_html=True)
    log_df["date"] = log_df["timestamp"].dt.date
    method_trend = log_df.groupby(["date", "verification_method"]).size().reset_index(name="count")
    fig = px.bar(
        method_trend, x="date", y="count", color="verification_method", barmode="stack",
        color_discrete_sequence=["#6C5CE7", "#33D399", "#FFBE5C", "#FF6B6B"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E6E6E6"}, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("")
    st.markdown('<div class="section-title" style="font-size:18px;">Full Log</div>', unsafe_allow_html=True)
    st.dataframe(log_df.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)


def main():
    page = sidebar_nav()
    if page == "Dashboard":
        page_dashboard()
    elif page == "Verify Attendance":
        page_verify()
    elif page == "Enroll Employee":
        page_enroll()
    elif page == "Employee Database":
        page_database()
    elif page == "Analytics":
        page_analytics()


if __name__ == "__main__":
    main()