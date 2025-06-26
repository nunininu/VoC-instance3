import streamlit as st

st.title("상담 데이터 시각화 대시보드")

# 대표 이미지 (대시보드 미리보기 스크린샷 등)
st.image("/home/gmlwls5168/code/wh04-3rd-1team-VoCabulary/data/dashboard.png", caption="상담 분석 대시보드")

# Superset 링크 연결 버튼
dashboard_url = "https://e935-118-36-174-103.ngrok-free.app/"
st.markdown(f"[👉 Superset 대시보드 바로가기]({dashboard_url})", unsafe_allow_html=True)


