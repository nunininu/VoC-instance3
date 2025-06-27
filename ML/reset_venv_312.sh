#!/bin/bash

echo "🔁 기존 .venv 삭제 중..."
rm -rf .venv

echo "🐍 Python 3.12로 새로운 가상환경 생성 중..."
python3.12 -m venv .venv

echo "✅ 가상환경 활성화"
source .venv/bin/activate

echo "🔗 PDM에 새 Python 연결"
pdm use -f .venv/bin/python

echo "📦 기존 의존성 복원"
pdm install

echo "➕ pyvis, jinja2 설치"
pdm add pyvis jinja2

echo "🎉 완료: Python 3.12 기반 .venv 설정 완료"
