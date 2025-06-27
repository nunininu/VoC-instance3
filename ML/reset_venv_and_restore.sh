#!/bin/bash

echo "📦 1. 패키지 목록 백업 중..."
pdm export --without-urls --without-hashes > requirements_backup.txt

echo "🧹 2. 기존 .venv 삭제 중..."
rm -rf .venv

echo "🐍 3. Python 3.12 기반 가상환경 재생성 중..."
pdm use -f 3.12

echo "📥 4. 패키지 재설치 (pdm install)"
pdm install

echo "📦 5. 이전 패키지 복원 중..."
pdm add -dG :all $(cat requirements_backup.txt | grep -v '^#')

echo "📊 6. 분석용 패키지 추가 설치 (no-lock)"
pdm add pyvis matplotlib python-louvain networkx pandas --no-lock

echo "✅ 완료: 가상환경 복구 및 분석용 도구 재설치 완료"
