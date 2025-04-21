# GOAT Stock Alert

매집 돌파 감지 및 알림 시스템

## 설치
```bash
git clone https://github.com/luck777756/stock-alert.git
cd stock-alert
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
구성
main.py: 워커 서비스 진입점

train_model_updated.py: 모델 학습 스크립트

utils/common_utils.py: 공통 함수 (특징 생성, 점수 계산 등)

tests/: 단위 테스트 디렉터리

.github/workflows/ci.yml: CI 설정

tickers_nasdaq.txt: NASDAQ 티커 리스트

사용법
알림 워커 실행
bash
복사
편집
python3 main.py
모델 학습
bash
복사
편집
bash train.sh
환경 변수
NTFY_CHANNEL: ntfy 알림 채널 URL (기본: https://ntfy.sh/my-stock-alert)
