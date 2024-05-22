import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 제목
st.title('Match Outcome Prediction')

# 데이터 로드 (여기서는 예시로 로컬 CSV 파일을 사용)
DATA_URL = 'https://path_to_your_data/match_data.csv'

# 데이터 불러오기 함수
@st.cache
def load_data():
    data = pd.read_csv(DATA_URL)
    return data

# 데이터 로드
data = load_data()

# 데이터 전처리
# 여기서는 간단한 전처리 예시를 사용
data['match_result'] = data['match_result'].map({'win': 1, 'loss': 0, 'draw': 2})

# 특성 선택 및 라벨
X = data[['home_team_rank', 'away_team_rank']]
y = data['match_result']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 훈련
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 모델 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Model Accuracy: {accuracy * 100:.2f}%')

# 사용자 입력 받기
st.sidebar.header('Input Features')
home_team_rank = st.sidebar.number_input('Home Team Rank', min_value=1, max_value=100, value=50)
away_team_rank = st.sidebar.number_input('Away Team Rank', min_value=1, max_value=100, value=50)

# 예측
input_data = np.array([[home_team_rank, away_team_rank]])
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

# 예측 결과 표시
result = ['Win', 'Loss', 'Draw'][prediction[0]]
st.write(f'Predicted Match Outcome: {result}')
st.write(f'Prediction Probability: {prediction_proba[0]}')
