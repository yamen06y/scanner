import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# بيانات تدريب بسيطة (وهمية) لتعليم النموذج
training_data = {
    'team_home_score_avg': [100, 95, 110, 90, 105, 97],
    'team_away_score_avg': [98, 101, 102, 87, 99, 94],
    'team_home_win_rate': [0.7, 0.5, 0.8, 0.4, 0.6, 0.55],
    'team_away_win_rate': [0.6, 0.4, 0.75, 0.3, 0.65, 0.5],
    'team_home_wins': [1, 0, 1, 0, 1, 1]  # 1 = فوز الفريق المضيف
}

df = pd.DataFrame(training_data)

# تدريب النموذج
X = df.drop('team_home_wins', axis=1)
y = df['team_home_wins']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# إدخال البيانات من المستخدم
print("\n🏟️ أدخل إحصائيات المباراة:")
home_score_avg = float(input("متوسط أهداف الفريق المضيف: "))
away_score_avg = float(input("متوسط أهداف الفريق الضيف: "))
home_win_rate = float(input("نسبة فوز الفريق المضيف (من 0 إلى 1): "))
away_win_rate = float(input("نسبة فوز الفريق الضيف (من 0 إلى 1): "))

# إنشاء إطار بيانات واحد لتوقع النتيجة
new_data = pd.DataFrame([{
    'team_home_score_avg': home_score_avg,
    'team_away_score_avg': away_score_avg,
    'team_home_win_rate': home_win_rate,
    'team_away_win_rate': away_win_rate
}])

# التوقع
prediction = model.predict(new_data)[0]
prob = model.predict_proba(new_data)[0]

# عرض التوقع
print("\n🔍 التوقع: الفريق المضيف", "سيفوز ✅" if prediction else "سيخسر ❌")
print(f"نسبة الثقة: {prob[prediction]*100:.2f}%")

