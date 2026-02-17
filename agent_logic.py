import json
import ollama
from ollama import Client

# إعداد العميل (تأكد من تشغيل Ollama)
client = Client(host='http://localhost:11434')
MODEL = "qwen3:0.6b"  # أو الاسم الدقيق للموديل عندك

SYSTEM_PROMPT = """You are an expert predictive maintenance assistant for ReneWind turbines.
Your role is to analyze sensor data (V1-V40) and RUL (Remaining Useful Life).

RULES:
1. Always use the provided data to answer.
2. SEVERITY LEVELS: 
   - Critical: Risk > 70% or RUL < 30 cycles.
   - Warning: Risk 40-70%.
   - Healthy: Risk < 40%.
3. If a sensor value is > 1.5 or < -1.5, it's an outlier.
4. Be concise and professional.
"""


def ask_agent(df, user_query):
    """
    وظيفة الوكيل الذكي التي تستقبل البيانات وسؤال المستخدم
    """
    try:
        # تحويل جزء من البيانات لسياق نصي (Context) ليفهمه الموديل الصغير
        # نأخذ أول 5 أسطر كمثال عام أو السطر المختار
        context_data = df.head(3).to_json(orient='records')

        full_prompt = f"""
        {SYSTEM_PROMPT}

        DATA CONTEXT (JSON):
        {context_data}

        USER QUESTION:
        {user_query}

        ASSISTANT RESPONSE:"""

        response = client.chat(
            model=MODEL,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': full_prompt}
            ],
            options={'temperature': 0.2}  # درجة حرارة منخفضة لضمان الدقة
        )

        return response['message']['content']

    except Exception as e:
        return f"⚠️ Agent Error: {str(e)}. Make sure Ollama is running with {MODEL}"