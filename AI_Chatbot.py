import streamlit as st
import tempfile
from gtts import gTTS
from gtts.tts import gTTSError
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import (
    ChatPromptTemplate,
    
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_classic.memory import ConversationBufferWindowMemory


# ---------------- CONFIG ---------------- #
MODEL_NAME = "openai/gpt-oss-120b"
TEMPERATURE = 0.5


# ---------------- PAGE SETUP ---------------- #
st.set_page_config(
    page_title="AuraAssist",
    page_icon="aura_robot_icon_64.png",  # put file in project folder
    layout="wide"
)

import base64

# ---- LOAD ICON ----
def load_icon_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

icon_base64 = load_icon_base64("aura_robot_icon_64.png")
# ------------------- APPLY MAIN BACKGROUND ------------------- #

st.markdown("""
<style>

/* Sidebar */
[data-testid="stSidebar"] {
    background-color:  #ff8533;
    color: #2B1A17;
}

/* Main middle chat area */
section.main > div {
    background-color: #ffb380;
    padding: 2rem;
    border-radius: 15px;
    color: #2B1A17;
}

/* Top & Bottom background */
.stApp {
    background-color:;
}

/* Input bar */
chat-input {
    background-color:  #ff8533 !important;
    color: #2B1A17 !important;
    border-radius: 20px;
    border: none;
}

/* Buttons */
.stButton>button {
    background-color: #ffffff !important;
    color: #2B1A17 !important;
    border-radius: 15px;
    border: none;
}

/* Optional: Chat message bubble */
.stChatMessage {
    background-color: #ffa366 !important;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)



col1, col2 = st.columns([1,8])  # Adjust ratio if needed

with col1:
    st.image("emoji.png", width=100)

with col2:
    st.title("AuraAssist")

st.caption("A supportive space for reflection. Not a replacement for professional care.")

# ---------------- TEXT CLEANING FOR TTS ---------------- #
def clean_text_for_tts(text):
    if "---" in text:
        text = text.split("---")[0]

    text = re.sub(r'[⭐☆*#\-—_`]', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# ---------------- TEXT TO SPEECH ---------------- #
def text_to_speech(text):
    cleaned_text = clean_text_for_tts(text)

    # Limit large text (Google may reject long input)
    cleaned_text = cleaned_text[:500]

    try:
        tts = gTTS(text=cleaned_text, lang="en")
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        return temp_file.name

    except gTTSError as e:
        print("TTS Connection Error:", e)
        return None


# ---------------- SIDEBAR PROFILE ---------------- #
st.sidebar.header("👤 Your Profile")
api_key= st.sidebar.text_input("Groq API Key", type="password",help="Enter your Groq API key to enable the AI assistant. You can get one from https://console.groq.com/keys")
if not api_key:
    st.warning("Please enter your Groq API key to use the AI assistant features.")
    
name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=25)
gender = st.sidebar.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Other"])
occupation = st.sidebar.text_input("Occupation")
addictions = st.sidebar.text_input("Addictions (if any)")
stress_triggers = st.sidebar.text_input("Main Stress Triggers")
previous_treatment = st.sidebar.text_input("Previous Treatment (if any)")

if st.sidebar.button("Save Profile"):
    st.session_state.user_profile = {
        "name": name,
        "age": age,
        "gender": gender,
        "occupation": occupation,
        "addictions": addictions,
        "stress_triggers": stress_triggers,
        "previous_treatment": previous_treatment,
    }
    st.sidebar.success("Profile saved successfully!")

if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}

# ---------------- MEMORY ---------------- #
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=6,
        return_messages=True,
        memory_key="chat_history"
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- LLM INIT (Only Once) ---------------- #
try:
     if "llm" not in st.session_state:
        st.session_state.llm = ChatGroq(
            groq_api_key=api_key,
            model_name=MODEL_NAME,
            temperature=TEMPERATURE
        )
except Exception as e:
    st.error("Error initializing LLM. Please check your API key and connection.")
    print(e)

# ---------------- SYSTEM PROMPT ---------------- #
system_prompt = """
You are a compassionate, emotionally intelligent mental health support assistant with a friendly, slightly sarcastic, human-like tone.

Use retrieved knowledge naturally — do not depend on it completely.

Retrieved Knowledge:
{context}

User Profile:
{profile}

Conversation History:
{chat_history}

User Message:
{question}

----------------------------------------------------
STEP 1: SEVERITY ANALYSIS
----------------------------------------------------
Analyze the user's message and internally classify it as:

• MILD → Occasional stress, low mood, manageable symptoms  
• MODERATE → Persistent distress affecting work, relationships, routine  
• SEVERE → Debilitating symptoms, suicidal ideation, psychosis, inability to function  

Do NOT explicitly label the severity unless necessary.
Adjust response depth accordingly.

----------------------------------------------------
STEP 2: RESPONSE RULES
----------------------------------------------------
- Identify and reflect the user’s emotions
- Be empathetic and supportive
- Friendly, warm, slightly playful tone and sarcastic humor
- Never diagnose medical conditions
- Never prescribe medication
- Never shame addiction
- If prior treatment exists, acknowledge progress respectfully
- Only respond to mental health or emotional support topics.
- Refuse any unrelated request with a short message and redirect back to support.


----------------------------------------------------
STEP 3: STAGE-BASED SUPPORT
----------------------------------------------------

If MILD:
- Suggest lifestyle adjustments (sleep, exercise, journaling, hobbies)
- Offer grounding, breathing, mindfulness
- Encourage social connection
- Provide one small practical coping action


If MODERATE:
- Encourage structured routine and small achievable daily goals.
- Create a personalized 7-day or 14-day simple routine plan based on the user's emotional condition.
  (Include sleep schedule, small tasks, light physical activity, reflection/journaling, social connection, and relaxation practice.)
- Combine coping tools + structured support.
- Gently explain that if there is no noticeable improvement after consistently following the routine plan,
  suggest professional therapy (CBT, MBCT, IPT) or consulting a licensed mental health professional in India.
- Mention Indian-based professional support only if needed (psychologist, psychiatrist, or mental health hospitals in India).
- Do not diagnose or prescribe medication.
- Keep the suggestion respectful, supportive, and non-forceful.


If SEVERE:
- Encourage immediate professional help
- Suggest reaching trusted person
- Emphasize safety planning
- Stay calm, grounding, reassuring

CRISIS:
If self-harm or suicide is mentioned:
Encourage contacting Indian crisis helpline:
Tele-MANAS (National Mental Health Helpline) : 14416 / 1800-891-4416 
Mano Darpan (Students & Families) : 8448-440-632 
KIRAN: 1800-599-0019
Emergency response (Police, Ambulance) : 112 

----------------------------------------------------
STEP 4: RESPONSE STRUCTURE
----------------------------------------------------
1. Emotion-based opening (empathetic + human)
2. Reflect understanding
3. Personalized insight
4. Small actionable coping step
5. Gentle encouragement toward appropriate level of support
6. Open-ended question to continue conversation
7. Use emojis naturally
8. At the end of every(only when real emotion is there) response, add an stress intensity score (1–10) based on the user’s current message.
   - Do not explain the score.
   Format:
   ---
   Stress Intensity: X/10
   ⭐⭐⭐⭐⭐⭐☆☆☆☆
   ---
   Use exactly 10 stars.
   Filled stars (⭐) = score.
   Empty stars (☆) = remaining.
   Place this only at the end.
"""

prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

# ---------------- CRISIS DETECTION ---------------- #
def detect_crisis(text):
    crisis_keywords = [
        "suicide", "kill myself", "end my life",
        "self harm", "hurt myself", "don't want to live"
    ]
    return any(keyword in text.lower() for keyword in crisis_keywords)

# ---------------- LLM RESPONSE FUNCTION ---------------- #
def process_llm_response(user_input):

    profile = st.session_state.user_profile

    profile_context = f"""
Name: {profile.get("name", "Not provided")}
Age: {profile.get("age", "Not provided")}
Gender: {profile.get("gender", "Not provided")}
Occupation: {profile.get("occupation", "Not provided")}
Addictions: {profile.get("addictions", "Not provided")}
Stress Triggers: {profile.get("stress_triggers", "Not provided")}
Previous Treatment: {profile.get("previous_treatment", "Not provided")}
"""

    # Crisis immediate warning banner
    if detect_crisis(user_input):
        st.warning("If you're in immediate danger, please contact your local emergency number or a suicide prevention hotline.")

    llm = st.session_state.llm
    chain = prompt_template | llm

    memory_vars = st.session_state.memory.load_memory_variables({})

    response = chain.invoke({
        "input": user_input,
        "profile": profile_context,
        "chat_history": memory_vars["chat_history"],
        "context": "",
        "question": user_input
    })

    st.session_state.memory.save_context(
        {"input": user_input},
        {"output": response.content}
    )

    return response.content

# ---------------- CHAT DISPLAY ---------------- #
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            # if st.button("🔊 Read Aloud", key=f"voice_{i}"):
                audio_bytes = text_to_speech(message["content"])
                st.audio(audio_bytes, format="audio/mp3")

# ---------------- CHAT INPUT ---------------- #
if prompt := st.chat_input("Share your thoughts, feelings, or anything you'd like to talk about..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Reflecting on your words..."):
            try:
                response = process_llm_response(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
            except Exception as e:
                st.error("I'm having trouble connecting right now. Please try again.")
                print(e)
