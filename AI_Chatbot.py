import streamlit as st
import tempfile
import base64
import re
from gtts import gTTS
from gtts.tts import gTTSError
from langchain_groq import ChatGroq
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_classic.memory import ConversationBufferWindowMemory

# ---------------- CONFIG & CONSTANTS ---------------- #
MODEL_NAME = "openai/gpt-oss-120b"
TEMPERATURE = 0.7
BG_IMAGE_PATH = "background .jpeg"  # Ensure this file exists in your directory
LOGO_PATH = "emoji.png"
PAGE_ICON="aura_robot_icon_64.png"

# ---------------- HELPER FUNCTIONS ---------------- #
def get_base64_bin(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_bg_and_style():
    try:
        bin_str = get_base64_bin(BG_IMAGE_PATH)
        bg_css = f"""
        <style>
        /* Base background setup */
        .stApp {{
            background-image: url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Force Black Text globally on all text-bearing elements */
        html, body, [class*="st-"], .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
            color: #fcfcfc !important;
        }}

        /* Fix Chat Container Contrast */
        [data-testid="stChatMessage"] {{
            background-color: #0f0202 !important; /* Higher opacity for legibility */
            border: 1px solid rgba(0, 0, 0, 0.1);
            border-radius: 15px;
            padding: 15px;
            margin-bottom: 10px;
        }}

        /* Fix Chat Avatars/Icons alignment and visibility */
        [data-testid="stChatMessageContent"] {{
            color: #000000 !important;
        }}
        
        
        /* Main Chat Input Bar */
        .stChatInputContainer textarea {{
            background-color: #ffffff !important;
            color: #000000 !important;
            border-radius: 10px !important;
        }}

        /* Button consistency */
        .stButton>button {{
            background-color: #0f0202 !important;
            color: #ffffff !important;
            border-radius: 8px;
            border: none;
            transition: 0.2s ease;
        }}
        
        .stButton>button:hover {{
            background-color: #333333 !important;
            transform: scale(1.02);
        }}

         /* Hide Streamlit branding for a cleaner look */
        #MainMenu, footer, header,sidebar {{visibility: hidden;}}
        </style>
        """
        st.markdown(bg_css, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("Background image not found. Ensure background.jpg is in the directory.")



def clean_text_for_tts(text):
    text = text.split("---")[0] if "---" in text else text
    text = re.sub(r'[⭐☆*#\-—_`]', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def text_to_speech(text):
    cleaned_text = clean_text_for_tts(text)[:500]
    try:
        tts = gTTS(text=cleaned_text, lang="en")
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        return temp_file.name
    except gTTSError:
        return None

# ---------------- INITIALIZATION ---------------- #
st.set_page_config(page_title="AuraAssist | AI Support", layout="wide",initial_sidebar_state="collapsed",page_icon=PAGE_ICON)

set_bg_and_style()

if "page" not in st.session_state:
    st.session_state.page = "Home"
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=6, return_messages=True, memory_key="chat_history")

# ---------------- PAGE 1: HOME ---------------- #
if st.session_state.page == "Home":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(LOGO_PATH, width=150)
        st.title("Welcome to AuraAssist")
        st.markdown("""
        ### Your Intelligent Emotional Companion
        AuraAssist leverages Large Language Models and Sentiment Analysis to provide a supportive space for reflection. 
        Whether you are managing daily stress or seeking a moment of mindfulness, our AI is here to listen.
        
        **Key Features:**
        - 🧠 **Contextual Understanding**: Personalized support based on your profile.
        - 🎙️ **Voice Synthesis**: Listen to responses for a more human experience.
        - 🛡️ **Crisis Detection**: Safety-first architecture with emergency resources.
        """)
        
        if st.button("Get Started →"):
            st.session_state.page = "Profile"
            st.rerun()

# ---------------- PAGE 2: PROFILE FORM ---------------- #
elif st.session_state.page == "Profile":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.subheader("👤 Personalize Your Experience")
        st.info("Your data helps the AI tailor its supportive techniques to your specific lifestyle.")
        
        with st.form("profile_form"):
            api_key = st.text_input("Groq API Key", type="password")
            name = st.text_input("Full Name")
            age = st.number_input("Age", 10, 100, 25)
            occupation = st.text_input("Occupation")
            triggers = st.text_area("What are your main stress triggers?")
            addictions = st.text_input("Addictions or habits you're managing (optional)")
            
            submit = st.form_submit_button("Initialize AuraAssist")
            
            if submit:
                if not api_key:
                    st.error("API Key is required to proceed.")
                else:
                    st.session_state.user_profile = {
                        "name": name, "age": age, "occupation": occupation,
                        "stress_triggers": triggers, "addictions": addictions
                    }
                    st.session_state.api_key = api_key
                    # Init LLM
                    st.session_state.llm = ChatGroq(
                        groq_api_key=api_key,
                        model_name=MODEL_NAME,
                        temperature=TEMPERATURE
                    )
                    st.session_state.page = "Chat"
                    st.rerun()

# ---------------- PAGE 3: CHAT INTERFACE ---------------- #
elif st.session_state.page == "Chat":
    st.sidebar.title("AuraAssist")
    st.sidebar.write(f"Hello, {st.session_state.user_profile.get('name', 'Friend')}!")
    if st.sidebar.button("Reset Session"):
        st.session_state.clear()
        st.rerun()
    
    # Prompt Setup (reusing your logic)
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

    # Display Messages
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if st.button("🔊 Play", key=f"audio_{i}"):
                    path = text_to_speech(msg["content"])
                    if path: st.audio(path)

    # Input logic
    if prompt := st.chat_input("How are you feeling today?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                # LLM Chain execution
                chain = prompt_template | st.session_state.llm
                mem = st.session_state.memory.load_memory_variables({})
                
                res = chain.invoke({
                    "input": prompt,
                    "profile": str(st.session_state.user_profile),
                    "chat_history": mem["chat_history"],
                    "context": "",
                    "question": prompt
                })
                
                st.markdown(res.content)
                st.session_state.messages.append({"role": "assistant", "content": res.content})
                st.session_state.memory.save_context({"input": prompt}, {"output": res.content})
                st.rerun()
