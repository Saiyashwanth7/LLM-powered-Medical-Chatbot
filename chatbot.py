import streamlit as st
import json
import csv
import datetime
import re
from typing import Dict, List, Any, Optional
import requests
import time
# Streamlit app for an AI-powered medical assistant chatbot
# Combines OpenRouter LLM for conversation and RapidAPI for diagnosis
# Provides dynamic questioning, emergency detection, and a health assessment summary

class LLMMedicalChatbot:
    def __init__(self):
        """Initialize the LLM-powered medical chatbot."""
        self.symptoms_list = [
            'fever', 'cough', 'headache', 'sore throat', 'runny nose', 
            'body aches', 'chills', 'fatigue', 'nausea', 'vomiting',
            'diarrhea', 'chest pain', 'shortness of breath', 'dizziness',
            'loss of appetite', 'joint pain', 'muscle weakness', 'rash',
            'abdominal pain', 'back pain', 'neck stiffness', 'confusion',
            'blurred vision', 'constipation', 'insomnia', 'anxiety',
            'depression', 'weight loss', 'weight gain', 'palpitations'
        ]
        
        self.chronic_conditions = [
            'diabetes', 'hypertension', 'heart disease', 'asthma', 
            'arthritis', 'kidney disease', 'liver disease', 'cancer',
            'depression', 'anxiety', 'thyroid disorders', 'COPD',
            'high cholesterol', 'osteoporosis', 'epilepsy', 'none'
        ]
        
        # System prompt for the medical assistant
        self.system_prompt = """You are a professional medical assistant AI helping patients gather their health information for medical assessment. Your role is to:

1. Conduct a natural, conversational interview to collect essential medical information
2. Ask relevant follow-up questions based on user responses
3. Show empathy and understanding while maintaining professionalism
4. Extract key medical details: age, symptoms, pain levels, duration, medical history, medications, allergies
5. Identify emergency situations and recommend immediate medical attention when needed
6. Keep responses concise but thorough (2-3 sentences max per response)
7. Never provide definitive diagnoses - only gather information for proper medical evaluation

Essential information to collect:
- Basic demographics (age, gender if relevant)
- Current symptoms and their severity/duration
- Pain levels (1-10 scale)
- Medical history and chronic conditions
- Current medications and allergies
- Recent changes in health
- Emergency red flags

When you have sufficient information, indicate that you're ready to provide preliminary assessment by saying "ASSESSMENT_READY" at the end of your response.

Emergency symptoms requiring immediate attention:
- Chest pain, shortness of breath
- Severe headache with neck stiffness
- Loss of consciousness, confusion
- Severe bleeding, trauma
- Signs of stroke or heart attack

Always maintain a caring, professional tone and remind users this is preliminary screening, not medical diagnosis."""

        # Initialize session state variables
        self.init_session_state()

    def init_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        
        if 'extracted_data' not in st.session_state:
            st.session_state.extracted_data = {}
            
        if 'consultation_active' not in st.session_state:
            st.session_state.consultation_active = False
            
        if 'assessment_ready' not in st.session_state:
            st.session_state.assessment_ready = False

    def call_openrouter_api(self, messages: List[Dict], max_tokens: int = 500) -> Optional[str]:
        """Call OpenRouter API for LLM responses."""
        # Construct request headers for the OpenRouter endpoint
# Note: Replace 'your-repo' with your actual GitHub/app URL

        try:
            # Get API key from secrets or manual input
            api_key = st.secrets.get("OPENROUTER_API_KEY") or st.session_state.get("openrouter_api_key")
            
            if not api_key:
                st.error("OpenRouter API key not configured. Please add it in the sidebar.")
                return None

            url = "https://openrouter.ai/api/v1/chat/completions"
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/your-repo",  # Replace with your app URL
                "X-Title": "Medical Chatbot Assistant"
            }
            
            payload = {
                "model": "anthropic/claude-3.5-sonnet",  # Using Claude for medical conversations
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.3,  # Lower temperature for more consistent medical responses
                "top_p": 0.9
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                st.error(f"OpenRouter API error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            st.error("API request timed out. Please try again.")
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Unexpected error calling LLM: {str(e)}")
            return None

    def extract_medical_data(self, conversation_text: str) -> Dict:
        # Regex is used to find the first JSON-looking object in LLM response
# This helps handle cases where LLM may include extra text or formatting

        extraction_prompt = f"""
        Based on the following medical conversation, extract and structure the key medical information in JSON format.
        Extract only information that was explicitly mentioned by the user.

        Conversation:
        {conversation_text}

        Please extract the following information in this JSON format:
        {{
            "age": null,
            "gender": null,
            "symptoms": [],
            "symptom_duration": null,
            "pain_level": null,
            "chronic_conditions": [],
            "medications": [],
            "allergies": [],
            "has_fever": null,
            "emergency_symptoms": false,
            "additional_concerns": []
        }}

        Rules:
        - Only include information explicitly stated by the user
        - Use null for missing information
        - symptoms should be an array of strings
        - pain_level should be a number 1-10 or null
        - Set emergency_symptoms to true if any critical symptoms are mentioned
        - Return only valid JSON
        """

        messages = [
            {"role": "system", "content": "You are a medical data extraction assistant. Extract information accurately and return only valid JSON."},
            {"role": "user", "content": extraction_prompt}
        ]

        try:
            response = self.call_openrouter_api(messages, max_tokens=800)
            if response:
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            return {}
        except (json.JSONDecodeError, Exception) as e:
            st.error(f"Error extracting medical data: {str(e)}")
            return {}

    def get_llm_response(self, user_message: str) -> str:
        """Get LLM response for conversational medical interview."""
        # Build conversation context
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add recent conversation history (last 10 messages to stay within token limits)
        recent_history = st.session_state.conversation_history[-10:]
        for entry in recent_history:
            role = "assistant" if entry['type'] == 'bot' else "user"
            messages.append({"role": role, "content": entry['message']})
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        return self.call_openrouter_api(messages)

    def get_ai_diagnosis(self, extracted_data: Dict) -> Dict:
        """Get AI diagnosis using RapidAPI Medical Diagnosis API."""
        try:
            symptoms = extracted_data.get('symptoms', [])
            if not symptoms:
                return None
                
            url = "https://ai-medical-diagnosis-api-symptoms-to-results.p.rapidapi.com/api/diagnosis"
            
            headers = {
                "X-RapidAPI-Key": st.secrets.get("RAPIDAPI_KEY") or st.session_state.get("rapidapi_key", ""),
                "X-RapidAPI-Host": "ai-medical-diagnosis-api-symptoms-to-results.p.rapidapi.com",
                "Content-Type": "application/json"
            }
            
            payload = {
                "symptoms": symptoms,
                "age": extracted_data.get('age', 25),
                "gender": extracted_data.get('gender', 'unknown'),
                "medical_history": extracted_data.get('chronic_conditions', [])
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except Exception as e:
            st.error(f"Diagnosis API error: {str(e)}")
            return None

    def generate_comprehensive_assessment(self, extracted_data: Dict) -> str:
        """Generate comprehensive medical assessment using LLM."""
        # First try to get AI diagnosis
        ai_diagnosis = self.get_ai_diagnosis(extracted_data)
        
        assessment_prompt = f"""
        Based on the following patient information, provide a comprehensive medical assessment and recommendations.
        
        Patient Data: {json.dumps(extracted_data, indent=2)}
        
        AI Diagnosis Results: {json.dumps(ai_diagnosis, indent=2) if ai_diagnosis else "Not available"}
        
        Please provide:
        1. Summary of presented symptoms and concerns
        2. Possible conditions or differential diagnoses (if AI diagnosis available, incorporate those findings)
        3. Urgency level (Low/Moderate/High/Emergency)
        4. Specific recommendations for next steps
        5. General health advice
        6. When to seek immediate medical attention
        
        Format your response with clear sections using markdown headers.
        Be thorough but concise, and always emphasize that this is preliminary assessment requiring professional medical evaluation.
        """

        messages = [
            {"role": "system", "content": "You are a medical assessment AI providing preliminary health evaluations. Always emphasize the need for professional medical consultation."},
            {"role": "user", "content": assessment_prompt}
        ]
        # If diagnosis is unavailable, LLM will still generate an advisory report
        # Clearly states urgency level and next steps, but never gives a final diagnosis

        try:
            assessment = self.call_openrouter_api(messages, max_tokens=1200)
            return assessment or "Unable to generate assessment. Please consult a healthcare provider."
        except Exception as e:
            return f"Error generating assessment: {str(e)}. Please consult a healthcare provider."

    def save_conversation(self, format_type='json'):
        """Save conversation history and extracted data."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type == 'json':
            filename = f"medical_consultation_{timestamp}.json"
            data = {
                'consultation_info': {
                    'timestamp': timestamp,
                    'session_id': f"session_{timestamp}",
                    'consultation_type': 'AI Medical Assistant',
                    'status': 'completed' if st.session_state.assessment_ready else 'in_progress'
                },
                'extracted_medical_data': st.session_state.extracted_data,
                'full_conversation': st.session_state.conversation_history,
                'conversation_summary': {
                    'total_messages': len(st.session_state.conversation_history),
                    'user_messages': len([m for m in st.session_state.conversation_history if m['type'] == 'user']),
                    'bot_messages': len([m for m in st.session_state.conversation_history if m['type'] == 'bot']),
                    'duration_minutes': self._calculate_session_duration()
                }
            }
            
            json_str = json.dumps(data, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Download Full Consultation (JSON)",
                data=json_str,
                file_name=filename,
                mime="application/json",
                help="Complete consultation data in JSON format - ideal for healthcare providers"
            )
        
        elif format_type == 'csv':
            filename = f"medical_consultation_summary_{timestamp}.csv"
            
            # Create comprehensive CSV data
            csv_data = []
            
            # Header information
            csv_data.extend([
                ['MEDICAL CONSULTATION SUMMARY', ''],
                ['Generated On', timestamp],
                ['Session ID', f"session_{timestamp}"],
                ['Status', 'completed' if st.session_state.assessment_ready else 'in_progress'],
                ['', ''],
                ['EXTRACTED MEDICAL DATA', '']
            ])
            
            # Add extracted medical data
            if st.session_state.extracted_data:
                for key, value in st.session_state.extracted_data.items():
                    if isinstance(value, list):
                        if value:  # Only add if list is not empty
                            csv_data.append([key.replace('_', ' ').title(), '; '.join(str(v) for v in value)])
                        else:
                            csv_data.append([key.replace('_', ' ').title(), 'None reported'])
                    elif value is not None:
                        csv_data.append([key.replace('_', ' ').title(), str(value)])
                    else:
                        csv_data.append([key.replace('_', ' ').title(), 'Not specified'])
            
            # Add conversation log
            csv_data.extend([
                ['', ''],
                ['CONVERSATION LOG', ''],
                ['Speaker', 'Message', 'Timestamp']
            ])
            
            for entry in st.session_state.conversation_history:
                speaker = 'AI Assistant' if entry['type'] == 'bot' else 'Patient'
                message = entry['message'].replace('\n', ' ').replace('\r', ' ')  # Clean message for CSV
                timestamp_entry = entry.get('timestamp', 'N/A')
                csv_data.append([speaker, message, timestamp_entry])
            
            # Convert to CSV string
            import io
            output = io.StringIO()
            writer = csv.writer(output, quoting=csv.QUOTE_ALL)
            writer.writerows(csv_data)
            csv_str = output.getvalue()
            
            st.download_button(
                label="📊 Download Summary Report (CSV)",
                data=csv_str,
                file_name=filename,
                mime="text/csv",
                help="Structured medical data summary in CSV format - easy to import into spreadsheets"
            )

    def _calculate_session_duration(self) -> float:
        """Calculate session duration in minutes."""
        try:
            if len(st.session_state.conversation_history) >= 2:
                first_msg = st.session_state.conversation_history[0]
                last_msg = st.session_state.conversation_history[-1]
                
                if 'timestamp' in first_msg and 'timestamp' in last_msg:
                    start_time = datetime.datetime.fromisoformat(first_msg['timestamp'])
                    end_time = datetime.datetime.fromisoformat(last_msg['timestamp'])
                    duration = (end_time - start_time).total_seconds() / 60
                    return round(duration, 2)
            return 0.0
        except:
            return 0.0

def main():
    # Set up the Streamlit UI, including the sidebar, chat display, and control buttons
    # Handles user input, stores messages, triggers LLM response, and shows assessment

    st.set_page_config(
        page_title="AI Medical Health Assistant",
        page_icon="🤖🏥",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        margin-bottom: 1rem;
        background-color: #fafafa;
    }
    .user-message {
        background: linear-gradient(135deg, #2196f3 0%, #21cbf3 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        margin-left: 20%;
    }
    .bot-message {
        background: white;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        margin-right: 20%;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .assessment-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .emergency-alert {
        background: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
                color: #333;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize chatbot
    chatbot = LLMMedicalChatbot()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI-Powered Medical Health Assistant</h1>
        <p>Natural conversation powered by advanced language models</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 API Configuration")
        
        # OpenRouter API Key
        st.subheader("OpenRouter LLM API")
        openrouter_configured = 'OPENROUTER_API_KEY' in st.secrets
        
        if openrouter_configured:
            st.success("✅ OpenRouter API Key Configured")
        else:
            st.warning("⚠️ OpenRouter API Key Required")
            api_key = st.text_input("Enter OpenRouter API Key:", type="password", key="openrouter_key_input")
            if api_key:
                st.session_state.openrouter_api_key = api_key
                st.success("✅ API Key Set for Session")
        
        # Medical Diagnosis API
        st.subheader("Medical Diagnosis API")
        rapidapi_configured = 'RAPIDAPI_KEY' in st.secrets or 'rapidapi_key' in st.session_state
        
        if rapidapi_configured:
            st.success("✅ RapidAPI Key Configured")
        else:
            st.info("ℹ️ Optional: Enhances diagnosis accuracy")
            rapid_key = st.text_input("Enter RapidAPI Key (Optional):", type="password", key="rapid_key_input")
            if rapid_key:
                st.session_state.rapidapi_key = rapid_key
                st.success("✅ RapidAPI Key Set")
        
        with st.expander("📖 Setup Instructions"):
            st.markdown("""
            **OpenRouter Setup:**
            1. Visit [OpenRouter.ai](https://openrouter.ai)
            2. Sign up and get API key
            3. Add credits to your account
            4. Enter key above or in secrets.toml
            
            **RapidAPI Setup (Optional):**
            1. Visit [RapidAPI](https://rapidapi.com)
            2. Subscribe to AI Medical Diagnosis API
            3. Copy your API key
            4. Enter key above
            """)
        
        st.header("📋 Features")
        st.markdown("""
        ✅ **Natural LLM Conversation**  
        ✅ **Dynamic Question Flow**  
        ✅ **Intelligent Data Extraction**  
        ✅ **Emergency Detection**  
        ✅ **Comprehensive Assessment**  
        ✅ **Professional Medical Format**
        """)
        
        # Export options
        if st.session_state.conversation_history:
            st.header("💾 Export Consultation")
            st.markdown("Save your consultation for your records or to share with healthcare providers.")
            
            # Show data preview
            with st.expander("📋 Preview Export Data"):
                if st.session_state.extracted_data:
                    st.json(st.session_state.extracted_data)
                else:
                    st.info("Complete the consultation to see extracted medical data")
            
            chatbot.save_conversation('json')
            chatbot.save_conversation('csv')
            
            # Additional export info
            st.caption("💡 JSON format preserves full conversation structure. CSV format provides tabular data summary.")
        
        st.header("⚠️ Medical Disclaimer")
        st.error("""
        This AI assistant provides preliminary health information only. 
        Always consult qualified healthcare professionals for medical advice, 
        diagnosis, and treatment.
        """)

    # Main chat interface
    if not st.session_state.consultation_active:
        st.markdown("### Welcome to Your AI Health Assistant! 👋")
        
        st.markdown("""
        <div class="info-card">
        <h4>🤖 How This Works:</h4>
        <ul>
            <li><strong>Natural Conversation:</strong> Chat naturally about your health concerns</li>
            <li><strong>Intelligent Questions:</strong> AI asks relevant follow-up questions</li>
            <li><strong>Smart Analysis:</strong> Extracts and organizes your medical information</li>
            <li><strong>Professional Assessment:</strong> Provides preliminary health evaluation</li>
        </ul>
        
        <h4>🏥 What We'll Discuss:</h4>
        <ul>
            <li>Your current symptoms and concerns</li>
            <li>Medical history and chronic conditions</li>
            <li>Pain levels and symptom duration</li>
            <li>Medications and allergies</li>
            <li>Recent health changes</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start AI Health Consultation", type="primary", use_container_width=True):
                # Check if API key is available
                if not (st.secrets.get("OPENROUTER_API_KEY") or st.session_state.get("openrouter_api_key")):
                    st.error("Please configure your OpenRouter API key in the sidebar first.")
                else:
                    st.session_state.consultation_active = True
                    initial_message = "Hello! I'm your AI health assistant. I'm here to help gather information about your health concerns in a natural, conversational way. What brings you here today? Are you experiencing any symptoms or health issues you'd like to discuss?"
                    
                    st.session_state.conversation_history.append({
                        'type': 'bot',
                        'message': initial_message,
                        'timestamp': datetime.datetime.now().isoformat()
                    })
                    st.rerun()
    
    else:
        # Active consultation interface
        st.markdown("### 💬 Health Consultation in Progress")
        
        # Display conversation history
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for entry in st.session_state.conversation_history:
            if entry['type'] == 'bot':
                st.markdown(f"""
                <div class="bot-message">
                    <strong>🤖 Health Assistant:</strong><br>
                    {entry['message']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="user-message">
                    <strong>You:</strong><br>
                    {entry['message']}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # User input area
        if not st.session_state.assessment_ready:
            with st.form("chat_form", clear_on_submit=True):
                user_input = st.text_area(
                    "Your message:",
                    height=100,
                    placeholder="Describe your symptoms, answer questions, or share any health concerns..."
                )
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    submitted = st.form_submit_button("Send Message", type="primary", use_container_width=True)
                with col2:
                    emergency_btn = st.form_submit_button("🚨 Emergency", help="Click if you're experiencing a medical emergency")
                
                if emergency_btn:
                    st.error("""
                    🚨 **MEDICAL EMERGENCY**
                    
                    If you're experiencing a medical emergency, please:
                    - Call 911 (US) or your local emergency number
                    - Go to the nearest emergency room
                    - Contact emergency services immediately
                    
                    Do not rely on this chatbot for emergency medical situations.
                    """)
                
                if submitted and user_input.strip():
                    # Add user message to history
                    st.session_state.conversation_history.append({
                        'type': 'user',
                        'message': user_input,
                        'timestamp': datetime.datetime.now().isoformat()
                    })
                    
                    # Get LLM response
                    with st.spinner("🤖 AI is thinking..."):
                        bot_response = chatbot.get_llm_response(user_input)
                        
                        if bot_response:
                            # Check if assessment is ready
                            if "ASSESSMENT_READY" in bot_response:
                                bot_response = bot_response.replace("ASSESSMENT_READY", "").strip()
                                st.session_state.assessment_ready = True
                            
                            st.session_state.conversation_history.append({
                                'type': 'bot',
                                'message': bot_response,
                                'timestamp': datetime.datetime.now().isoformat()
                            })
                        else:
                            st.session_state.conversation_history.append({
                                'type': 'bot',
                                'message': "I apologize, but I'm having trouble processing your message right now. Could you please try again?",
                                'timestamp': datetime.datetime.now().isoformat()
                            })
                    
                    st.rerun()
        
        # Assessment generation
        if st.session_state.assessment_ready:
            st.markdown("---")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("### 🩺 Ready for Medical Assessment")
                st.info("I've gathered enough information to provide you with a preliminary health assessment.")
            
            with col2:
                if st.button("📋 Generate Assessment", type="primary", use_container_width=True):
                    with st.spinner("🔍 Analyzing your health information and generating comprehensive assessment..."):
                        # Extract structured data from conversation
                        conversation_text = "\n".join([
                            f"{'Assistant' if entry['type'] == 'bot' else 'User'}: {entry['message']}"
                            for entry in st.session_state.conversation_history
                        ])
                        
                        extracted_data = chatbot.extract_medical_data(conversation_text)
                        st.session_state.extracted_data = extracted_data
                        
                        # Generate comprehensive assessment
                        assessment = chatbot.generate_comprehensive_assessment(extracted_data)
                        
                        # Display assessment
                        st.markdown("---")
                        
                        st.markdown("""
                        <div class="assessment-box">
                            <h2>🩺 Medical Assessment Report</h2>
                            <p>Based on our conversation, here's your preliminary health evaluation:</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(assessment)
                        
                        # Display extracted data summary
                        st.markdown("### 📊 Consultation Summary")
                        
                        if extracted_data:
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown("**👤 Patient Information**")
                                info_data = {
                                    "Age": extracted_data.get('age', 'Not specified'),
                                    "Gender": extracted_data.get('gender', 'Not specified'),
                                    "Pain Level": f"{extracted_data.get('pain_level', 'Not specified')}/10" if extracted_data.get('pain_level') else 'Not specified'
                                }
                                for key, value in info_data.items():
                                    st.write(f"**{key}:** {value}")
                            
                            with col2:
                                st.markdown("**🔍 Symptoms**")
                                symptoms = extracted_data.get('symptoms', [])
                                if symptoms:
                                    for symptom in symptoms:
                                        st.write(f"• {symptom}")
                                else:
                                    st.write("No specific symptoms documented")
                            
                            with col3:
                                st.markdown("**🏥 Medical History**")
                                conditions = extracted_data.get('chronic_conditions', [])
                                medications = extracted_data.get('medications', [])
                                
                                if conditions:
                                    st.write("**Conditions:**")
                                    for condition in conditions:
                                        st.write(f"• {condition}")
                                
                                if medications:
                                    st.write("**Medications:**")
                                    for med in medications:
                                        st.write(f"• {med}")
                                
                                if not conditions and not medications:
                                    st.write("No medical history documented")
                        
                        # Final disclaimer
                        st.markdown("""
                        <div class="emergency-alert">
                            <h4>⚠️ Important Medical Disclaimer</h4>
                            <p>This assessment is for informational purposes only and does not constitute medical advice. 
                            Please consult with a qualified healthcare provider for proper medical diagnosis and treatment.</p>
                            
                            <p><strong>Seek immediate medical attention if you experience:</strong></p>
                            <ul>
                                <li>Chest pain or difficulty breathing</li>
                                <li>Severe headache with neck stiffness</li>
                                <li>Loss of consciousness or severe confusion</li>
                                <li>Signs of stroke or heart attack</li>
                                <li>Severe bleeding or trauma</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Reset option
                        st.markdown("---")
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            if st.button("🔄 Start New Consultation", use_container_width=True):
                                # Clear session state
                                keys_to_clear = ['conversation_history', 'extracted_data', 'consultation_active', 'assessment_ready']
                                for key in keys_to_clear:
                                    if key in st.session_state:
                                        del st.session_state[key]
                                st.success("Starting new consultation...")
                                time.sleep(1)
                                st.rerun()

if __name__ == "__main__":
    main()
