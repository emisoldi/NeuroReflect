# Recursive AI Logic Implementation: Stage 5 (Integration with Front-End & Interface Optimization using LangChain and Llama3.2)

# === Import Essential Libraries ===
import os
import json
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.base import LLM
import streamlit as st
from dotenv import load_dotenv
from ollama import chat, ChatResponse

# === Load Environment Variables ===
load_dotenv()

# === Load NeuroReflect Circuits from JSON ===
with open("NeuroReflect_Circuit_Logic_Final_Fixed.json", "r") as file:
    circuit_data = json.load(file)

# Load comprehensive circuit data for enhanced analysis
with open("NeuroReflect_Circuit_Logic.json", "r") as file:
    comprehensive_circuit_data = json.load(file)
    # Clean up NaN entries
    comprehensive_circuit_data = [circuit for circuit in comprehensive_circuit_data if isinstance(circuit.get("circuit"), str)]

# Extract circuit names properly
circuits = [circuit["name"] for circuit in circuit_data if circuit.get("name")]
comprehensive_circuits = [circuit["circuit"] for circuit in comprehensive_circuit_data if circuit.get("circuit")]

archetypes = ["Hero", "Mentor", "Explorer", "Creator", "Healer"]

# Custom LLM class for Ollama
class OllamaLLM(LLM):
    def _call(self, prompt: str, stop: None = None) -> str:
        response: ChatResponse = chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"].strip()

    @property
    def _llm_type(self) -> str:
        return "ollama"

# Initialize LangChain LLM with Ollama
llm = OllamaLLM()

# === Deterministic Input Parsing ===
class InputParser:
    def __init__(self, circuits, archetypes, circuit_details, comprehensive_circuit_data=None):
        self.circuits = circuits
        self.archetypes = archetypes
        self.circuit_details = circuit_details
        self.comprehensive_circuit_data = comprehensive_circuit_data or []

        # LangChain prompt for sentiment analysis
        self.sentiment_prompt = PromptTemplate(
            input_variables=["text"],
            template=(
                "You are Ahura Mazda, a specialized assistant for NeuroReflect. Your role is to classify the emotional tone "
                "of the following statement as either 'positive', 'neutral', or 'negative'. Only respond with the exact word: "
                "positive, neutral, or negative. Statement: {text}"
            ),
        )
        self.sentiment_chain = LLMChain(llm=llm, prompt=self.sentiment_prompt)

    def parse(self, input_text):
        sentiment = self.analyze_sentiment(input_text)
        symbol = self.symbolic_mapping(input_text)
        circuit = self.match_circuit(input_text, sentiment, symbol)
        return {"sentiment": sentiment, "symbol": symbol, "circuit": circuit}

    def analyze_sentiment(self, text):
        # Update to use invoke instead of run
        result = self.sentiment_chain.invoke({"text": text})["text"].strip().lower()
        if result in ["positive", "neutral", "negative"]:
            return result
        return "neutral"

    def symbolic_mapping(self, text):
        for archetype in self.archetypes:
            if archetype.lower() in text.lower():
                return archetype
        return "Neutral"

    def match_circuit(self, input_text, sentiment, symbol):
        # First try to match using the simplified circuit data
        for circuit in self.circuit_details:
            symptoms = " ".join(circuit.get("symptoms_of_dysregulation", [])).lower()
            traits = " ".join(circuit.get("traits_influenced", [])).lower()
            if sentiment.lower() in symptoms or symbol.lower() in traits:
                return circuit.get("name", "Default Mode Network (Self-Referential Thought & Internal Narrative)")
        
        # If no match in simplified data, try comprehensive data for better matching
        if self.comprehensive_circuit_data:
            best_match = None
            max_score = 0
            
            # Look for keyword matches in the comprehensive data
            for circuit in self.comprehensive_circuit_data:
                score = 0
                circuit_name = circuit.get("circuit", "")
                
                # Check symptoms for matches
                symptoms = [s.lower() for s in circuit.get("symptoms", [])]
                symptoms_text = " ".join(symptoms).lower()
                
                # Check user statements for matches
                example_statements = [s.lower() for s in circuit.get("example_user_statements", [])]
                statements_text = " ".join(example_statements).lower()
                
                # Check if input text contains keywords from this circuit
                for keyword in input_text.lower().split():
                    if keyword in symptoms_text or keyword in statements_text:
                        score += 1
                        
                # Check sentiment alignment
                if sentiment.lower() in symptoms_text:
                    score += 2
                
                # If this is the best match so far, remember it
                if score > max_score:
                    max_score = score
                    best_match = circuit_name
            
            if best_match and max_score > 0:
                # Map the comprehensive circuit to a simplified one if possible
                for simple_circuit in self.circuit_details:
                    if simple_circuit.get("name", "").lower() in best_match.lower():
                        return simple_circuit.get("name")
                
                # If no mapping exists, return the comprehensive circuit name
                return best_match
        
        # Default fallback
        return "Default Mode Network (Self-Referential Thought & Internal Narrative)"

# === Deterministic Logic Engine ===
class LogicEngine:
    def __init__(self, parser):
        self.parser = parser
        self.memory = []

        # LangChain prompt for response generation
        self.response_prompt = PromptTemplate(
            input_variables=["user_input", "sentiment", "symbol", "circuit", "circuit_summary", "flow_state"],
            template=(
                "You are Ahura Mazda, a NeuroReflect assistant. Based on the user's current state and input, respond with a helpful, reflective message.\n"
                "User Input: {user_input}\n"
                "State Information:\n"
                "- Sentiment: {sentiment}\n"
                "- Archetype: {symbol}\n"
                "- Circuit: {circuit}\n"
                "- Circuit Description: {circuit_summary}\n"
                "- Flow State: {flow_state}\n"
                "Use wisdom rooted in determinism, energy flow, neuroplasticity, and spiritual integration.\n"
                "Ask a follow-up question to keep the conversation flowing."
            ),
        )
        self.response_chain = LLMChain(llm=llm, prompt=self.response_prompt)

    def process_input(self, input_text):
        parsed_data = self.parser.parse(input_text)
        parsed_data["user_input"] = input_text  # Add user input to parsed data
        self.memory.append(parsed_data)
        response = self.generate_response(parsed_data)
        flow_feedback = self.evaluate_flow()
        self.refine_logic(parsed_data, flow_feedback)
        return response, parsed_data, flow_feedback

    def generate_response(self, data):
        circuit = data["circuit"]
        sentiment = data["sentiment"]
        symbol = data["symbol"]
        flow_state = self.evaluate_flow()

        # Get circuit description if available from simplified data
        circuit_info = next((c for c in circuit_data if c.get("name") == circuit), None)
        
        # If not found in simplified data, try comprehensive data
        if not circuit_info:
            circuit_info = next((c for c in comprehensive_circuit_data if c.get("circuit") == circuit), {})
            circuit_summary = circuit_info.get("metaphoric_summary", "an undefined pattern")
            
            # Extract additional context from comprehensive data if available
            additional_context = ""
            if circuit_info:
                if circuit_info.get("functional_description"):
                    additional_context += f"\nNeural Basis: {circuit_info.get('functional_description')}"
                if circuit_info.get("symptoms"):
                    additional_context += f"\nCommon Symptoms: {', '.join(circuit_info.get('symptoms'))}"
                if circuit_info.get("recommended_interventions"):
                    additional_context += f"\nPotential Interventions: {circuit_info.get('recommended_interventions')}"
        else:
            circuit_summary = circuit_info.get("metaphoric_summary", "an undefined pattern")
            additional_context = ""

        # Generate response using LangChain with invoke instead of run
        response = self.response_chain.invoke({
            "user_input": data["user_input"],
            "sentiment": sentiment,
            "symbol": symbol,
            "circuit": circuit,
            "circuit_summary": circuit_summary + additional_context,
            "flow_state": flow_state,
        })["text"].strip()
        
        return response

    def evaluate_flow(self):
        if len(self.memory) < 2:
            return "Flow stable: Let's keep this energy."
        recent_states = [m["circuit"] for m in self.memory[-3:]]
        if any(state in recent_states for state in ["Stress", "Anxiety", "Fear & Stress Response"]):
            return "Flow disrupted: Let's recalibrate your state gently."
        elif any(state in recent_states for state in ["Joy", "Clarity", "Play & Social Joy Circuit"]):
            return "Flow optimal: Continue your momentum!"
        else:
            return "Flow stable: Keep maintaining this balanced state."

    def refine_logic(self, parsed_data, flow_feedback):
        refinement_summary = {
            "recent_input": parsed_data,
            "flow_feedback": flow_feedback
        }
        file_path = 'refinement_log.json'

        # Check and initialize the file if not present
        if not os.path.exists(file_path):
            with open(file_path, 'w') as file:
                file.write('')

        with open(file_path, 'a') as file:
            json.dump(refinement_summary, file)
            file.write('\n')

# === Streamlit Frontend Integration ===
st.title("Recursive AI: NeuroReflect Chatbot")
st.markdown("Welcome to the NeuroReflect Chatbot! Share your thoughts, and let the AI assist you with reflective insights.")

# Initialize session state for chatbot memory and follow-up
if 'engine' not in st.session_state:
    st.session_state.parser = InputParser(circuits, archetypes, circuit_data, comprehensive_circuit_data)
    st.session_state.engine = LogicEngine(st.session_state.parser)
    st.session_state.chat_history = []  # Store conversation history
    st.session_state.thinking = False  # Flag to indicate if the model is "thinking"

# Add custom CSS for chat bubbles and layout
st.markdown("""
<style>
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-bottom: 20px;
}
.user-message-container {
    display: flex;
    justify-content: flex-end;
    align-items: center;
    gap: 10px;
}
.ai-message-container {
    display: flex;
    justify-content: flex-start;
    align-items: center;
    gap: 10px;
}
.user-avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background-color: #4CAF50;
    display: flex;
    justify-content: center;
    align-items: center;
    color: white;
    font-weight: bold;
}
.ai-avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background-color: #2196F3;
    display: flex;
    justify-content: center;
    align-items: center;
    color: white;
    font-weight: bold;
}
.user-message {
    background-color: #E8F5E9;
    padding: 10px 15px;
    border-radius: 18px 18px 0 18px;
    max-width: 70%;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    color: black;
    border: 1px solid #C8E6C9;
}
.ai-message {
    background-color: #E3F2FD;
    padding: 10px 15px;
    border-radius: 18px 18px 18px 0;
    max-width: 70%;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    color: black;
    border: 1px solid #BBDEFB;
}
</style>
""", unsafe_allow_html=True)

# Display chat history in a conversational format with positioning and avatars
st.markdown("### Chat History")
chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    for message in st.session_state.chat_history:
        role, content = message
        if role == "user":
            st.markdown(f"""
            <div class="user-message-container">
                <div class="user-message">{content}</div>
                <div class="user-avatar">👤</div>
            </div>
            """, unsafe_allow_html=True)
        elif role == "ai":
            st.markdown(f"""
            <div class="ai-message-container">
                <div class="ai-avatar">🤖</div>
                <div class="ai-message">{content}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Show thinking indicator if the model is processing
    if st.session_state.get('thinking', False):
        st.markdown("""
        <div class="ai-message-container">
            <div class="ai-avatar">🤖</div>
            <div class="ai-message"><em>thinking...</em></div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Fixed input bar for user messages
input_container = st.container()
with input_container:
    # Use a form to handle the input
    with st.form(key="message_form", clear_on_submit=True):  # Clear form after submission
        user_input = st.text_input("Type your message here:", key="input_field")
        submit_button = st.form_submit_button("Send")
        
        if submit_button and user_input.strip():
            # Add user message to chat history
            st.session_state.chat_history.append(("user", user_input))
            
            # Set thinking flag to true to show the thinking indicator
            st.session_state.thinking = True
            
            # Force a rerun to update UI and show the thinking indicator
            st.rerun()

# Process the model response outside the form to avoid resubmission issues
if st.session_state.get('thinking', False):
    # Get the last user message
    last_user_message = next((msg[1] for msg in reversed(st.session_state.chat_history) 
                             if msg[0] == "user"), None)
    
    if last_user_message:
        # Process the input and get AI response
        response, _, flow_feedback = st.session_state.engine.process_input(last_user_message)
        
        # Add AI response to chat history
        st.session_state.chat_history.append(("ai", response))
        
        # For debugging: Log the flow feedback to console
        print(f"Flow Feedback: {flow_feedback}")
        
        # Turn off thinking indicator
        st.session_state.thinking = False
        
        # Force a rerun to update the chat display
        st.rerun()

# Allow clearing the chat history
if st.button("Clear Chat", key="clear_button"):
    st.session_state.chat_history = []
    st.session_state.thinking = False
    st.rerun()



