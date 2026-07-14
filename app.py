import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import streamlit as st

from src.rag_chain import RAGChain
from src.auth import AuthManager

# -----------------------------------------------------
# Page Configuration
# -----------------------------------------------------

st.set_page_config(
    page_title="Real Estate AI Assistant",
    page_icon="🏠",
    layout="wide"
)


# -----------------------------------------------------
# Session State Initialization
# -----------------------------------------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"

if "user" not in st.session_state:
    st.session_state.user = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag" not in st.session_state:
    st.session_state.rag = RAGChain()
if "selected_question" not in st.session_state:
    st.session_state.selected_question = None

# -----------------------------------------------------
# Login Function
# -----------------------------------------------------

def login():

    st.title("🏠 Real Estate AI Assistant")

    st.subheader("Login")

    email = st.text_input("Email")

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button("Login", use_container_width=True):

        success, user = AuthManager.login(
            email,
            password
        )

        if success:

            st.session_state.logged_in = True

            st.session_state.user = user

            st.rerun()

        else:

            st.error("Invalid email or password.")

    st.divider()

    st.write("Don't have an account?")

    if st.button("Create Account"):

        st.session_state.auth_mode = "signup"

        st.rerun()

def signup():

    st.title("🏠 Real Estate AI Assistant")

    st.subheader("Create Account")

    name = st.text_input("Full Name")

    email = st.text_input("Email")

    password = st.text_input(
        "Password",
        type="password"
    )

    confirm = st.text_input(
        "Confirm Password",
        type="password"
    )

    with st.expander("🔒 Password Requirements", expanded=True):

        st.markdown("""
    ✅ **Length:** 8–12 characters

    ✅ **At least one uppercase letter**

    ✅ **At least one lowercase letter**

    ✅ **At least one number**

    ✅ **At least one special character (@$!%*?&)**
    """)

    if st.button(
        "Create Account",
        use_container_width=True
    ):

        if password != confirm:

            st.error("Passwords do not match.")

        else:

            success, message = AuthManager.register(
                name,
                email,
                password
            )

            if success:

                st.success(message)

                st.info(
                    "Please login using your credentials."
                )

                st.session_state.auth_mode = "login"

                st.rerun()

            else:

                st.error(message)

    st.divider()

    st.write("Already have an account?")

    if st.button("Back to Login"):

        st.session_state.auth_mode = "login"

        st.rerun()

# -----------------------------------------------------
# Sidebar
# -----------------------------------------------------

def sidebar():

    with st.sidebar:

        st.title("🏠 AI Assistant")

        st.success(
         f"Welcome, {st.session_state.user['name']}"
        )

        st.divider()

        st.markdown("### Knowledge Base")

        st.write("📄 Documents : 92")

        st.write("🧠 Embeddings : BGE Small")

        st.write("📦 Vector DB : FAISS")

        st.write("🤖 LLM : Groq")

        st.divider()

        if st.button(
            "🗑 Clear Chat",
            use_container_width=True
        ):

            st.session_state.chat_history = []

            st.rerun()

        if st.button(
            "🚪 Logout",
            use_container_width=True
        ):

            st.session_state.logged_in = False
            st.session_state.user = None
            st.session_state.auth_mode = "login"

            st.session_state.chat_history = []

            st.rerun()


# -----------------------------------------------------
# Chat UI
# -----------------------------------------------------

def chat_page():

    sidebar()

    st.title("🏠 Real Estate AI Assistant")

    st.caption(
        "Ask anything about properties, builders, "
        "pricing, payment plans, amenities, legal "
        "documents and policies."
    )

    st.divider()
    if len(st.session_state.chat_history) == 0:

        st.info(
            "👋 Welcome! Here are a few questions you can try."
        )

        questions = [
    "Tell me about Horizon Business Park",
    "Explain the payment plan",
    "Show project amenities",
    "What is the cancellation policy?",
    "What are the possession guidelines?",
    "Compare available projects",
]

        col1, col2 = st.columns(2)

        for i, question in enumerate(questions):

            if i % 2 == 0:

                with col1:

                    if st.button(
                        question,
                        key=f"suggestion_{i}",
                        use_container_width=True
                    ):

                        st.session_state.selected_question = question
                        st.rerun()

            else:

                with col2:

                    if st.button(
                        question,
                        key=f"suggestion_{i}",
                        use_container_width=True
                    ):

                        st.session_state.selected_question = question
                        st.rerun()

    # Display Previous Messages

    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if "sources" in message:

                st.markdown("##### 📄 Sources")

                for source in message["sources"]:

                    st.write(f"- {source}")


    # User Input

    user_prompt = None

    typed_prompt = st.chat_input(
        "Ask a question about properties, builders or policies..."
    )

    if typed_prompt:
        user_prompt = typed_prompt

    elif st.session_state.selected_question:
        user_prompt = st.session_state.selected_question
        st.session_state.selected_question = None


    if user_prompt:

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_prompt
            }
        )

        with st.chat_message("user"):

            st.markdown(user_prompt)

        #
        # PART 2
        # RAG RESPONSE WILL COME HERE
        #

        try:

            with st.spinner("🤖 Thinking..."):

                response = st.session_state.rag.ask(
                    user_prompt
                )

            answer = response["answer"]

            sources = response["sources"]

            with st.chat_message("assistant"):

                st.markdown(answer)

                st.markdown("##### 📄 Sources")

                for source in sources:

                    st.write(f"• {source}")

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                }
            )

        except Exception as e:

            st.error("Something went wrong while generating the response.")

            with st.expander("Technical Details"):

                st.code(str(e))


# -----------------------------------------------------
# Main
# -----------------------------------------------------

if not st.session_state.logged_in:

    if st.session_state.auth_mode == "login":

        login()

    else:

        signup()

else:

    chat_page()
