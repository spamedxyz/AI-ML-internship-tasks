"""
Task 5: Streamlit interface for an empathetic mental health support chatbot.

Run:
    streamlit run app.py

The app uses a fine-tuned local model when task5_mental_health_chatbot/
empathetic_model exists. If the model is not available, it uses a safe local
supportive-response fallback so the demo still works without downloads.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

MODEL_PATH = Path(__file__).parent / "empathetic_model"
CRISIS_TERMS = {
    "suicide",
    "kill myself",
    "self harm",
    "self-harm",
    "end my life",
    "hurt myself",
}


def contains_crisis_language(text: str) -> bool:
    """Detect crisis language that should receive an urgent safety response."""
    lowered = text.lower()
    return any(term in lowered for term in CRISIS_TERMS)


def local_supportive_response(user_message: str) -> str:
    """Deterministic supportive fallback for offline demonstration."""
    if contains_crisis_language(user_message):
        return (
            "I'm really sorry you're feeling this much pain. You deserve immediate "
            "support right now. Please contact local emergency services or a crisis "
            "helpline, and if possible, tell a trusted person near you what is happening."
        )

    return (
        "That sounds difficult, and it makes sense that you would feel affected by it. "
        "Try to slow the moment down: take a few steady breaths, name one thing you can "
        "control right now, and consider reaching out to someone you trust. If these "
        "feelings keep returning or become overwhelming, a mental health professional "
        "can help you work through them safely."
    )


@st.cache_resource
def load_generator():
    """Load a local fine-tuned model if available; otherwise return None."""
    if not MODEL_PATH.exists():
        return None

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=90,
            do_sample=True,
            temperature=0.75,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    except Exception as exc:
        st.sidebar.warning(f"Model unavailable, using fallback: {exc}")
        return None


def generate_response(user_message: str, generator=None) -> str:
    """Generate an empathetic response with model or local fallback."""
    if generator is None or contains_crisis_language(user_message):
        return local_supportive_response(user_message)

    prompt = (
        "User message: "
        f"{user_message}\n"
        "Supportive response:"
    )
    output = generator(prompt)[0]["generated_text"]
    response = output.split("Supportive response:")[-1].split("<|endoftext|>")[0].strip()
    return response[:600] or local_supportive_response(user_message)


def main() -> None:
    st.set_page_config(page_title="Mental Health Support", page_icon="heart")
    st.title("Mental Health Support Chatbot")
    st.caption("Educational empathetic-response demo")

    st.warning(
        "This is not therapy, diagnosis, or emergency care. If you may be in danger, "
        "contact local emergency services or a crisis helpline immediately."
    )

    generator = load_generator()
    st.sidebar.info("Model: fine-tuned local model" if generator else "Model: safe offline fallback")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Share how you are feeling...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        response = generate_response(prompt, generator)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)


if __name__ == "__main__":
    main()
