import streamlit as st
import httpx
from streamlit_app.components.cards import render_confidence_badge


def render_chat_page(api_base_url: str, active_doc_id: int, active_doc_name: str):
    st.header(f"💬 Ask the HLD — {active_doc_name}")
    st.caption("Grounded architectural Q&A powered by local embeddings and private LLM inference.")

    # Sample prompt suggestions
    st.markdown("**Quick Engineering Inquiries:**")
    col1, col2, col3 = st.columns(3)
    sample_q = None

    with col1:
        if st.button("🔌 Which components communicate with CAN?", use_container_width=True):
            sample_q = "Which software components communicate with the CAN interface?"
    with col2:
        if st.button("🔋 What signals are in BatteryStatusIf?", use_container_width=True):
            sample_q = "What signals are transmitted over the BatteryStatusIf interface?"
    with col3:
        if st.button("❌ Non-existent Component Query", use_container_width=True):
            sample_q = "What is the memory layout of the FlightNavigationSWC?"

    # Chat history session storage
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Display past interactions
    for item in st.session_state["chat_history"]:
        with st.chat_message("user"):
            st.markdown(f"**{item['question']}**")
        with st.chat_message("assistant"):
            st.markdown(item["answer"])
            
            # Confidence badge
            col_b, col_e = st.columns([1, 3])
            with col_b:
                render_confidence_badge(item.get("confidence", "Medium"), item.get("similarity_score", 0.0))
            with col_e:
                st.caption(f"⚡ Processing time: {item.get('execution_time_ms', 0)} ms")

            # Citations
            if item.get("citations"):
                st.markdown("##### 📄 Sources & Citations")
                for cit in item["citations"]:
                    st.markdown(f"- 📄 **Page {cit['page']}** | *Section: {cit['section']}* `({cit.get('source_type', 'text')})`")

            # Expandable retrieved evidence
            if item.get("retrieved_evidence"):
                with st.expander("▶ Retrieved Evidence (Top Chunks)"):
                    for idx, chunk in enumerate(item["retrieved_evidence"]):
                        st.markdown(
                            f"**Chunk #{idx + 1} — Page {chunk['page']} | Section: {chunk['section']} | Sim: {chunk['similarity']:.3f}**"
                        )
                        st.code(chunk["content"], language="text")

    # Question Input
    default_text = sample_q if sample_q else ""
    user_query = st.chat_input("Ask a question about components, interfaces, ports, signals, or dependencies...")

    # If clicked button, trigger query
    query_to_run = user_query or sample_q

    if query_to_run:
        with st.chat_message("user"):
            st.markdown(f"**{query_to_run}**")

        with st.chat_message("assistant"):
            with st.spinner("Retrieving isolated document chunks and synthesizing grounded answer..."):
                try:
                    payload = {
                        "document_id": active_doc_id,
                        "question": query_to_run
                    }
                    res = httpx.post(f"{api_base_url}/query", json=payload, timeout=90.0)

                    if res.status_code == 200:
                        data = res.json()
                        st.markdown(data["answer"])

                        col_b, col_e = st.columns([1, 3])
                        with col_b:
                            render_confidence_badge(data["confidence"], data["similarity_score"])
                        with col_e:
                            st.caption(f"⚡ Processing time: {data['execution_time_ms']} ms")

                        # Citations
                        if data.get("citations"):
                            st.markdown("##### 📄 Sources & Citations")
                            for cit in data["citations"]:
                                st.markdown(f"- 📄 **Page {cit['page']}** | *Section: {cit['section']}* `({cit.get('source_type', 'text')})`")

                        # Retrieved chunks
                        if data.get("retrieved_evidence"):
                            with st.expander("▶ Retrieved Evidence (Top Chunks)"):
                                for idx, chunk in enumerate(data["retrieved_evidence"]):
                                    st.markdown(
                                        f"**Chunk #{idx + 1} — Page {chunk['page']} | Section: {chunk['section']} | Sim: {chunk['similarity']:.3f}**"
                                    )
                                    st.code(chunk["content"], language="text")

                        # Store in history
                        st.session_state["chat_history"].append({
                            "question": query_to_run,
                            "answer": data["answer"],
                            "confidence": data["confidence"],
                            "similarity_score": data["similarity_score"],
                            "citations": data.get("citations", []),
                            "retrieved_evidence": data.get("retrieved_evidence", []),
                            "execution_time_ms": data.get("execution_time_ms", 0)
                        })
                    else:
                        st.error(f"Query failed: {res.text}")
                except Exception as e:
                    st.error(f"Error connecting to backend API: {str(e)}")
