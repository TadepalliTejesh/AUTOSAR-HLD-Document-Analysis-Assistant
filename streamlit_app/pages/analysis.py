import streamlit as st
import httpx
from streamlit_app.components.cards import render_inconsistency_card


def render_analysis_page(api_base_url: str, active_doc_id: int, active_doc_name: str):
    st.header(f"🔍 Architecture Analysis & Inconsistencies — {active_doc_name}")
    st.caption("Cross-section consistency validation, missing dependency targets, and interface integrity check.")

    # Fetch inventory & inconsistencies
    with st.spinner("Analyzing document consistency..."):
        try:
            res = httpx.get(f"{api_base_url}/documents/{active_doc_id}/inventory", timeout=20.0)
            if res.status_code != 200:
                st.error(f"Could not load analysis data: {res.text}")
                return
            data = res.json()
        except Exception as e:
            st.error(f"Connection error: {str(e)}")
            return

    summary = data.get("summary", {})
    inconsistencies = data.get("inconsistencies", [])

    st.subheader("📊 Architecture Summary Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Components", summary.get("components", 0))
    c2.metric("Interfaces", summary.get("interfaces", 0))
    c3.metric("Ports", summary.get("ports", 0))
    c4.metric("Signals", summary.get("signals", 0))
    c5.metric("Dependencies", summary.get("dependencies", 0))

    st.markdown("---")
    st.subheader("⚠️ Potential Architectural Issues & Inconsistencies")

    st.markdown(
        "> **Project Principle:** *This is an engineering-assistance system, NOT an autonomous architecture approval system. "
        "The system highlights potential discrepancies between document sections for human engineer verification.*"
    )

    if not inconsistencies:
        st.success("✓ No potential inconsistencies detected across sections in this document.")
    else:
        st.warning(f"Found {len(inconsistencies)} potential inconsistencies requiring engineering review.")

        for idx, inc in enumerate(inconsistencies):
            render_inconsistency_card(
                entity=inc["entity_name"],
                issue_type=inc["issue_type"],
                evidence1=inc["evidence_1"],
                evidence2=inc["evidence_2"],
                reason=inc["reason"],
                status=inc.get("status", "Requires Engineer Review")
            )
