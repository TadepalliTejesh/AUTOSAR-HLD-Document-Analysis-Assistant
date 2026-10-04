import json
import streamlit as st
import httpx
import pandas as pd


def render_audit_page(api_base_url: str, active_doc_id: int):
    st.header("📋 Immutable Audit Trail & Verification Logs")
    st.caption("Complete auditability of document lifecycle, engineering queries, retrieval events, and exports.")

    filter_option = st.radio(
        "Audit Scope",
        ["Current Document", "All System Events"],
        horizontal=True
    )

    endpoint = f"{api_base_url}/documents/{active_doc_id}/audit" if filter_option == "Current Document" else f"{api_base_url}/audit"

    with st.spinner("Fetching audit log..."):
        try:
            res = httpx.get(endpoint, timeout=15.0)
            if res.status_code == 200:
                events = res.json()
                if not events:
                    st.info("No audit events recorded yet.")
                    return

                table_rows = []
                for e in events:
                    table_rows.append({
                        "ID": e["id"],
                        "Event": e["event_type"],
                        "Doc ID": e.get("document_id", "-"),
                        "Timestamp": e["timestamp"],
                        "Details": e["details"]
                    })

                df = pd.DataFrame(table_rows)
                st.dataframe(df, use_container_width=True, hide_index=True)

                st.markdown("---")
                st.subheader("Inspection of Event Payload")
                selected_event_id = st.selectbox("Select Event ID to inspect raw JSON payload:", [e["id"] for e in events])
                selected_event = next((e for e in events if e["id"] == selected_event_id), None)
                if selected_event:
                    try:
                        parsed = json.loads(selected_event["details"])
                        st.json(parsed)
                    except Exception:
                        st.text(selected_event["details"])
            else:
                st.error(f"Failed to fetch audit log: {res.text}")
        except Exception as e:
            st.error(f"Error fetching audit log: {str(e)}")
