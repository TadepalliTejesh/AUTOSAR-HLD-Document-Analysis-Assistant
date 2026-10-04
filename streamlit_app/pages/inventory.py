import streamlit as st
import httpx
import pandas as pd


def render_inventory_page(api_base_url: str, active_doc_id: int, active_doc_name: str):
    st.header(f"📦 Architecture Inventory — {active_doc_name}")
    st.caption("Extracted and cataloged architectural entities with exact page and section provenance.")

    # Fetch inventory data
    with st.spinner("Fetching architectural inventory..."):
        try:
            res = httpx.get(f"{api_base_url}/documents/{active_doc_id}/inventory", timeout=20.0)
            if res.status_code != 200:
                st.error(f"Could not load inventory: {res.text}")
                return
            data = res.json()
        except Exception as e:
            st.error(f"Connection error: {str(e)}")
            return

    # Top summary metrics
    summary = data.get("summary", {})
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Components", summary.get("components", 0))
    m2.metric("Interfaces", summary.get("interfaces", 0))
    m3.metric("Ports", summary.get("ports", 0))
    m4.metric("Signals", summary.get("signals", 0))
    m5.metric("Dependencies", summary.get("dependencies", 0))

    # Export Buttons
    col_csv, col_xlsx, _ = st.columns([1, 1, 3])
    with col_csv:
        csv_url = f"{api_base_url}/documents/{active_doc_id}/export/csv"
        try:
            csv_data = httpx.get(csv_url, timeout=10.0).content
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"{active_doc_name}_inventory.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception:
            st.button("📥 Download CSV", disabled=True)

    with col_xlsx:
        xlsx_url = f"{api_base_url}/documents/{active_doc_id}/export/xlsx"
        try:
            xlsx_data = httpx.get(xlsx_url, timeout=10.0).content
            st.download_button(
                label="📊 Download Excel",
                data=xlsx_data,
                file_name=f"{active_doc_name}_architecture.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except Exception:
            st.button("📊 Download Excel", disabled=True)

    st.markdown("---")

    # Tabs for each entity type
    tab_comp, tab_if, tab_port, tab_sig, tab_dep = st.tabs([
        "🧩 Components",
        "🔗 Interfaces",
        "🔌 Ports",
        "📡 Signals",
        "🔄 Dependencies"
    ])

    with tab_comp:
        st.subheader("Software Components (SWCs)")
        comp_list = data.get("components", [])
        if comp_list:
            df_comp = pd.DataFrame(comp_list)
            st.dataframe(df_comp, use_container_width=True, hide_index=True)
        else:
            st.info("No software components detected in this document.")

    with tab_if:
        st.subheader("Port Interfaces")
        if_list = data.get("interfaces", [])
        if if_list:
            df_if = pd.DataFrame(if_list)
            st.dataframe(df_if, use_container_width=True, hide_index=True)
        else:
            st.info("No interfaces detected in this document.")

    with tab_port:
        st.subheader("Component Ports (P-Ports / R-Ports)")
        port_list = data.get("ports", [])
        if port_list:
            df_port = pd.DataFrame(port_list)
            st.dataframe(df_port, use_container_width=True, hide_index=True)
        else:
            st.info("No ports detected in this document.")

    with tab_sig:
        st.subheader("Signals and Data Elements")
        sig_list = data.get("signals", [])
        if sig_list:
            df_sig = pd.DataFrame(sig_list)
            st.dataframe(df_sig, use_container_width=True, hide_index=True)
        else:
            st.info("No signals detected in this document.")

    with tab_dep:
        st.subheader("Architectural Dependencies & Flows")
        dep_list = data.get("dependencies", [])
        if dep_list:
            df_dep = pd.DataFrame(dep_list)
            st.dataframe(df_dep, use_container_width=True, hide_index=True)
        else:
            st.info("No explicit dependencies detected in this document.")
