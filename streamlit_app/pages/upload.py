import os
import streamlit as st
import httpx
from pathlib import Path


def render_upload_page(api_base_url: str):
    st.header("📄 Upload AUTOSAR HLD Document")
    st.markdown(
        "Upload an AUTOSAR High-Level Design (HLD) PDF specification to extract architecture knowledge, "
        "detect sections and tables, build vector embeddings, and catalog architectural entities."
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Document Ingestion")
        uploaded_file = st.file_uploader(
            "Choose AUTOSAR HLD PDF",
            type=["pdf"],
            help="Select an AUTOSAR Classic or Adaptive High-Level Design PDF specification."
        )

        project_name = st.text_input("Project Name", value="AUTOSAR HLD Pilot")

        upload_btn = st.button("📤 Upload & Analyze", type="primary", use_container_width=True)

        if upload_btn and uploaded_file is not None:
            with st.spinner("Processing PDF: Extracting text, sections, tables, and generating embeddings..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    data = {"project_name": project_name}

                    response = httpx.post(
                        f"{api_base_url}/upload-hld",
                        files=files,
                        data=data,
                        timeout=180.0
                    )

                    if response.status_code == 200:
                        res = response.json()
                        st.success(f"✓ {res.get('message', 'Document indexed successfully!')}")

                        st.markdown(
                            f"""
                            <div style="background-color: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 6px; padding: 14px; margin-top: 10px;">
                                <div style="font-weight: 700; color: #166534; font-size: 15px; margin-bottom: 8px;">Document: {res.get('document_name')}</div>
                                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; font-size: 13px; color: #15803D;">
                                    <div><strong>Pages:</strong> {res.get('pages')}</div>
                                    <div><strong>Sections detected:</strong> {res.get('sections_detected')}</div>
                                    <div><strong>Chunks created:</strong> {res.get('chunks_created')}</div>
                                    <div><strong>Embeddings generated:</strong> {res.get('embeddings_generated')}</div>
                                    <div><strong>Entities detected:</strong> {res.get('entities_detected')}</div>
                                    <div><strong>Status:</strong> <span style="color: #16A34A; font-weight: bold;">{res.get('status')}</span></div>
                                </div>
                                <div style="margin-top: 8px; font-size: 11px; color: #166534;">SHA-256: <code>{res.get('sha256')}</code></div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        st.session_state["active_document_id"] = res.get("document_id")
                        st.rerun()
                    else:
                        st.error(f"Upload failed: {response.text}")
                except Exception as e:
                    st.error(f"Connection or ingestion error: {str(e)}")

        elif upload_btn and uploaded_file is None:
            st.warning("Please choose a PDF file first or use the sample generator on the right.")

    with col2:
        st.subheader("🧪 Quick Test Sample")
        st.info(
            "Don't have an AUTOSAR HLD PDF handy? Generate and ingest our verified "
            "**SYNTHETIC AUTOSAR HLD SAMPLE** containing realistic components, interfaces, "
            "ports, signals, and test inconsistency patterns."
        )

        sample_btn = st.button("⚡ Generate & Ingest Synthetic Sample", use_container_width=True)
        if sample_btn:
            with st.spinner("Generating and ingesting synthetic AUTOSAR HLD..."):
                try:
                    from sample_data.generate_synthetic_hld import generate_synthetic_autosar_hld
                    sample_path = "./data/uploads/SYNTHETIC_AUTOSAR_HLD_SAMPLE.pdf"
                    generate_synthetic_autosar_hld(sample_path)

                    with open(sample_path, "rb") as f:
                        file_bytes = f.read()

                    files = {"file": ("SYNTHETIC_AUTOSAR_HLD_SAMPLE.pdf", file_bytes, "application/pdf")}
                    data = {"project_name": "AUTOSAR Synthetic Pilot"}

                    response = httpx.post(f"{api_base_url}/upload-hld", files=files, data=data, timeout=180.0)
                    if response.status_code == 200:
                        res = response.json()
                        st.success("✓ Synthetic AUTOSAR HLD generated and indexed!")
                        st.session_state["active_document_id"] = res.get("document_id")
                        st.rerun()
                    else:
                        st.error(f"Sample ingestion failed: {response.text}")
                except Exception as e:
                    st.error(f"Error generating sample: {str(e)}")

    st.markdown("---")
    st.subheader("📚 Ingested Documents Repository")

    try:
        docs_res = httpx.get(f"{api_base_url}/documents", timeout=10.0)
        if docs_res.status_code == 200:
            docs = docs_res.json()
            if docs:
                table_rows = []
                for d in docs:
                    table_rows.append({
                        "ID": d["id"],
                        "Filename": d["filename"],
                        "Pages": d["page_count"],
                        "Sections": d["sections_count"],
                        "Chunks": d["chunks_count"],
                        "Entities": d["entities_count"],
                        "Status": d["status"],
                        "Version": d["version"],
                        "SHA-256": d["sha256"][:16] + "..."
                    })
                st.dataframe(table_rows, use_container_width=True, hide_index=True)
            else:
                st.info("No documents uploaded yet. Upload a PDF or click 'Generate & Ingest Synthetic Sample' above.")
    except Exception as e:
        st.warning(f"Could not load documents: {e}")
