import streamlit as st


def render_metric_card(label: str, value: str, subtext: str = ""):
    """Render a clean engineering metric card."""
    st.markdown(
        f"""
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 12px; margin-bottom: 8px;">
            <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #64748B; font-weight: 600;">{label}</div>
            <div style="font-size: 22px; font-weight: 700; color: #0F172A; margin: 4px 0;">{value}</div>
            <div style="font-size: 11px; color: #94A3B8;">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_confidence_badge(confidence: str, score: float = 0.0):
    """Render confidence level indicator."""
    color_map = {
        "High": ("#15803D", "#DCFCE7", "🟢"),
        "Medium": ("#B45309", "#FEF3C7", "🟡"),
        "Low": ("#C2410C", "#FFEDD5", "🟠"),
        "Insufficient": ("#B91C1C", "#FEE2E2", "🔴")
    }
    fg, bg, emoji = color_map.get(confidence, ("#475569", "#F1F5F9", "⚪"))
    score_str = f" (Sim: {score:.3f})" if score > 0 else ""

    st.markdown(
        f"""
        <div style="display: inline-block; background-color: {bg}; color: {fg}; border: 1px solid {fg}40; border-radius: 4px; padding: 3px 8px; font-size: 12px; font-weight: 600;">
            {emoji} Confidence: {confidence}{score_str}
        </div>
        """,
        unsafe_allow_html=True
    )


def render_inconsistency_card(entity: str, issue_type: str, evidence1: str, evidence2: str, reason: str, status: str):
    """Render potential inconsistency finding in clean engineering style."""
    st.markdown(
        f"""
        <div style="background-color: #FFFBEB; border-left: 4px solid #F59E0B; border-radius: 4px; padding: 12px 16px; margin-bottom: 12px; border-top: 1px solid #FEF3C7; border-right: 1px solid #FEF3C7; border-bottom: 1px solid #FEF3C7;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-weight: 700; color: #92400E; font-size: 14px;">⚠️ Potential Inconsistency — {entity}</span>
                <span style="font-size: 11px; background: #FEF3C7; color: #B45309; padding: 2px 6px; border-radius: 3px; font-weight: 600;">{status}</span>
            </div>
            <div style="font-size: 12px; color: #78350F; margin-bottom: 6px;"><strong>Type:</strong> {issue_type}</div>
            <div style="font-size: 12px; color: #451A03; margin-bottom: 4px;"><strong>Evidence 1:</strong> <code>{evidence1}</code></div>
            <div style="font-size: 12px; color: #451A03; margin-bottom: 6px;"><strong>Evidence 2:</strong> <code>{evidence2}</code></div>
            <div style="font-size: 12px; color: #78350F; background: #FEF9C3; padding: 6px 8px; border-radius: 3px;">
                <strong>Reason:</strong> {reason}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
