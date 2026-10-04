import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether
)


def generate_synthetic_autosar_hld(output_path: str) -> str:
    """
    Generates a realistic multi-page SYNTHETIC AUTOSAR HLD document in PDF format.
    Explicitly labeled as SYNTHETIC. Contains SWCs, Interfaces, Ports, Signals,
    Dependencies, and intentional edge cases to test inconsistency detection.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0B2545"),
        spaceAfter=10
    )
    
    notice_style = ParagraphStyle(
        "Notice",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#C0392B"),
        fontName="Helvetica-Bold",
        spaceAfter=15
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#134074"),
        spaceBefore=12,
        spaceAfter=6,
        fontName="Helvetica-Bold"
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Heading3"],
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#1D4E89"),
        spaceBefore=8,
        spaceAfter=4,
        fontName="Helvetica-Bold"
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1D2D44"),
        spaceAfter=6
    )

    table_header_style = ParagraphStyle(
        "THStyle",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        fontName="Helvetica-Bold"
    )

    cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#1D2D44")
    )

    story = []

    # Title & Synthetic Banner
    story.append(Paragraph("SYNTHETIC AUTOSAR HLD SAMPLE", title_style))
    story.append(Paragraph("High-Level Design Specification — Vehicle Domain & Powertrain Coordinator", h1_style))
    story.append(Paragraph("DISCLAIMER: THIS DOCUMENT IS A SYNTHETIC AUTOSAR ARCHITECTURE SPECIFICATION GENERATED FOR TESTING AND DEMONSTRATION PURPOSES. IT DOES NOT REPRESENT A REAL OEM SYSTEM.", notice_style))
    story.append(Spacer(1, 10))

    # Section 1
    story.append(Paragraph("1. System Overview and Architectural Goals", h1_style))
    story.append(Paragraph(
        "This High-Level Design (HLD) defines the AUTOSAR Classic architecture for the Vehicle Domain Coordinator. "
        "The system coordinates powertrain torque requests, communicates with the central CAN bus through a dedicated gateway, "
        "monitors high-voltage battery state of charge, and performs thermal management and vehicle diagnostics. "
        "All software components adhere to the AUTOSAR 4.4 standard.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # Section 2
    story.append(Paragraph("2. Software Component Architecture", h1_style))
    story.append(Paragraph(
        "The Vehicle Domain architecture comprises six primary Software Components (SWCs). Each component encapsulates "
        "specific functional responsibilities, executing within predefined periodic task containers.",
        body_style
    ))

    comp_data = [
        [Paragraph("Component", table_header_style), Paragraph("Type", table_header_style), Paragraph("Period", table_header_style), Paragraph("Description", table_header_style)],
        [Paragraph("VehicleControlSWC", cell_style), Paragraph("Application SWC", cell_style), Paragraph("10ms", cell_style), Paragraph("Master vehicle control state machine and velocity regulation.", cell_style)],
        [Paragraph("BatteryManagementSWC", cell_style), Paragraph("Application SWC", cell_style), Paragraph("20ms", cell_style), Paragraph("High-voltage battery pack monitoring and State of Charge (SoC).", cell_style)],
        [Paragraph("MotorDriveCoordinatorSWC", cell_style), Paragraph("Application SWC", cell_style), Paragraph("5ms", cell_style), Paragraph("Traction torque arbitrations and electric inverter command.", cell_style)],
        [Paragraph("CanGatewaySWC", cell_style), Paragraph("Service SWC", cell_style), Paragraph("10ms", cell_style), Paragraph("PDU routing, CAN message packing, and transceiver interface.", cell_style)],
        [Paragraph("DiagnosticHandlerSWC", cell_style), Paragraph("Service SWC", cell_style), Paragraph("100ms", cell_style), Paragraph("UDS diagnostics (ISO 14229) and fault event logging.", cell_style)],
        [Paragraph("ThermalControlSWC", cell_style), Paragraph("Sensor/Actuator SWC", cell_style), Paragraph("50ms", cell_style), Paragraph("Coolant loop pump control and battery pack temperature regulation.", cell_style)]
    ]
    t_comp = Table(comp_data, colWidths=[130, 95, 50, 255])
    t_comp.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#134074")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#8DA9C4")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#EEF4F8"), colors.white]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_comp)
    story.append(Spacer(1, 15))

    # Page Break for clean page citations
    story.append(PageBreak())

    # Section 3
    story.append(Paragraph("3. Interface Specifications", h1_style))
    story.append(Paragraph(
        "AUTOSAR Port Interfaces define the contract between SWCs. Sender-Receiver (S/R) interfaces facilitate asynchronous "
        "data element publication, while Client-Server (C/S) interfaces provide synchronous operation invocation.",
        body_style
    ))

    if_data = [
        [Paragraph("Interface", table_header_style), Paragraph("Type", table_header_style), Paragraph("Description", table_header_style)],
        [Paragraph("VehicleSpeedIf", cell_style), Paragraph("Sender-Receiver", cell_style), Paragraph("Broadcasts current calibrated vehicle speed in km/h.", cell_style)],
        [Paragraph("BatteryStatusIf", cell_style), Paragraph("Sender-Receiver", cell_style), Paragraph("Provides battery voltage, current, and state of charge.", cell_style)],
        [Paragraph("MotorTorqueCmdIf", cell_style), Paragraph("Sender-Receiver", cell_style), Paragraph("Carries filtered acceleration torque demands to motor controller.", cell_style)],
        [Paragraph("CanCommunicationIf", cell_style), Paragraph("Client-Server", cell_style), Paragraph("Synchronous frame transmission and message reception service.", cell_style)],
        [Paragraph("DiagnosticServiceIf", cell_style), Paragraph("Client-Server", cell_style), Paragraph("Routine control and DTC status reporting interface.", cell_style)],
        [Paragraph("ThermalStatusIf", cell_style), Paragraph("Sender-Receiver", cell_style), Paragraph("Reports coolant temperatures and chiller valve states.", cell_style)]
    ]
    t_if = Table(if_data, colWidths=[140, 100, 290])
    t_if.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#134074")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#8DA9C4")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#EEF4F8"), colors.white]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_if)
    story.append(Spacer(1, 15))

    # Section 4: Ports
    story.append(Paragraph("4. Port and Interface Mapping", h1_style))
    story.append(Paragraph(
        "Ports instantiate interface definitions on components. Provide Ports (P-Ports) publish data or offer server operations. "
        "Require Ports (R-Ports) consume published data or invoke client operations.",
        body_style
    ))

    port_data = [
        [Paragraph("Port", table_header_style), Paragraph("Direction", table_header_style), Paragraph("Component", table_header_style), Paragraph("Interface", table_header_style)],
        [Paragraph("PPort_VehicleSpeed", cell_style), Paragraph("P-Port", cell_style), Paragraph("VehicleControlSWC", cell_style), Paragraph("VehicleSpeedIf", cell_style)],
        [Paragraph("RPort_VehicleSpeed", cell_style), Paragraph("R-Port", cell_style), Paragraph("MotorDriveCoordinatorSWC", cell_style), Paragraph("VehicleSpeedIf", cell_style)],
        [Paragraph("PPort_TorqueCommand", cell_style), Paragraph("P-Port", cell_style), Paragraph("MotorDriveCoordinatorSWC", cell_style), Paragraph("MotorTorqueCmdIf", cell_style)],
        [Paragraph("PPort_BatteryTelemetry", cell_style), Paragraph("P-Port", cell_style), Paragraph("BatteryManagementSWC", cell_style), Paragraph("BatteryStatusIf", cell_style)],
        [Paragraph("RPort_BatteryTelemetry", cell_style), Paragraph("R-Port", cell_style), Paragraph("ThermalControlSWC", cell_style), Paragraph("BatteryStatusIf", cell_style)],
        [Paragraph("PPort_CanTx", cell_style), Paragraph("P-Port", cell_style), Paragraph("CanGatewaySWC", cell_style), Paragraph("CanCommunicationIf", cell_style)],
        [Paragraph("RPort_CanRx", cell_style), Paragraph("R-Port", cell_style), Paragraph("VehicleControlSWC", cell_style), Paragraph("CanCommunicationIf", cell_style)],
        [Paragraph("PPort_DiagServer", cell_style), Paragraph("P-Port", cell_style), Paragraph("DiagnosticHandlerSWC", cell_style), Paragraph("DiagnosticServiceIf", cell_style)]
    ]
    t_port = Table(port_data, colWidths=[165, 60, 155, 150])
    t_port.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#134074")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#8DA9C4")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#EEF4F8"), colors.white]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_port)
    story.append(Spacer(1, 15))

    # Page Break
    story.append(PageBreak())

    # Section 5: Signals
    story.append(Paragraph("5. Signal and Data Element Catalog", h1_style))
    story.append(Paragraph(
        "Individual signals represent typed atomic data elements transmitted within Sender-Receiver interfaces.",
        body_style
    ))

    sig_data = [
        [Paragraph("Signal", table_header_style), Paragraph("Type", table_header_style), Paragraph("Interface", table_header_style), Paragraph("Description", table_header_style)],
        [Paragraph("Sig_VehicleSpeed", cell_style), Paragraph("uint16", cell_style), Paragraph("VehicleSpeedIf", cell_style), Paragraph("Vehicle ground speed in 0.01 km/h resolution (0 to 300 km/h).", cell_style)],
        [Paragraph("Sig_MotorTorque", cell_style), Paragraph("int16", cell_style), Paragraph("MotorTorqueCmdIf", cell_style), Paragraph("Demanded axle torque in 0.1 Nm resolution (-500 to +500 Nm).", cell_style)],
        [Paragraph("Sig_BatteryVoltage", cell_style), Paragraph("uint16", cell_style), Paragraph("BatteryStatusIf", cell_style), Paragraph("Total battery pack terminal voltage in 0.1 V (0 to 800 V).", cell_style)],
        [Paragraph("Sig_BatteryStateOfCharge", cell_style), Paragraph("uint8", cell_style), Paragraph("BatteryStatusIf", cell_style), Paragraph("Remaining pack energy percentage (0 to 100 %).", cell_style)],
        [Paragraph("Sig_ThermalTemperature", cell_style), Paragraph("int16", cell_style), Paragraph("ThermalStatusIf", cell_style), Paragraph("Coolant loop temperature in degC (-40 to +150 degC).", cell_style)]
    ]
    t_sig = Table(sig_data, colWidths=[165, 55, 140, 170])
    t_sig.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#134074")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#8DA9C4")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#EEF4F8"), colors.white]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_sig)
    story.append(Spacer(1, 15))

    # Section 6: Architectural Dependencies & Functional Flows
    story.append(Paragraph("6. Architectural Dependencies and Communication Flows", h1_style))
    story.append(Paragraph(
        "Communication dependencies establish physical and logical ties between software units across the ECU software stack.",
        body_style
    ))
    story.append(Paragraph(
        "The VehicleControlSWC communicates with the CAN interface through the CanGatewaySWC via CanCommunicationIf. "
        "VehicleControlSWC transmits to CanGatewaySWC for remote telemetry dispatching. "
        "The MotorDriveCoordinatorSWC communicates with VehicleControlSWC to obtain filtered vehicle velocity and pedal requests. "
        "ThermalControlSWC interacts with BatteryManagementSWC to adjust liquid cooling based on high-voltage battery cell thermals. "
        "DiagnosticHandlerSWC connects to CanGatewaySWC to receive UDS diagnostic requests from external OBD-II scan tools.",
        body_style
    ))

    dep_data = [
        [Paragraph("Source", table_header_style), Paragraph("Target", table_header_style), Paragraph("Connection", table_header_style), Paragraph("Flow Description", table_header_style)],
        [Paragraph("VehicleControlSWC", cell_style), Paragraph("CanGatewaySWC", cell_style), Paragraph("CanCommunicationIf", cell_style), Paragraph("CAN telemetry and frame serialization.", cell_style)],
        [Paragraph("MotorDriveCoordinatorSWC", cell_style), Paragraph("VehicleControlSWC", cell_style), Paragraph("VehicleSpeedIf", cell_style), Paragraph("Velocity feedback loop for motor torque regulation.", cell_style)],
        [Paragraph("ThermalControlSWC", cell_style), Paragraph("BatteryManagementSWC", cell_style), Paragraph("BatteryStatusIf", cell_style), Paragraph("Thermal loop closed feedback control.", cell_style)],
        [Paragraph("DiagnosticHandlerSWC", cell_style), Paragraph("CanGatewaySWC", cell_style), Paragraph("CanCommunicationIf", cell_style), Paragraph("UDS diagnostic response transport over CAN.", cell_style)]
    ]
    t_dep = Table(dep_data, colWidths=[140, 130, 110, 150])
    t_dep.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#134074")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#8DA9C4")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#EEF4F8"), colors.white]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_dep)
    story.append(Spacer(1, 15))

    # Page Break
    story.append(PageBreak())

    # Section 7: Auxiliary Notes & Potential Inconsistencies for Verification
    story.append(Paragraph("7. Auxiliary Configuration and Verification Notes", h1_style))
    story.append(Paragraph(
        "This section documents auxiliary interface mappings and legacy compatibility notes for validation tools. "
        "Architectural inconsistency detection engines should inspect these definitions against previous sections.",
        body_style
    ))

    story.append(Paragraph(
        "Note on Port Mapping: In this legacy auxiliary table, Port PPort_BatteryTelemetry is declared as an R-Port "
        "rather than a P-Port for backward compatibility with diagnostic monitoring harness. "
        "Additionally, Signal Sig_BatteryVoltage is documented here with data type float32 instead of uint16.",
        body_style
    ))

    legacy_port_data = [
        [Paragraph("Port", table_header_style), Paragraph("Direction", table_header_style), Paragraph("Component", table_header_style), Paragraph("Interface", table_header_style)],
        [Paragraph("PPort_BatteryTelemetry", cell_style), Paragraph("R-Port", cell_style), Paragraph("BatteryManagementSWC", cell_style), Paragraph("BatteryStatusIf", cell_style)]
    ]
    t_leg = Table(legacy_port_data, colWidths=[165, 60, 155, 150])
    t_leg.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#C0392B")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E74C3C")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_leg)
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        "Auxiliary Dependency: VehicleControlSWC interacts with TirePressureSWC for tire circumference compensation.",
        body_style
    ))

    doc.build(story)
    return output_path


if __name__ == "__main__":
    out = "./data/uploads/SYNTHETIC_AUTOSAR_HLD_SAMPLE.pdf"
    generate_synthetic_autosar_hld(out)
    print(f"Generated synthetic AUTOSAR HLD at: {out}")
