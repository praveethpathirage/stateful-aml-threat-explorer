"""
CYBER-APEX // PROJECT: OVERWATCH
Enterprise-Grade Anti-Money Laundering (AML) Threat Intelligence Dashboard

This application leverages a Neo4j Graph Database, NetworkX for complex topology 
analysis, and Bokeh Server for high-performance interactive visualizations. 
It features a "Risk-Based Subgraph Extraction" algorithm to isolate illicit 
financial flows from large-scale (Big Data) transaction networks.

Author: Praveeth Pathirage
Version: 1.0.0 
"""

import os
import math
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import networkx as nx
from neo4j import GraphDatabase
from jinja2 import Template

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource, HoverTool, TapTool, Range1d, Select, TextInput, Button, Div,
    NodesAndLinkedEdges, Scatter, MultiLine, CategoricalColorMapper, LinearColorMapper, 
    WheelZoomTool, FactorRange
)
from bokeh.plotting import figure, from_networkx
from bokeh.palettes import Magma256
from bokeh.transform import transform

# Suppress minor warnings for a clean production console
warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION & SECURE CREDENTIALS
# ============================================================
# Neo4j AuraDB Connection URI & Credentials
NEO4J_URI = "neo4j+s://ed947b28.databases.neo4j.io"
NEO4J_USER = "ed947b28"
NEO4J_PASS = "i7RMg-1xKhIEgSG1OIkM2c4x5cABcL8BojZShHRR_co"

# Premium Dark Theme UI Color Palette (Glassmorphism inspired)
BG_SIDEBAR = "rgba(9, 14, 23, 0.85)"
BG_CARD = "rgba(13, 19, 33, 0.85)"
NEON_CYAN = "#00E5FF"
NEON_CRIMSON = "#FF003C"
MATRIX_GREEN = "#00FF41"
WARNING_AMBER = "#F59E0B"
TEXT_PRIMARY = "#F3F4F6"
TEXT_SECONDARY = "#9CA3AF"
BORDER_COLOR = "#1F2937"
BORDER_GLOW = "rgba(0, 229, 255, 0.2)"


def get_driver():
    """Initializes and returns a connection driver to the Neo4j AuraDB instance."""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))


# ============================================================
# 2. DATA EXTRACTION & RISK-BASED SUBGRAPH FILTERING
# ============================================================
def load_graph_data(driver):
    """
    Executes Cypher queries to fetch the full transaction database (up to 25,000 edges).
    Implements Risk-Based Subgraph Extraction to filter out non-essential nodes, 
    keeping the visual network beautiful, compact, and highly focused on threats.
    
    Returns:
        G_full: NetworkX directed graph containing the entire dataset (for accurate KPIs).
        G_vis: NetworkX directed graph containing the isolated threat subgraph (for UI).
        edges: Raw edge data dictionary for DataFrame construction.
    """
    print("[SYSTEM] Connecting to Neo4j AuraDB... Fetching FULL Database.")
    with driver.session() as session:
        # Fetch all unique account nodes
        nodes_result = session.run("MATCH (a:Account) RETURN a.id AS id")
        all_nodes = [r["id"] for r in nodes_result]
        
        # Fetch detailed transaction flows with all ML features
        rels_result = session.run(
            """
            MATCH (a:Account)-[t:TRANSFERRED_TO]->(b:Account)
            RETURN a.id AS source, b.id AS target, t.amount AS amount,
                   t.payment_channel AS payment_channel, t.time_since_last_transaction AS time_since_last_transaction,
                   t.velocity_score AS velocity_score, t.spending_deviation_score AS spending_deviation_score,
                   t.hour AS hour, t.day_of_week AS day_of_week,
                   t.is_first_transaction AS is_first_transaction, t.is_new_receiver AS is_new_receiver,
                   t.is_new_bank AS is_new_bank, t.is_new_payment_format AS is_new_payment_format,
                   t.is_cross_bank_transfer AS is_cross_bank_transfer, t.is_cross_currency_transfer AS is_cross_currency_transfer,
                   t.is_fraud AS is_fraud
            LIMIT 25000
            """
        )
        edges = [dict(r) for r in rels_result]

        # Construct the full dataset topology
        G_full = nx.DiGraph()
        for n in all_nodes:
            G_full.add_node(n)
            
        for e in edges:
            G_full.add_edge(
                e["source"], 
                e["target"], 
                **{k: v for k, v in e.items() if k not in ("source", "target")}
            )

        # Calculate absolute node degrees globally
        deg = dict(G_full.degree())
        nx.set_node_attributes(G_full, deg, "degree")

        # Identify all nodes directly involved in fraudulent transactions
        fraud_nodes_set = set()
        for u, v, d in G_full.edges(data=True):
            if d.get("is_fraud", 0) == 1:
                fraud_nodes_set.add(u)
                fraud_nodes_set.add(v)
                
        nx.set_node_attributes(G_full, {n: (1 if n in fraud_nodes_set else 0) for n in G_full.nodes}, "is_fraud")

        # --- RISK-BASED SUBGRAPH EXTRACTION ALGORITHM ---
        keep_nodes = set(fraud_nodes_set)
        
        # Include 1st-degree connections of known threat actors
        for f_node in fraud_nodes_set:
            keep_nodes.update(G_full.neighbors(f_node))
            if G_full.is_directed():
                keep_nodes.update(G_full.predecessors(f_node))
                
        # Fill the remaining visual capacity (~1800 nodes) with the largest hubs
        if len(keep_nodes) < 1800:
            degree_sorted = sorted(G_full.nodes, key=lambda n: G_full.nodes[n].get("degree", 0), reverse=True)
            for n in degree_sorted:
                if len(keep_nodes) >= 1800:
                    break
                keep_nodes.add(n)

        # Generate the visual subgraph
        G_vis = G_full.subgraph(list(keep_nodes)).copy()
        
        # Remove floating/isolated nodes for an aesthetically perfect core network
        isolates = list(nx.isolates(G_vis))
        G_vis.remove_nodes_from(isolates)

        return G_full, G_vis, edges


def edges_to_dataframe(edges):
    """
    Transforms raw edge dictionaries into a highly optimized Pandas DataFrame.
    Performs data cleaning, strict type casting, and generates composite ML features 
    (e.g., novelty_score) used by the analytics engine.
    """
    df = pd.DataFrame(edges)
    if df.empty:
        return df
    
    if "is_cross_border" not in df.columns:
        df["is_cross_border"] = np.random.randint(0, 2, size=len(df))

    # Type casting for core numeric analytics
    numeric_cols = ["amount", "velocity_score", "spending_deviation_score", "time_since_last_transaction"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    # Type casting for Boolean/Integer flags
    int_cols = ["hour", "day_of_week", "is_fraud", "is_first_transaction", "is_new_receiver", "is_new_bank", "is_new_payment_format", "is_cross_bank_transfer", "is_cross_currency_transfer"]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        else:
            df[col] = 0
        
    if "payment_channel" in df.columns:
        df["payment_channel"] = df["payment_channel"].fillna("Unknown").astype(str)
    
    # Feature Engineering for Dashboard Data Sources
    df["is_fraud_str"] = df["is_fraud"].apply(lambda x: "1" if str(x) == "1" else "0")
    df["novelty_score"] = df["is_first_transaction"] + df["is_new_receiver"] + df["is_new_bank"] + df["is_new_payment_format"]
    df["novelty_str"] = df["novelty_score"].astype(str)
    
    day_map = {0:"Mon", 1:"Tue", 2:"Wed", 3:"Thu", 4:"Fri", 5:"Sat", 6:"Sun"}
    df["day_name"] = df["day_of_week"].map(day_map)
    df["hour_str"] = df["hour"].astype(str)
    df["is_cross_border"] = df[["is_cross_bank_transfer", "is_cross_currency_transfer"]].max(axis=1)
    
    return df


# ============================================================
# 3. TOPOLOGY LAYOUT & NODE CLASSIFICATION
# ============================================================
def compute_layout(G_vis):
    """Calculates the physical network layout using Force-Directed graph algorithms."""
    print("[SYSTEM] Processing Compact Topological Layout...")
    return nx.spring_layout(G_vis, k=0.12, iterations=35, seed=42)

def classify_nodes(G_vis):
    """
    Classifies nodes based on statistical thresholds (75th & 50th percentiles) 
    of network degrees combined with known fraud telemetry flags.
    """
    degrees = [G_vis.nodes[n].get("degree", 0) for n in G_vis.nodes]
    if degrees:
        p75 = np.percentile(degrees, 75)
        p50 = np.percentile(degrees, 50)
    else:
        p75 = 10
        p50 = 5

    node_types = {}
    for n in G_vis.nodes:
        deg = G_vis.nodes[n].get("degree", 0)
        fraud = G_vis.nodes[n].get("is_fraud", 0)
        
        # AML Behavioral Classification Logic
        if fraud == 1 and deg >= p75:
            node_types[n] = "Primary Fraud Hub"
        elif fraud == 1 and deg >= p50:
            node_types[n] = "Mule Account"
        elif fraud == 1:
            node_types[n] = "Illicit Node"
        elif deg >= p75:
            node_types[n] = "Legitimate Hub"
        else:
            node_types[n] = "Cleared Entity"
            
    nx.set_node_attributes(G_vis, node_types, "node_type")
    return node_types


# ============================================================
# 4. FRONTEND UI & CSS INJECTION
# ============================================================
global_styles = Div(text=f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap');
    
    .bk-input, select.bk {{ background-color: rgba(13, 19, 33, 0.9) !important; color: {NEON_CYAN} !important; border: 1px solid {BORDER_COLOR} !important; border-radius: 6px !important; font-weight: bold; padding: 8px !important; }}
    .bk-input:focus, select.bk:focus {{ border-color: {NEON_CYAN} !important; outline: none; box-shadow: 0 0 8px {BORDER_GLOW}; }}
    .bk-btn-primary {{ background-color: {NEON_CYAN} !important; color: #000 !important; font-weight: 800 !important; border: none !important; border-radius: 6px !important; transition: all 0.3s; margin-top: 15px; }}
    .bk-btn-primary:hover {{ box-shadow: 0 0 15px {BORDER_GLOW} !important; transform: translateY(-2px); }}
    label.bk {{ color: {TEXT_SECONDARY} !important; font-size: 11px !important; font-weight: 800 !important; text-transform: uppercase; letter-spacing: 1px; margin-top: 10px; }}
</style>
""", sizing_mode="stretch_width")

def generate_kpi_html(total_accounts, fraud_txns, fraud_rate, total_amount, avg_velocity, fraud_hubs):
    """Generates the dynamic HTML/CSS grid for the upper Master KPI cards."""
    if total_amount >= 1e9:
        amt_str = f"${total_amount/1e9:.1f}B"
    else:
        amt_str = f"${total_amount/1e6:.1f}M"
        
    return f"""
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; width: 100%; margin-bottom: 20px;">
        <div style="background: {BG_CARD}; border: 1px solid {BORDER_COLOR}; border-radius: 12px; padding: 20px; border-top: 3px solid {NEON_CYAN}; text-align: center; transition: all 0.3s; backdrop-filter: blur(10px);">
            <div style="color: #9CA3AF; font-size: 11px; font-weight: 800; letter-spacing: 1px; margin-bottom: 5px;">◉ TOTAL ENTITIES (FULL DB)</div>
            <div style="font-size: 26px; font-weight: 800; color: #FFF;">{total_accounts:,}</div>
        </div>
        <div style="background: {BG_CARD}; border: 1px solid {BORDER_COLOR}; border-radius: 12px; padding: 20px; border-top: 3px solid {NEON_CRIMSON}; text-align: center; transition: all 0.3s; backdrop-filter: blur(10px);">
            <div style="color: #9CA3AF; font-size: 11px; font-weight: 800; letter-spacing: 1px; margin-bottom: 5px;">⚠ FRAUD INCIDENTS</div>
            <div style="font-size: 26px; font-weight: 800; color: {NEON_CRIMSON};">{fraud_txns:,}</div>
        </div>
        <div style="background: {BG_CARD}; border: 1px solid {BORDER_COLOR}; border-radius: 12px; padding: 20px; border-top: 3px solid {WARNING_AMBER}; text-align: center; transition: all 0.3s; backdrop-filter: blur(10px);">
            <div style="color: #9CA3AF; font-size: 11px; font-weight: 800; letter-spacing: 1px; margin-bottom: 5px;">$ ILLICIT EXPOSURE</div>
            <div style="font-size: 26px; font-weight: 800; color: {WARNING_AMBER};">{amt_str}</div>
        </div>
        <div style="background: {BG_CARD}; border: 1px solid {BORDER_COLOR}; border-radius: 12px; padding: 20px; border-top: 3px solid {MATRIX_GREEN}; text-align: center; transition: all 0.3s; backdrop-filter: blur(10px);">
            <div style="color: #9CA3AF; font-size: 11px; font-weight: 800; letter-spacing: 1px; margin-bottom: 5px;">▲ FRAUD HUBS</div>
            <div style="font-size: 26px; font-weight: 800; color: {MATRIX_GREEN};">{fraud_hubs}</div>
        </div>
    </div>
    """


# ============================================================
# 5. DATA SCIENCE VISUALIZATIONS ENGINE (BOKEH PLOTS)
# ============================================================
def create_base_figure(title, h=330, tools="pan,wheel_zoom,box_zoom,reset,save"):
    """Utility function to standardize Bokeh plot aesthetics across the dashboard."""
    p = figure(title=title, height=h, sizing_mode="stretch_width", tools=tools, 
               background_fill_color=BG_CARD, border_fill_color=BG_CARD)
    p.title.text_color = NEON_CYAN
    p.title.text_font_size = "12px"
    p.title.text_font_style = "bold"
    p.axis.axis_line_color = BORDER_COLOR
    p.axis.major_label_text_color = TEXT_SECONDARY
    p.grid.grid_line_color = BORDER_COLOR
    p.outline_line_color = BORDER_COLOR
    
    if p.select_one(WheelZoomTool):
        p.toolbar.active_scroll = p.select_one(WheelZoomTool)
        
    return p

def build_network_plot(G_vis, pos, node_types):
    """Builds the central interactive Topological Threat Ecosystem graph."""
    color_map = {
        "Primary Fraud Hub": NEON_CRIMSON, 
        "Mule Account": WARNING_AMBER, 
        "Illicit Node": "#FF6B6B", 
        "Legitimate Hub": NEON_CYAN, 
        "Cleared Entity": MATRIX_GREEN
    }
    marker_map = {
        "Primary Fraud Hub": "hex", 
        "Mule Account": "triangle", 
        "Illicit Node": "circle", 
        "Legitimate Hub": "circle", 
        "Cleared Entity": "circle"
    }
    
    # Inject visual metadata directly into NetworkX nodes
    for n in G_vis.nodes:
        ntype = node_types.get(n, "Cleared Entity")
        visible_deg = G_vis.degree(n) 
        
        G_vis.nodes[n]['node_type'] = str(ntype)
        G_vis.nodes[n]['node_color'] = str(color_map.get(ntype, MATRIX_GREEN))
        G_vis.nodes[n]['node_marker'] = str(marker_map.get(ntype, "circle"))
        G_vis.nodes[n]['node_size'] = float(max(5, min(18, visible_deg * 1.2))) 
        G_vis.nodes[n]['id_str'] = str(n)
        G_vis.nodes[n]['degree_str'] = str(visible_deg) 
        G_vis.nodes[n]['alpha'] = 0.8  

    for u, v in G_vis.edges():
        G_vis.edges[u, v]['alpha'] = 0.15

    plot = create_base_figure("TOPOLOGICAL THREAT ECOSYSTEM (Click a Node for Intelligence Dossier | Scroll to Zoom)", h=450)
    plot.axis.visible = False
    plot.grid.visible = False

    graph_renderer = None
    if len(G_vis.nodes) > 0:
        graph_renderer = from_networkx(G_vis, pos)
        
        # Configure Node and Edge Renderers
        graph_renderer.node_renderer.glyph = Scatter(marker="node_marker", size="node_size", fill_color="node_color", fill_alpha="alpha", line_color="#FFF", line_width=0.4, line_alpha="alpha")
        graph_renderer.edge_renderer.glyph = MultiLine(line_color="#4B5563", line_alpha="alpha", line_width=0.6)
        
        # Setup Selection Interactivity (Dimming unselected nodes)
        graph_renderer.selection_policy = NodesAndLinkedEdges()
        graph_renderer.inspection_policy = NodesAndLinkedEdges()
        
        graph_renderer.node_renderer.selection_glyph = Scatter(fill_alpha=1.0, line_color=NEON_CYAN, line_width=2.5)
        graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=NEON_CYAN, line_width=2.0, line_alpha=1.0)
        
        graph_renderer.node_renderer.nonselection_glyph = None
        graph_renderer.edge_renderer.nonselection_glyph = None
        
        plot.renderers.append(graph_renderer)
        
        hover_tool = HoverTool(
            renderers=[graph_renderer.node_renderer], 
            tooltips=[
                ("Entity ID", "@id_str"), 
                ("Classification", "@node_type"), 
                ("Visible Connections", "@degree_str")
            ]
        )
        plot.add_tools(hover_tool)
        plot.add_tools(TapTool())

    return plot, graph_renderer

def build_bubble_chart(df):
    """Plot 1: Identifies high-volume anomalies (Whales) vs normal activity (Minnows)."""
    if not df.empty:
        max_amt = df['amount'].max() if df['amount'].max() > 0 else 1
        df['bubble_size'] = (df['amount'] / max_amt * 25) + 5
    else:
        df['bubble_size'] = []
        
    src = ColumnDataSource(df)
    plot = create_base_figure("1. WHALE VS MINNOW (Volume, Velocity, Deviation)")
    plot.xaxis.axis_label = "Velocity Score"
    plot.yaxis.axis_label = "Spending Deviation"
    
    color_mapper = CategoricalColorMapper(factors=["0", "1"], palette=[MATRIX_GREEN, NEON_CRIMSON])
    rend = plot.scatter("velocity_score", "spending_deviation_score", size="bubble_size", source=src, color={"field": "is_fraud_str", "transform": color_mapper}, alpha=0.6, line_color="#FFF", line_width=0.5)
    
    plot.add_tools(HoverTool(renderers=[rend], tooltips=[("Amount", "$@amount{0,0.00}"), ("Velocity", "@velocity_score"), ("Type", "@is_fraud_str")]))
    return plot, src

def build_heatmap_chart(df):
    """Plot 2: Visualizes density of fraud attempts across time and weekdays."""
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hours = [str(i) for i in range(24)]
    
    if not df.empty:
        fraud_df = df[df['is_fraud'] == 1]
        heat_df = fraud_df.groupby(["day_name", "hour_str"]).size().reset_index(name='fraud_count')
    else:
        heat_df = pd.DataFrame({"day_name":[], "hour_str":[], "fraud_count":[]})
        
    src = ColumnDataSource(heat_df)
    plot = figure(title="2. TEMPORAL THREAT HEATMAP (Fraud Hotspots)", height=330, sizing_mode="stretch_width", x_range=FactorRange(factors=hours), y_range=FactorRange(factors=days), tools="hover,save", background_fill_color=BG_CARD, border_fill_color=BG_CARD)
    plot.title.text_color = NEON_CYAN
    plot.title.text_font_size = "12px"
    plot.title.text_font_style = "bold"
    plot.axis.axis_line_color = None
    plot.axis.major_label_text_color = TEXT_SECONDARY
    plot.grid.grid_line_color = None
    plot.outline_line_color = BORDER_COLOR
    
    mapper = LinearColorMapper(palette=Magma256[::-1], low=0, high=heat_df['fraud_count'].max() if not heat_df.empty else 10)
    plot.rect(x="hour_str", y="day_name", width=1, height=1, source=src, fill_color=transform('fraud_count', mapper), line_color=BG_CARD)
    plot.hover.tooltips = [("Day", "@day_name"), ("Hour", "@hour_str:00"), ("Fraud Incidents", "@fraud_count")]
    return plot, src

def build_novelty_chart(df):
    """Plot 3: Assesses risk based on unprecedented combinations of accounts/channels."""
    if not df.empty:
        novelty_stats = df.groupby("novelty_str").agg(fraud_count=("is_fraud", lambda x: (x==1).sum())).reset_index()
    else:
        novelty_stats = pd.DataFrame({"novelty_str":[], "fraud_count":[]})
        
    src = ColumnDataSource(novelty_stats)
    factors = ["0", "1", "2", "3", "4"]
    
    plot = figure(title="3. NOVELTY RISK INDEX (New Behavior Score)", height=330, sizing_mode="stretch_width", x_range=FactorRange(factors=factors), tools="hover,save", background_fill_color=BG_CARD, border_fill_color=BG_CARD)
    plot.title.text_color = NEON_CYAN
    plot.title.text_font_size = "12px"
    plot.title.text_font_style = "bold"
    plot.xaxis.axis_label = "Novelty Score (0 to 4)"
    plot.axis.axis_line_color = BORDER_COLOR
    plot.axis.major_label_text_color = TEXT_SECONDARY
    plot.grid.grid_line_color = BORDER_COLOR
    
    color_mapper = CategoricalColorMapper(factors=factors, palette=[MATRIX_GREEN, "#85C1E9", WARNING_AMBER, "#E74C3C", NEON_CRIMSON])
    plot.vbar(x="novelty_str", top="fraud_count", width=0.6, source=src, color={"field": "novelty_str", "transform": color_mapper}, alpha=0.9)
    plot.hover.tooltips = [("Novelty Score", "@novelty_str"), ("Fraud Incidents", "@fraud_count")]
    return plot, src

def build_crossborder_chart(df):
    """Plot 4: Distinguishes between domestic vs international laundering channels."""
    if not df.empty:
        fraud_df = df[df['is_fraud'] == 1]
        cb_stats = fraud_df.groupby("payment_channel").agg(domestic=("is_cross_border", lambda x: (x==0).sum()), cross_border=("is_cross_border", lambda x: (x==1).sum())).reset_index()
        channels = cb_stats["payment_channel"].astype(str).tolist()
    else:
        cb_stats = pd.DataFrame({"payment_channel":[], "domestic":[], "cross_border":[]})
        channels = ["Unknown"]
        
    src = ColumnDataSource(cb_stats)
    plot = figure(title="4. CROSS-BORDER VULNERABILITY (Fraud Flows)", height=330, sizing_mode="stretch_width", x_range=FactorRange(factors=channels if channels else ["Unknown"]), tools="pan,wheel_zoom,reset,hover,save", background_fill_color=BG_CARD, border_fill_color=BG_CARD)
    plot.title.text_color = NEON_CYAN
    plot.title.text_font_size = "12px"
    plot.title.text_font_style = "bold"
    plot.xaxis.major_label_orientation = 0.5
    plot.axis.axis_line_color = BORDER_COLOR
    plot.axis.major_label_text_color = TEXT_SECONDARY
    plot.grid.grid_line_color = None
    
    if plot.select_one(WheelZoomTool):
        plot.toolbar.active_scroll = plot.select_one(WheelZoomTool)
        
    plot.vbar_stack(['domestic', 'cross_border'], x="payment_channel", width=0.6, source=src, color=[WARNING_AMBER, NEON_CRIMSON], legend_label=['Domestic Fraud', 'Cross-Border Fraud'])
    plot.hover.tooltips = [("Channel", "@payment_channel"), ("Domestic", "@domestic"), ("Cross Border", "@cross_border")]
    plot.legend.location = "top_right"
    plot.legend.background_fill_color = BG_CARD
    plot.legend.label_text_color = TEXT_PRIMARY
    return plot, src

def build_smurfing_chart(df):
    """Plot 5: Identifies micro-structuring (Smurfing) via high velocity + short time gaps."""
    src = ColumnDataSource(df)
    plot = create_base_figure("5. SMURFING ANALYSIS (Time-Gap vs Velocity)")
    plot.xaxis.axis_label = "Time Since Last Txn (Hours)"
    plot.yaxis.axis_label = "Velocity Score"
    
    color_mapper = CategoricalColorMapper(factors=["0", "1"], palette=[MATRIX_GREEN, NEON_CRIMSON])
    scatter_rend = plot.scatter("time_since_last_transaction", "velocity_score", source=src, color={"field": "is_fraud_str", "transform": color_mapper}, size=6, alpha=0.6)
    plot.add_tools(HoverTool(renderers=[scatter_rend], tooltips=[("Time Gap", "@time_since_last_transaction"), ("Velocity", "@velocity_score")]))
    return plot, src

def build_radar_chart():
    """Plot 6: Dynamic 5-dimensional polygon assessing combined multi-vector threat risk."""
    plot = figure(title="6. FEATURE THREAT PROFILE (Radar Index)", height=330, sizing_mode="stretch_width", x_range=Range1d(-1.5, 1.5), y_range=Range1d(-1.5, 1.5), tools="hover", background_fill_color=BG_CARD, border_fill_color=BG_CARD)
    plot.title.text_color = NEON_CYAN
    plot.title.text_font_size = "12px"
    plot.title.text_font_style = "bold"
    plot.axis.visible = False
    plot.grid.visible = False
    plot.outline_line_color = None
    
    angles = np.linspace(0, 2*np.pi, 6)
    for r in [0.33, 0.66, 1.0]:
        plot.line(r * np.cos(angles), r * np.sin(angles), line_color=BORDER_COLOR, line_dash="dotted")
    for a in angles[:-1]:
        plot.line([0, np.cos(a)], [0, np.sin(a)], line_color=BORDER_COLOR, line_dash="dotted")
        
    labels = ["Velocity Risk", "Deviation Risk", "Time Spikes", "Volume Risk", "Cross-Border"]
    plot.text(x=1.2 * np.cos(angles[:-1]), y=1.2 * np.sin(angles[:-1]), text=labels, text_color=TEXT_SECONDARY, text_font_size="9px", text_align="center")
    
    src = ColumnDataSource(dict(x=[], y=[]))
    plot.patch('x', 'y', source=src, fill_color=NEON_CRIMSON, fill_alpha=0.4, line_color=NEON_CRIMSON, line_width=2)
    plot.circle('x', 'y', source=src, size=6, color=NEON_CYAN)
    return plot, src


# ============================================================
# 6. MAIN APPLICATION ENGINE & CALLBACK REGISTRY
# ============================================================
def build_app():
    """Bootstraps the Bokeh Server Document, orchestrating data, plots, and UI interactivity."""
    print("=" * 60)
    print("  CYBER-APEX // PROJECT: OVERWATCH — BOOTUP SEQUENCE")
    print("=" * 60)

    # 1. Establish Secure Database Connection
    driver = get_driver()
    try:
        G_full, G_vis, edges = load_graph_data(driver)
    except Exception as e:
        print(f"[WARN] Neo4j connection failed: {e}")
        # Fallback to local data generation if remote DB fails
        G_full, G_vis, edges = generate_synthetic_data()
    finally:
        driver.close()

    # 2. Build the primary Analytics Dataframe
    df = edges_to_dataframe(edges)
    
    # 3. Compute expensive operations (Physics layout & ML Classification) ONCE
    pos = compute_layout(G_vis)
    node_types = classify_nodes(G_vis)

    # 4. Instantiate all Data Science Visualizations
    net_plot, net_rend = build_network_plot(G_vis, pos, node_types)
    bubble_plot, bubble_src = build_bubble_chart(df)
    heat_plot, heat_src = build_heatmap_chart(df)
    novelty_plot, novelty_src = build_novelty_chart(df)
    cb_plot, cb_src = build_crossborder_chart(df)
    smurf_plot, smurf_src = build_smurfing_chart(df)
    radar_plot, radar_src = build_radar_chart()

    master_kpi_div = Div(text="", sizing_mode="stretch_width")
    
    # Intelligence Dossier Base HTML Template
    default_dossier_html = f"""
    <div style="background: {BG_CARD}; border: 1px solid {BORDER_COLOR}; border-radius: 12px; padding: 20px; height: 100%; border-top: 3px solid #374151; backdrop-filter: blur(10px); box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
        <div style="color: #9CA3AF; font-size: 11px; font-weight: 800; letter-spacing: 1px; margin-bottom: 15px;">TARGET DOSSIER // INTELLIGENCE</div>
        <div style="color: #F3F4F6; font-size: 13px; font-style: italic; line-height: 1.6;">Awaiting target selection...<br><br>Click on any node in the Topological Network to retrieve live entity analytics and classified financial records.</div>
    </div>
    """
    dossier_div = Div(text=default_dossier_html, width=320, sizing_mode="stretch_height")

    # Global Maxima for Radar Chart Normalization
    if not df.empty:
        max_vel = df['velocity_score'].max() if df['velocity_score'].max() > 0 else 1
        max_dev = df['spending_deviation_score'].max() if df['spending_deviation_score'].max() > 0 else 1
        max_time = df['time_since_last_transaction'].max() if df['time_since_last_transaction'].max() > 0 else 1
        max_amt = df['amount'].max() if df['amount'].max() > 0 else 1
    else:
        max_vel = 1
        max_dev = 1
        max_time = 1
        max_amt = 1

    # --- GLOBAL ANALYTICS UPDATE CONTROLLER ---
    def update_analytics(filtered_df):
        """Updates the ColumnDataSources of all 6 graphs and re-renders HTML KPIs in real-time."""
        total_accounts = len(G_full.nodes)
        
        # Core Analytics Aggregations
        if not filtered_df.empty:
            fraud_txns = int((filtered_df["is_fraud"] == 1).sum())
            fraud_rate = (fraud_txns / max(len(filtered_df), 1)) * 100
        else:
            fraud_txns = 0
            fraud_rate = 0
        
        fraud_df = filtered_df[filtered_df['is_fraud'] == 1]
        
        # Calculate strict illicit exposure (Only derived from flagged fraud flows)
        if not fraud_df.empty:
            illicit_amount = fraud_df['amount'].sum()
        else:
            illicit_amount = 0
        
        if not filtered_df.empty:
            avg_velocity = filtered_df["velocity_score"].mean()
        else:
            avg_velocity = 0
            
        fraud_hubs = sum(1 for v in node_types.values() if v == 'Primary Fraud Hub')
        
        master_kpi_div.text = generate_kpi_html(total_accounts, fraud_txns, fraud_rate, illicit_amount, avg_velocity, fraud_hubs)

        # Syncing downstream UI Components
        if not filtered_df.empty:
            max_amt_f = filtered_df['amount'].max() if filtered_df['amount'].max() > 0 else 1
            
            scatter_data = dict(
                velocity_score=filtered_df["velocity_score"], 
                spending_deviation_score=filtered_df["spending_deviation_score"],
                time_since_last_transaction=filtered_df["time_since_last_transaction"], 
                amount=filtered_df["amount"],
                bubble_size=(filtered_df['amount'] / max_amt_f * 25) + 5,
                is_fraud_str=filtered_df["is_fraud"].apply(lambda x: "1" if str(x)=="1" else "0")
            )
            bubble_src.data = scatter_data
            smurf_src.data = scatter_data

            if not fraud_df.empty:
                heat_df = fraud_df.groupby(["day_name", "hour_str"]).size().reset_index(name='fraud_count')
            else:
                heat_df = pd.DataFrame({"day_name":[], "hour_str":[], "fraud_count":[]})
                
            heat_src.data = dict(day_name=heat_df["day_name"], hour_str=heat_df["hour_str"], fraud_count=heat_df["fraud_count"])

            nov_df = filtered_df.groupby("novelty_str").agg(fraud_count=("is_fraud", lambda x: (x==1).sum())).reset_index()
            novelty_src.data = dict(novelty_str=nov_df["novelty_str"].astype(str), fraud_count=nov_df["fraud_count"])

            if not fraud_df.empty:
                cb_stats = fraud_df.groupby("payment_channel").agg(domestic=("is_cross_border", lambda x: (x==0).sum()), cross_border=("is_cross_border", lambda x: (x==1).sum())).reset_index()
                factors = cb_stats["payment_channel"].astype(str).tolist()
                cb_src.data = dict(payment_channel=cb_stats["payment_channel"].astype(str), domestic=cb_stats["domestic"], cross_border=cb_stats["cross_border"])
                cb_plot.x_range.factors = factors if factors else ["Unknown"]
            else:
                cb_src.data = dict(payment_channel=[], domestic=[], cross_border=[])
                cb_plot.x_range.factors = ["Unknown"]

            if not fraud_df.empty:
                v_risk = fraud_df['velocity_score'].mean() / max_vel
                d_risk = fraud_df['spending_deviation_score'].mean() / max_dev
                t_risk = 1 - (fraud_df['time_since_last_transaction'].mean() / max_time)
                a_risk = fraud_df['amount'].mean() / max_amt
                c_risk = fraud_df['is_cross_border'].mean()
            else:
                v_risk = d_risk = t_risk = a_risk = c_risk = 0.1
                
        else:
            # Fallback Empty States
            bubble_src.data = dict(velocity_score=[], spending_deviation_score=[], time_since_last_transaction=[], amount=[], bubble_size=[], is_fraud_str=[])
            smurf_src.data = dict(velocity_score=[], spending_deviation_score=[], time_since_last_transaction=[], amount=[], bubble_size=[], is_fraud_str=[])
            heat_src.data = dict(day_name=[], hour_str=[], fraud_count=[])
            novelty_src.data = dict(novelty_str=[], fraud_count=[])
            cb_src.data = dict(payment_channel=[], domestic=[], cross_border=[])
            cb_plot.x_range.factors = ["Unknown"]
            v_risk = d_risk = t_risk = a_risk = c_risk = 0.0

        scores = [v_risk, d_risk, t_risk, a_risk, c_risk]
        scores.append(scores[0])
        angles = np.linspace(0, 2*np.pi, 6)
        radar_src.data = dict(x=scores * np.cos(angles), y=scores * np.sin(angles))

    # Initialize analytics cascade on initial load
    update_analytics(df)

    # --- UI COMPONENT REGISTRY (Sidebar) ---
    search_input = TextInput(title="TARGET ID SEARCH:", placeholder="e.g. 2043")
    volume_opts = ["All Volumes", "Micro (< $1,000)", "Standard ($1k - $50k)", "Whale (> $50,000)"]
    volume_select = Select(title="TRANSACTION VOLUME", value="All Volumes", options=volume_opts)
    threat_select = Select(title="THREAT CLASSIFICATION", value="All", options=["All", "High Risk (Fraud)", "Legitimate"])
    time_opts = ["All Hours (0-24)", "Morning Shift (06:00 - 14:00)", "Evening Shift (14:00 - 22:00)", "Night Shift (22:00 - 06:00)"]
    time_select = Select(title="TEMPORAL SHIFT", value="All Hours (0-24)", options=time_opts)
    region_opts = ["All Regions", "Domestic Only", "Cross-Border Only"]
    region_select = Select(title="GEOGRAPHIC FLOW", value="All Regions", options=region_opts)
    channel_opts = ["All"] + sorted(df["payment_channel"].unique().tolist()) if not df.empty else ["All"]
    channel_select = Select(title="PAYMENT CHANNEL", value="All", options=channel_opts)
    reset_btn = Button(label="⟲ INITIALIZE RESET", button_type="primary")

    sidebar = column(
        Div(text=f"""
        <div style="margin-bottom: 25px;">
            <h1 style="color:#FFF; font-size:24px; font-weight:900; margin:0; letter-spacing:1px; font-family:sans-serif;">CYBER-APEX</h1>
            <p style="color:{NEON_CYAN}; font-size:12px; margin:0; font-weight:bold; letter-spacing:2px; font-family:sans-serif;">PROJECT: OVERWATCH</p>
            <p style="color:{TEXT_SECONDARY}; font-size:10px; margin-top:5px; border-bottom:1px solid {BORDER_COLOR}; padding-bottom:10px; font-family:sans-serif;">Global Financial Threat Intelligence Syndicate</p>
        </div>
        <p style="color:#FFF; font-size:13px; font-weight:bold; margin-bottom:10px; font-family:sans-serif;">⚙ Core Telemetry Filters</p>
        """),
        search_input, volume_select, threat_select, region_select, time_select, channel_select, reset_btn,
        width=300, sizing_mode="stretch_height", background=BG_SIDEBAR, margin=(0, 20, 0, 0)
    )

    # UI Grid Assembly
    network_with_dossier = row(net_plot, dossier_div, sizing_mode="stretch_width", spacing=15)
    charts_grid_top = row(bubble_plot, heat_plot, novelty_plot, sizing_mode="stretch_width", spacing=15)
    charts_grid_bot = row(cb_plot, smurf_plot, radar_plot, sizing_mode="stretch_width", spacing=15)
    analytics_master_col = column(charts_grid_top, charts_grid_bot, sizing_mode="stretch_width", spacing=15)

    main_content = column(
        master_kpi_div,
        network_with_dossier,
        Div(text=f"<div style='margin-top:20px; margin-bottom:10px; border-left:4px solid {NEON_CYAN}; padding-left:10px; color:{TEXT_PRIMARY}; font-weight:bold; font-family:sans-serif; text-shadow: 0 0 5px rgba(0,229,255,0.3);'>THE 6-PART DATA SCIENCE ANALYTICS ENGINE</div>"),
        analytics_master_col,
        sizing_mode="stretch_width"
    )

    layout = column(
        global_styles,
        row(sidebar, main_content, sizing_mode="stretch_width"),
        sizing_mode="stretch_width", margin=(20, 20, 20, 20)
    )
    
    curdoc().theme = "dark_minimal"
    curdoc().title = "Cyber-Apex Overwatch"
    
    # ============================================================
    # 7. JINJA2 TEMPLATE INJECTION (Full-Screen World Network UI)
    # ============================================================
    curdoc().template = Template("""
    {% extends base %}
    {% block postamble %}
    <style>
        body, html {
            /* Implements a profound Dark Overlay over the Cyber-Matrix Background */
            background: linear-gradient(rgba(11, 19, 37, 0.85), rgba(1, 2, 5, 0.95)), 
                        url('https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop') no-repeat center center fixed !important;
            background-size: cover !important;
            margin: 0 !important;
            padding: 0 !important;
            min-height: 100vh !important;
            color: #F3F4F6 !important;
            font-family: 'Inter', sans-serif !important;
        }
        .bk-root {
            background-color: transparent !important;
        }
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #1F2937; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #00E5FF; }
    </style>
    {% endblock %}
    """)

    curdoc().add_root(layout)

    # --- CALLBACKS & EVENT LISTENERS ---
    def apply_filters(live_query=None):
        """Monitors UI state changes and dynamically masks the core DataFrame."""
        if df.empty: return
        mask = pd.Series([True] * len(df))
        
        # Strict Exact Match Search implementation
        raw_query = live_query if live_query is not None else search_input.value
        query = raw_query.upper().replace("ACC", "").replace("_", "").replace("-", "").strip()
        
        if query: 
            source_clean = df["source"].str.upper().str.replace("ACC", "").str.replace("_", "").str.replace("-", "")
            target_clean = df["target"].str.upper().str.replace("ACC", "").str.replace("_", "").str.replace("-", "")
            mask &= (source_clean == query) | (target_clean == query)
        
        # Sequential Volume Filtering
        if volume_select.value == "Micro (< $1,000)":
            mask &= df["amount"] < 1000
        elif volume_select.value == "Standard ($1k - $50k)":
            mask &= (df["amount"] >= 1000) & (df["amount"] <= 50000)
        elif volume_select.value == "Whale (> $50,000)":
            mask &= df["amount"] > 50000
            
        # Chronological Shift Filtering
        if time_select.value == "Morning Shift (06:00 - 14:00)":
            mask &= (df["hour"] >= 6) & (df["hour"] < 14)
        elif time_select.value == "Evening Shift (14:00 - 22:00)":
            mask &= (df["hour"] >= 14) & (df["hour"] < 22)
        elif time_select.value == "Night Shift (22:00 - 06:00)":
            mask &= (df["hour"] >= 22) | (df["hour"] < 6)

        # Spatial Filtering
        if region_select.value == "Domestic Only":
            mask &= df["is_cross_border"] == 0
        elif region_select.value == "Cross-Border Only":
            mask &= df["is_cross_border"] == 1
        
        # Entity Threat Filtering
        if threat_select.value == "High Risk (Fraud)":
            mask &= df["is_fraud"] == 1
        elif threat_select.value == "Legitimate":
            mask &= df["is_fraud"] == 0
            
        if channel_select.value != "All":
            mask &= df["payment_channel"] == channel_select.value

        filtered = df[mask]
        update_analytics(filtered)

        # Execute Visual Fade transitions across the Network Graph
        if len(filtered) == len(df) and not query:
            node_alphas = [0.8 for n in G_vis.nodes]
            edge_alphas = [0.15 for _ in net_rend.edge_renderer.data_source.data['start']]
        else:
            valid_nodes = set(filtered['source']).union(set(filtered['target']))
            valid_edges = set(zip(filtered['source'], filtered['target']))
            
            if query:
                matching = {n for n in G_vis.nodes if query == str(n).upper().replace("ACC", "").replace("_", "").replace("-", "")}
                valid_nodes.update(matching)
                
            node_alphas = [1.0 if str(n) in valid_nodes else 0.03 for n in G_vis.nodes]
            starts = net_rend.edge_renderer.data_source.data['start']
            ends = net_rend.edge_renderer.data_source.data['end']
            
            edge_alphas = []
            for s, e in zip(starts, ends):
                if (str(s), str(e)) in valid_edges or (str(e), str(s)) in valid_edges:
                    edge_alphas.append(0.8)
                else:
                    edge_alphas.append(0.01)

        net_rend.node_renderer.data_source.data['alpha'] = node_alphas
        net_rend.edge_renderer.data_source.data['alpha'] = edge_alphas


    def on_node_select(attr, old, new):
        """Interactively pulls full entity Intelligence records upon Node Click."""
        if not new:
            dossier_div.text = default_dossier_html
            apply_filters()
            return
            
        selected_idx = new[0]
        nodes_list = list(G_vis.nodes)
        
        if selected_idx >= len(nodes_list):
            return
        
        node_id = str(nodes_list[selected_idx])
        node_type = G_vis.nodes[node_id].get('node_type', 'Unknown')
        
        # Calculate metrics using full database relationships
        true_degree = G_full.degree(node_id) 
        node_txns = df[(df['source'] == node_id) | (df['target'] == node_id)]
        total_vol = node_txns['amount'].sum()
        
        if not node_txns.empty:
            avg_vel = node_txns['velocity_score'].mean()
        else:
            avg_vel = 0
            
        fraud_flags = node_txns['is_fraud'].sum()
        
        # Dynamic Risk Scoring Equation
        risk_score = min(99, int((fraud_flags * 30) + (avg_vel / 5) + (true_degree * 2)))
        
        if "Fraud" in node_type or "Illicit" in node_type:
            color = NEON_CRIMSON
        elif "Mule" in node_type:
            color = WARNING_AMBER
        else:
            color = MATRIX_GREEN

        # Render Dossier Profile
        dossier_div.text = f"""
        <div style="background: {BG_CARD}; border: 1px solid {BORDER_COLOR}; border-radius: 12px; padding: 20px; height: 100%; border-top: 3px solid {color}; backdrop-filter: blur(10px); box-shadow: 0 0 15px rgba(0,0,0,0.5);">
            <div style="color: #9CA3AF; font-size: 11px; font-weight: 800; letter-spacing: 1px; margin-bottom: 15px;">TARGET DOSSIER // INTELLIGENCE</div>
            <div style="font-size: 20px; font-weight: 900; color: #FFF; margin-bottom: 5px; letter-spacing: 1px;">{node_id}</div>
            <div style="color: {color}; font-size: 12px; font-weight: bold; margin-bottom: 20px;">{node_type.upper()}</div>
            
            <div style="margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #9CA3AF; font-size: 10px; font-weight: bold; text-transform: uppercase;">Threat Probability</span>
                    <span style="color: {color}; font-size: 12px; font-weight: bold;">{risk_score}%</span>
                </div>
                <div style="width: 100%; background-color: #374151; height: 6px; border-radius: 3px; margin-top: 5px;">
                    <div style="width: {risk_score}%; background-color: {color}; height: 100%; border-radius: 3px;"></div>
                </div>
            </div>

            <div style="margin-bottom: 10px; margin-top: 20px;">
                <div style="color: #9CA3AF; font-size: 10px; font-weight: bold; text-transform: uppercase;">True Network Connections</div>
                <div style="color: #FFF; font-size: 16px; font-weight: bold;">{true_degree} Entities</div>
            </div>
            <div style="margin-bottom: 10px;">
                <div style="color: #9CA3AF; font-size: 10px; font-weight: bold; text-transform: uppercase;">Total Volume Exposed</div>
                <div style="color: #FFF; font-size: 16px; font-weight: bold;">${total_vol:,.2f}</div>
            </div>
            <div style="margin-bottom: 10px;">
                <div style="color: #9CA3AF; font-size: 10px; font-weight: bold; text-transform: uppercase;">Avg Velocity Score</div>
                <div style="color: #FFF; font-size: 16px; font-weight: bold;">{avg_vel:,.1f}</div>
            </div>
            <div style="margin-bottom: 10px;">
                <div style="color: #9CA3AF; font-size: 10px; font-weight: bold; text-transform: uppercase;">Known Fraud Incidents</div>
                <div style="color: {NEON_CRIMSON}; font-size: 16px; font-weight: bold;">{fraud_flags} Detected</div>
            </div>
        </div>
        """
        
        # Localize graph focus (dimming unrelated entities)
        neighbors = set(G_vis.neighbors(node_id))
        if G_vis.is_directed():
            neighbors.update(G_vis.predecessors(node_id))
            
        valid_nodes = neighbors.union({node_id})
        
        node_alphas = [1.0 if str(n) in valid_nodes else 0.05 for n in G_vis.nodes]
        net_rend.node_renderer.data_source.data['alpha'] = node_alphas
        
        starts = net_rend.edge_renderer.data_source.data['start']
        ends = net_rend.edge_renderer.data_source.data['end']
        
        edge_alphas = []
        for s, e in zip(starts, ends):
            if str(s) == node_id or str(e) == node_id:
                edge_alphas.append(0.8)
            else:
                edge_alphas.append(0.02)
                
        net_rend.edge_renderer.data_source.data['alpha'] = edge_alphas

    # Attach Callbacks
    net_rend.node_renderer.data_source.selected.on_change('indices', on_node_select)
    search_input.on_change("value_input", lambda attr, old, new: apply_filters(live_query=new))
    search_input.on_change("value", lambda attr, old, new: apply_filters())
    volume_select.on_change("value", lambda attr, old, new: apply_filters())
    time_select.on_change("value", lambda attr, old, new: apply_filters())
    region_select.on_change("value", lambda attr, old, new: apply_filters())
    threat_select.on_change("value", lambda attr, old, new: apply_filters())
    channel_select.on_change("value", lambda attr, old, new: apply_filters())

    def reset_filters():
        """Resets all active UI telemetry filters to initial states."""
        search_input.value = ""
        volume_select.value = "All Volumes"
        time_select.value = "All Hours (0-24)"
        region_select.value = "All Regions"
        threat_select.value = "All"
        channel_select.value = "All"
        dossier_div.text = default_dossier_html
        apply_filters()

    reset_btn.on_click(reset_filters)

# ============================================================
# 8. OFFLINE FALLBACK (SYNTHETIC ML DATA GENERATOR)
# ============================================================
def generate_synthetic_data():
    """Generates an extremely imbalanced (realistic) AML dataset if DB is unreachable."""
    np.random.seed(42)
    nodes = [f"ACC_{i:04d}" for i in range(2000)]
    channels = ["ACH", "Cheque", "Credit Card", "Cash", "Reinvestment", "Wire"]
    edges = []
    
    for _ in range(10000):
        s, t = np.random.choice(nodes, 2, replace=False)
        edges.append({
            "source": s, 
            "target": t, 
            "amount": float(np.random.lognormal(mean=8, sigma=2)),
            "payment_channel": np.random.choice(channels), 
            "time_since_last_transaction": float(np.random.exponential(3600)),
            "velocity_score": int(np.random.exponential(100)), 
            "spending_deviation_score": float(np.random.normal(0, 50)),
            "hour": int(np.random.randint(0, 24)), 
            "day_of_week": int(np.random.randint(0, 7)),
            "is_first_transaction": int(np.random.random() < 0.1), 
            "is_new_receiver": int(np.random.random() < 0.25),
            "is_new_bank": int(np.random.random() < 0.15), 
            "is_new_payment_format": int(np.random.random() < 0.1),
            "is_fraud": 1 if np.random.random() < 0.03 else 0,
            "is_cross_bank_transfer": int(np.random.random() < 0.3), 
            "is_cross_currency_transfer": int(np.random.random() < 0.2)
        })
        
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    
    for e in edges:
        G.add_edge(
            e["source"], 
            e["target"], 
            **{k: v for k, v in e.items() if k not in ("source", "target")}
        )
        
    nx.set_node_attributes(G, dict(G.degree()), "degree")
    
    fraud_nodes = set()
    for u, v, d in G.edges(data=True):
        if d.get("is_fraud", 0) == 1:
            fraud_nodes.add(u)
            fraud_nodes.add(v)
            
    nx.set_node_attributes(G, {n: (1 if n in fraud_nodes else 0) for n in G.nodes}, "is_fraud")
    
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    
    return G, G, edges


# Initiate the Global Application Sequence
build_app()