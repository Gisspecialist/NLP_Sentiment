import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import pandas as pd
import networkx as nx


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

@dataclass
class GraphConfig:
    # Inputs
    messages_enriched_path: str = "outputs_simple/messages_enriched.csv"
    original_messages_path: str = "messages.csv"

    # Column names
    id_col: str = "id"
    time_col: str = "timestamp"
    actor_col: str = "actor"
    source_col: str = "source"
    topic_col: str = "topic_id"

    # New attributes
    location_col: str = "location"
    program_col: str = "program"
    project_col: str = "project_id"

    # Time binning
    time_freq: str = "M"  # "D"=day, "W"=week, "M"=month...

    # Output folder
    out_dir: str = "kg_outputs"


# ---------------------------------------------------------
# LOADING & ENRICHING
# ---------------------------------------------------------

def load_messages_enriched(cfg: GraphConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.messages_enriched_path)
    # parse timestamps
    df[cfg.time_col] = pd.to_datetime(df[cfg.time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[cfg.time_col])
    return df


def ensure_actor_column(df: pd.DataFrame, cfg: GraphConfig) -> pd.DataFrame:
    """
    Ensure we have an 'actor' column; fall back to `source`.
    """
    df = df.copy()
    if cfg.actor_col in df.columns:
        # if actor is all NaN, fall back to source
        if df[cfg.actor_col].notna().sum() == 0 and cfg.source_col in df.columns:
            df[cfg.actor_col] = df[cfg.source_col].astype(str)
        return df

    if cfg.source_col in df.columns:
        df[cfg.actor_col] = df[cfg.source_col].astype(str)
    else:
        df[cfg.actor_col] = "unknown"
    return df


def add_time_bin(df: pd.DataFrame, cfg: GraphConfig) -> pd.DataFrame:
    """
    Create a time-binned column (e.g., monthly) used for time nodes in the graph.
    """
    df = df.copy()
    if df[cfg.time_col].dt.tz is not None:
        df[cfg.time_col] = df[cfg.time_col].dt.tz_convert(None)
    period = df[cfg.time_col].dt.to_period(cfg.time_freq)
    df["time_bin"] = period.astype(str)
    return df


def enrich_original_messages(enriched: pd.DataFrame, cfg: GraphConfig) -> Optional[pd.DataFrame]:
    """
    Merge enriched info (sentiment, topics, etc.) back into original messages.csv.
    """
    orig_path = Path(cfg.original_messages_path)
    if not orig_path.exists():
        print(f"[INFO] Original messages file not found at {orig_path}, skipping enrichment.")
        return None

    original = pd.read_csv(orig_path)

    # Columns to bring over from enriched if they exist
    cols_to_add = []
    for col in [
        "clean_text",
        "sentiment_label",
        "sentiment_signed",
        "sentiment_compound",
        cfg.topic_col,
        "time_bin",
    ]:
        if col in enriched.columns:
            cols_to_add.append(col)

    if cfg.id_col not in original.columns or cfg.id_col not in enriched.columns:
        print(f"[WARN] '{cfg.id_col}' column missing in original or enriched; cannot merge by id.")
        return None

    original = original.copy()
    enriched = enriched.copy()

    # Align id types as strings
    original[cfg.id_col] = original[cfg.id_col].astype(str)
    enriched[cfg.id_col] = enriched[cfg.id_col].astype(str)

    enriched_for_merge = enriched[[cfg.id_col, cfg.time_col] + cols_to_add]

    merged = original.merge(
        enriched_for_merge,
        on=cfg.id_col,
        how="left",
        suffixes=("", "_enriched"),
    )

    out_path = orig_path.parent / "messages_enriched_joined.csv"
    merged.to_csv(out_path, index=False)
    print(f"[OK] Enriched original messages written to: {out_path.resolve()}")
    return merged


# ---------------------------------------------------------
# GRAPH BUILDING
# ---------------------------------------------------------

def build_actor_theme_time_graph(df: pd.DataFrame, cfg: GraphConfig) -> nx.Graph:
    """
    Build an undirected graph with nodes for:
      - actors
      - topics
      - time bins
      - locations
      - programs
      - projects

    Edges:
      - actor -- topic     (who talks about what)
      - topic -- time      (when topics appear)
      - actor -- time      (actor activity over time)
      - location -- topic  (where issues occur)
      - program -- topic   (which program associated with which issues)
      - project -- topic   (which project associated with which issues)
      - project -- time    (when a project is discussed)
      - location -- time   (when a location is mentioned)
    """
    G = nx.Graph()

    # Helper functions for node IDs
    def actor_node_id(v: str) -> str:
        return f"actor:{v}"

    def topic_node_id(v) -> str:
        return f"topic:{v}"

    def time_node_id(v: str) -> str:
        return f"time:{v}"

    def location_node_id(v: str) -> str:
        return f"location:{v}"

    def program_node_id(v: str) -> str:
        return f"program:{v}"

    def project_node_id(v: str) -> str:
        return f"project:{v}"

    # ----- NODES -----

    # Actors
    if cfg.actor_col in df.columns:
        for a in df[cfg.actor_col].dropna().unique():
            G.add_node(actor_node_id(a), type="actor", label=str(a))

    # Topics
    if cfg.topic_col in df.columns:
        for t in df[cfg.topic_col].dropna().unique():
            G.add_node(topic_node_id(t), type="topic", label=f"Topic {t}")

    # Time bins
    for tb in df["time_bin"].dropna().unique():
        G.add_node(time_node_id(tb), type="time", label=str(tb))

    # Locations
    if cfg.location_col in df.columns:
        for loc in df[cfg.location_col].dropna().unique():
            G.add_node(location_node_id(loc), type="location", label=str(loc))

    # Programs
    if cfg.program_col in df.columns:
        for prog in df[cfg.program_col].dropna().unique():
            G.add_node(program_node_id(prog), type="program", label=str(prog))

    # Projects
    if cfg.project_col in df.columns:
        for pid in df[cfg.project_col].dropna().unique():
            G.add_node(project_node_id(pid), type="project", label=str(pid))

    # ----- EDGES -----

    # 1) actor -- topic
    if cfg.actor_col in df.columns and cfg.topic_col in df.columns:
        at_group = (
            df.groupby([cfg.actor_col, cfg.topic_col])
              .size()
              .reset_index(name="weight")
        )
        for _, row in at_group.iterrows():
            a = actor_node_id(row[cfg.actor_col])
            t = topic_node_id(row[cfg.topic_col])
            w = int(row["weight"])
            if w > 0:
                G.add_edge(a, t, weight=w, edge_type="actor-topic")

    # 2) topic -- time
    if cfg.topic_col in df.columns:
        tt_group = (
            df.groupby([cfg.topic_col, "time_bin"])
              .size()
              .reset_index(name="weight")
        )
        for _, row in tt_group.iterrows():
            t = topic_node_id(row[cfg.topic_col])
            tb = time_node_id(row["time_bin"])
            w = int(row["weight"])
            if w > 0:
                G.add_edge(t, tb, weight=w, edge_type="topic-time")

    # 3) actor -- time
    if cfg.actor_col in df.columns:
        atime_group = (
            df.groupby([cfg.actor_col, "time_bin"])
              .size()
              .reset_index(name="weight")
        )
        for _, row in atime_group.iterrows():
            a = actor_node_id(row[cfg.actor_col])
            tb = time_node_id(row["time_bin"])
            w = int(row["weight"])
            if w > 0:
                if G.has_edge(a, tb):
                    existing = G[a][tb]
                    existing["weight"] = existing.get("weight", 0) + w
                    etype = existing.get("edge_type", "actor-time")
                    if isinstance(etype, str):
                        etype = {etype}
                    etype.add("actor-time")
                    existing["edge_type"] = ";".join(sorted(etype))
                else:
                    G.add_edge(a, tb, weight=w, edge_type="actor-time")

    # 4) location -- topic
    if cfg.location_col in df.columns and cfg.topic_col in df.columns:
        lt_group = (
            df.groupby([cfg.location_col, cfg.topic_col])
              .size()
              .reset_index(name="weight")
        )
        for _, row in lt_group.iterrows():
            loc = location_node_id(row[cfg.location_col])
            t = topic_node_id(row[cfg.topic_col])
            w = int(row["weight"])
            if w > 0:
                G.add_edge(loc, t, weight=w, edge_type="location-topic")

    # 5) program -- topic
    if cfg.program_col in df.columns and cfg.topic_col in df.columns:
        pt_group = (
            df.groupby([cfg.program_col, cfg.topic_col])
              .size()
              .reset_index(name="weight")
        )
        for _, row in pt_group.iterrows():
            prog = program_node_id(row[cfg.program_col])
            t = topic_node_id(row[cfg.topic_col])
            w = int(row["weight"])
            if w > 0:
                G.add_edge(prog, t, weight=w, edge_type="program-topic")

    # 6) project -- topic
    if cfg.project_col in df.columns and cfg.topic_col in df.columns:
        proj_topic_group = (
            df.groupby([cfg.project_col, cfg.topic_col])
              .size()
              .reset_index(name="weight")
        )
        for _, row in proj_topic_group.iterrows():
            proj = project_node_id(row[cfg.project_col])
            t = topic_node_id(row[cfg.topic_col])
            w = int(row["weight"])
            if w > 0:
                G.add_edge(proj, t, weight=w, edge_type="project-topic")

    # 7) project -- time
    if cfg.project_col in df.columns:
        proj_time_group = (
            df.groupby([cfg.project_col, "time_bin"])
              .size()
              .reset_index(name="weight")
        )
        for _, row in proj_time_group.iterrows():
            proj = project_node_id(row[cfg.project_col])
            tb = time_node_id(row["time_bin"])
            w = int(row["weight"])
            if w > 0:
                G.add_edge(proj, tb, weight=w, edge_type="project-time")

    # 8) location -- time
    if cfg.location_col in df.columns:
        loc_time_group = (
            df.groupby([cfg.location_col, "time_bin"])
              .size()
              .reset_index(name="weight")
        )
        for _, row in loc_time_group.iterrows():
            loc = location_node_id(row[cfg.location_col])
            tb = time_node_id(row["time_bin"])
            w = int(row["weight"])
            if w > 0:
                G.add_edge(loc, tb, weight=w, edge_type="location-time")

    return G


# ---------------------------------------------------------
# EXPORTS
# ---------------------------------------------------------

def export_graph(G: nx.Graph, cfg: GraphConfig):
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(exist_ok=True)

    # Nodes table
    node_records = []
    for node_id, data in G.nodes(data=True):
        rec = {"node_id": node_id}
        rec.update(data)
        node_records.append(rec)
    nodes_df = pd.DataFrame(node_records)
    nodes_path = out_dir / "kg_nodes.csv"
    nodes_df.to_csv(nodes_path, index=False)

    # Edges table
    edge_records = []
    for u, v, data in G.edges(data=True):
        rec = {"source": u, "target": v}
        rec.update(data)
        edge_records.append(rec)
    edges_df = pd.DataFrame(edge_records)
    edges_path = out_dir / "kg_edges.csv"
    edges_df.to_csv(edges_path, index=False)

    # GraphML
    graphml_path = out_dir / "actor_theme_time.graphml"
    nx.write_graphml(G, graphml_path)

    print(f"[OK] Nodes written to:   {nodes_path.resolve()}")
    print(f"[OK] Edges written to:   {edges_path.resolve()}")
    print(f"[OK] GraphML written to: {graphml_path.resolve()}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    cfg = GraphConfig(
        messages_enriched_path="outputs_simple/messages_enriched.csv",
        original_messages_path="messages.csv",
        time_freq="M",  # change to "W" or "D" if you want different time bins
    )

    print("[1/4] Loading messages_enriched...")
    df = load_messages_enriched(cfg)

    print("[2/4] Ensuring actor & time_bin columns...")
    df = ensure_actor_column(df, cfg)
    df = add_time_bin(df, cfg)

    print("[3/4] Enriching original messages.csv (if present)...")
    enrich_original_messages(df, cfg)

    print("[4/4] Building extended actor–theme–time graph...")
    G = build_actor_theme_time_graph(df, cfg)
    export_graph(G, cfg)

    print("[DONE] Knowledge graph build complete.")


if __name__ == "__main__":
    main()
