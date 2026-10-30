from .config import settings

TOOL_SPECS = [
    {
        "name": "search_documents",
        "description": "Semantic search over the uploaded documents. Returns the most relevant passages "
                       "with source and location. Use for any question about document content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Focused search query"},
                "top_k": {"type": "integer", "description": "Passages to return (1-8)", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_documents",
        "description": "List uploaded files and available tables.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "analyze_table",
        "description": "Safe analysis of an uploaded CSV/Excel table: schema, head, describe, or "
                       "aggregate (sum/mean/min/max/count, optionally grouped). Use for any numeric "
                       "question; never compute from guessed values.",
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {"type": "string", "description": "Table name from list_documents"},
                "operation": {"type": "string", "enum": ["schema", "head", "describe", "aggregate"]},
                "column": {"type": "string"},
                "agg": {"type": "string", "enum": ["sum", "mean", "min", "max", "count"]},
                "group_by": {"type": "string"},
            },
            "required": ["table", "operation"],
        },
    },
]


class ToolBox:
    """Executes tools for one question and records evidence + sources for the Validator."""

    def __init__(self, store, tables: dict):
        self.store, self.tables = store, tables
        self.evidence: list[str] = []
        self.sources: set[str] = set()

    def run(self, name: str, args: dict) -> str:
        try:
            if name == "search_documents":
                out = self._search(**args)
            elif name == "list_documents":
                out = f"Tables: {list(self.tables)}"
            elif name == "analyze_table":
                out = self._analyze(**args)
            else:
                out = f"Unknown tool: {name}"
        except Exception as e:  # tool errors go back to the model, not the user
            out = f"Tool error: {e}"
        return out[:6000]

    def _search(self, query: str, top_k: int = 5) -> str:
        top_k = max(1, min(int(top_k), 8))
        hits = [h for h in self.store.search(query, top_k) if h["distance"] <= settings.max_distance]
        if not hits:
            return "No relevant passages found."
        parts = []
        for h in hits:
            label = f'{h["metadata"]["source"]}, {h["metadata"]["location"]}'
            self.sources.add(label)
            self.evidence.append(f"[{label}] {h['text']}")
            parts.append(f"[{label}] (similarity {1 - h['distance']:.2f})\n{h['text']}")
        return "\n\n---\n\n".join(parts)

    def _analyze(self, table: str, operation: str, column: str | None = None,
                 agg: str | None = None, group_by: str | None = None) -> str:
        df = self.tables.get(table)
        if df is None:
            return f"Unknown table '{table}'. Available: {list(self.tables)}"
        if operation == "schema":
            out = f"rows={len(df)}\n{df.dtypes.to_string()}"
        elif operation == "head":
            out = df.head(10).to_string()
        elif operation == "describe":
            out = df.describe(include="all").T.to_string()
        elif operation == "aggregate":
            if column not in df.columns or agg not in {"sum", "mean", "min", "max", "count"}:
                return f"Need a valid column {list(df.columns)} and agg."
            if group_by:
                if group_by not in df.columns:
                    return f"Unknown group_by column. Columns: {list(df.columns)}"
                res = df.groupby(group_by)[column].agg(agg).sort_values(ascending=False).head(25)
                out = res.to_string()
            else:
                out = f"{agg}({column}) = {getattr(df[column], agg)()}"
        else:
            return "Unknown operation."
        self.sources.add(f"{table} (table analysis)")
        self.evidence.append(f"[{table}:{operation}] {out}")
        return out
