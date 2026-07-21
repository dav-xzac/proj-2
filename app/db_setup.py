import sqlite3
from pathlib import Path
import tempfile
import openpyxl

# /data if bucket is mounted for persistent database
DB_DIR = Path("/data" if Path("/data").exists() else "/tmp")
DB_PATH = DB_DIR / "predictions.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
                     id         INTEGER PRIMARY KEY AUTOINCREMENT,
                     ts         TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                     text       TEXT    NOT NULL,
                     label      TEXT    NOT NULL,
                     confidence REAL    NOT NULL,
                     text_len   INTEGER NOT NULL
                     )
                """)
        conn.commit()


def log_prediction(text, label, confidence):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO predictions (text, label, confidence, text_len) VALUES (?,?,?,?)",
            (text, label,round(confidence, 4), len(text))
        )
        conn.commit()


def export_to_excel(from_date,to_date,label_filter):
    query = "SELECT ts, text, label, confidence FROM predictions WHERE 1=1"
    params = []
    # add query filter and param for it for the following execution
    if label_filter and label_filter != "all":
        query += " AND label = ?"
        params.append(label_filter)
    if from_date:
        query += " AND date(ts) >= ?"
        params.append(from_date)
    if to_date:
        query += " AND date(ts) <= ?"
        params.append(to_date)
    query += " ORDER BY ts DESC"

    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()

    if not rows:
        return None
    
    # Create excel workbook and populate with the filtered data
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sentiment Report "
    if from_date:
        ws.title += f"from {from_date}"
    if to_date:
        ws.title += f"from {to_date}"
    ws.append(["Date", "Time", "Text", "Label", "Confidence"])
    for r in rows:
        # ts sliced between date and time
        ws.append([r["ts"][:10], r["ts"][11:19], r["text"], r["label"], r["confidence"]])
        ws.column_dimensions["A"].width = 12
        ws.column_dimensions["B"].width = 10
        ws.column_dimensions["C"].width = 60
        ws.column_dimensions["D"].width = 12
        ws.column_dimensions["E"].width = 12
    # return the file path string to gradio which send it to downloadable output 
    tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete = False)
    wb.save(tmp.name)
    return tmp.name
