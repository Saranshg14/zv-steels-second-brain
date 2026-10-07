"""Flask web UI for ZV Steels Second Brain."""
import sys, os
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory

# Fix Windows console Unicode
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add the parent directory to path so we can import brain.py
sys.path.insert(0, str(Path(__file__).parent))
import brain

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()
    role = data.get("role", "employee")
    company = data.get("company") or None
    if not question:
        return jsonify({"error": "No question provided"}), 400
    try:
        result = brain.answer(question, role, company)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ingest", methods=["POST"])
def ingest():
    try:
        # Capture printed output
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            brain.ingest()
        output = buf.getvalue().strip()
        return jsonify({"status": "ok", "message": output})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/status", methods=["GET"])
def status():
    try:
        brain_col, _ = brain.db()
        count = brain_col.count()
        return jsonify({"chunks": count, "data_dir": str(brain.DATA_DIR), "model": brain.GROK_MODEL})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🧠 ZV Steels Brain UI → http://localhost:5000")
    app.run(debug=False, port=5000)
