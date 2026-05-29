from flask import Flask, request, render_template_string, redirect, url_for, session, flash, jsonify
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import sympy as sp
from functools import wraps
import random
import base64
import os
import json
import re

from handwriting_recognizer import recognize_handwriting_from_data_url


app = Flask(__name__)
app.secret_key = "change-this-to-a-random-secret"

DB_PATH = "mathsite.db"


# -------------------------
# Database helpers
# -------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


init_db()


# -------------------------
# Auth helpers
# -------------------------
def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapper


# -------------------------
# UI Template
# -------------------------
BASE_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{{ title }}</title>

  <script>
    window.MathJax = {
      tex: { inlineMath: [['\\(', '\\)'], ['$', '$']] }
    };
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

  <style>
    body {
      font-family: Arial, sans-serif;
      max-width: 980px;
      margin: 32px auto;
      padding: 0 16px;
    }

    .topbar {
      display:flex;
      justify-content:space-between;
      align-items:center;
      gap:12px;
      margin-bottom:14px;
    }

    .brand { font-weight:700; }
    .auth a { margin-left:10px; text-decoration:none; }

    .tabs {
      display: flex;
      gap: 10px;
      border-bottom: 1px solid #ddd;
      padding-bottom: 10px;
      margin-bottom: 18px;
    }

    .tab {
      text-decoration: none;
      padding: 10px 12px;
      border-radius: 10px;
      color: #111;
    }

    .tab:hover { background: #f3f3f3; }
    .active { background: #111; color: #fff; }

    .card {
      border: 1px solid #e5e5e5;
      border-radius: 14px;
      padding: 16px;
    }

    input {
      width: 100%;
      padding: 10px;
      margin: 6px 0 12px;
      border: 1px solid #ddd;
      border-radius: 10px;
      box-sizing: border-box;
    }

    button {
      padding: 10px 14px;
      border: 0;
      border-radius: 10px;
      background: #111;
      color: #fff;
      cursor: pointer;
    }

    button:hover { opacity: 0.9; }
    .muted { color: #666; }

    .flash {
      background:#fff4cc;
      border:1px solid #ffe08a;
      padding:10px;
      border-radius:10px;
      margin: 12px 0;
    }

    .center {
      max-width: 520px;
      margin: 40px auto;
    }

    input[type="checkbox"], input[type="radio"] {
      width:auto !important;
      margin:0 !important;
      padding:0 !important;
    }

    .draw-wrap {
      border: 1px solid #e5e5e5;
      border-radius: 14px;
      padding: 12px;
      margin-top: 10px;
    }

    .draw-canvas {
      width: 100%;
      height: 220px;
      border: 1px solid #ddd;
      border-radius: 14px;
      background: #fff;
      touch-action: none;
      display: block;
    }

    .draw-actions {
      display:flex;
      gap:10px;
      flex-wrap:wrap;
      margin-top:10px;
    }

    .btn-lite { background:#333; }


    .type-stats-box {
      position: fixed;
      top: 18px;
      right: 22px;
      background: white;
      border: 1px solid #ddd;
      border-radius: 14px;
      padding: 12px 16px;
      box-shadow: 0 4px 14px rgba(0,0,0,0.12);
      font-size: 15px;
      line-height: 1.5;
      z-index: 1000;
      min-width: 165px;
    }

    .type-stats-box-title {
      font-weight: 700;
      margin-bottom: 4px;
    }

    .small-reset-btn {
      margin-top: 6px;
      padding: 5px 10px;
      border: none;
      border-radius: 8px;
      background: #111;
      color: white;
      cursor: pointer;
      font-size: 13px;
    }

    .small-reset-btn:hover {
      background: #333;
    }

  </style>
</head>

<body>
  <div class="topbar">
    <div class="brand">UCZAcademy</div>
    <div class="auth">
      {% if session.get("user_id") %}
        <span class="muted">Hi, {{ session.get("username") }}</span>
        <a href="{{ url_for('logout') }}">Logout</a>
      {% else %}
        <a href="{{ url_for('login') }}">Login</a>
        <a href="{{ url_for('signup') }}">Create Account</a>
      {% endif %}
    </div>
  </div>

  {% with messages = get_flashed_messages() %}
    {% if messages %}
      {% for m in messages %}
        <div class="flash">{{ m }}</div>
      {% endfor %}
    {% endif %}
  {% endwith %}

  {% if session.get("user_id") %}
    <div class="tabs">
      <a class="tab {{ 'active' if active=='algebra' else '' }}" href="{{ url_for('algebra') }}">Algebra Foundation</a>
      <a class="tab {{ 'active' if active=='geometry' else '' }}" href="{{ url_for('geometry') }}">Geometry</a>
      <a class="tab {{ 'active' if active=='trig' else '' }}" href="{{ url_for('trig') }}">Trigonometry</a>
      <a class="tab {{ 'active' if active=='calculus' else '' }}" href="{{ url_for('calculus') }}">Calculus</a>
    </div>
  {% endif %}

  <h2>{{ title }}</h2>
  <p class="muted">{{ subtitle }}</p>

  <div class="card">
    {{ content|safe }}
  </div>

<script>
window.canvasApis = window.canvasApis || {};

function initOneCanvas(canvas) {
  const ctx = canvas.getContext("2d");
  let strokes = [];
  let currentStroke = [];
  let drawing = false;
  let eraserMode = false;

  function fillWhiteBackground() {
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.restore();
  }

  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    canvas.width = Math.floor(rect.width * dpr);
    canvas.height = Math.floor(rect.height * dpr);

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);

    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = 4;
    ctx.strokeStyle = "#000";

    fillWhiteBackground();
  }

  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);

  function getMousePos(e) {
    const rect = canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top, t: Date.now() };
  }

  function getTouchPos(e) {
    const rect = canvas.getBoundingClientRect();
    const touch = e.touches[0] || e.changedTouches[0];
    return { x: touch.clientX - rect.left, y: touch.clientY - rect.top, t: Date.now() };
  }

  function startDraw(p) {
    drawing = true;
    currentStroke = [p];
    ctx.beginPath();
    ctx.moveTo(p.x, p.y);
  }

  function moveDraw(p) {
    if (!drawing) return;
    currentStroke.push(p);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
  }

  function endDraw() {
    if (!drawing) return;
    drawing = false;

    if (currentStroke.length > 0) {
      strokes.push(currentStroke);
    }

    currentStroke = [];
  }

  canvas.addEventListener("mousedown", function(e) {
    if (eraserMode) return;
    e.preventDefault();
    startDraw(getMousePos(e));
  });

  canvas.addEventListener("mousemove", function(e) {
    if (eraserMode) return;
    e.preventDefault();
    moveDraw(getMousePos(e));
  });

  canvas.addEventListener("mouseup", function(e) {
    e.preventDefault();
    endDraw();
  });

  canvas.addEventListener("mouseleave", function() {
    endDraw();
  });

  canvas.addEventListener("touchstart", function(e) {
    e.preventDefault();
    if (eraserMode) {
      eraseAtPoint(getTouchPos(e));
      return;
    }
    startDraw(getTouchPos(e));
  }, { passive: false });

  canvas.addEventListener("touchmove", function(e) {
    e.preventDefault();
    if (eraserMode) {
      eraseAtPoint(getTouchPos(e));
      return;
    }
    moveDraw(getTouchPos(e));
  }, { passive: false });

  canvas.addEventListener("touchend", function(e) {
    e.preventDefault();
    endDraw();
  }, { passive: false });

  function drawStroke(stroke) {
    if (!stroke || stroke.length === 0) return;
    ctx.beginPath();
    ctx.moveTo(stroke[0].x, stroke[0].y);
    for (let i = 1; i < stroke.length; i++) {
      ctx.lineTo(stroke[i].x, stroke[i].y);
    }
    ctx.stroke();
  }

  function redrawAll() {
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.restore();

    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = 4;
    ctx.strokeStyle = "#000";

    strokes.forEach(drawStroke);
  }

  function distPointToSegment(px, py, ax, ay, bx, by) {
    const dx = bx - ax;
    const dy = by - ay;
    if (dx === 0 && dy === 0) {
      return Math.hypot(px - ax, py - ay);
    }
    let t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy);
    t = Math.max(0, Math.min(1, t));
    const x = ax + t * dx;
    const y = ay + t * dy;
    return Math.hypot(px - x, py - y);
  }

  function strokeNearPoint(stroke, p, radius) {
    if (!stroke || stroke.length === 0) return false;
    if (stroke.length === 1) {
      return Math.hypot(stroke[0].x - p.x, stroke[0].y - p.y) <= radius;
    }
    for (let i = 1; i < stroke.length; i++) {
      const a = stroke[i - 1];
      const b = stroke[i];
      if (distPointToSegment(p.x, p.y, a.x, a.y, b.x, b.y) <= radius) {
        return true;
      }
    }
    return false;
  }

  function eraseAtPoint(p) {
    const before = strokes.length;
    strokes = strokes.filter(function(stroke) {
      return !strokeNearPoint(stroke, p, 18);
    });
    if (strokes.length !== before) {
      redrawAll();
    }
  }

  canvas.addEventListener("pointerdown", function(e) {
    if (!eraserMode) return;
    e.preventDefault();
    eraseAtPoint(getMousePos(e));
  });

  canvas.addEventListener("pointermove", function(e) {
    if (!eraserMode || e.buttons !== 1) return;
    e.preventDefault();
    eraseAtPoint(getMousePos(e));
  });

  return {
    clear: function() {
      ctx.save();
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.restore();

      strokes = [];
      currentStroke = [];
    },

    undo: function() {
      if (strokes.length > 0) {
        strokes.pop();
        redrawAll();
      }
    },

    toggleEraser: function(button) {
      eraserMode = !eraserMode;
      canvas.style.cursor = eraserMode ? "crosshair" : "default";
      if (button) {
        button.textContent = eraserMode ? "Eraser On" : "Eraser";
        button.style.background = eraserMode ? "#b00020" : "#333";
      }
      return eraserMode;
    },

    toDataURL: function() {
      const exportCanvas = document.createElement("canvas");
      exportCanvas.width = canvas.width;
      exportCanvas.height = canvas.height;

      const exportCtx = exportCanvas.getContext("2d");
      exportCtx.fillStyle = "#ffffff";
      exportCtx.fillRect(0, 0, exportCanvas.width, exportCanvas.height);
      exportCtx.drawImage(canvas, 0, 0);

      return exportCanvas.toDataURL("image/png");
    },

    getStrokes: function() {
      return strokes;
    }
  };
}

function initDrawingUI() {
  document.querySelectorAll("canvas.draw-canvas").forEach(function(canvas) {
    const api = initOneCanvas(canvas);
    const id = canvas.dataset.canvasId;
    window.canvasApis[id] = api;

    const clearBtn = document.querySelector(`[data-action="clear"][data-canvas-id="${id}"]`);
    const undoBtn = document.querySelector(`[data-action="undo"][data-canvas-id="${id}"]`);
    const eraserBtn = document.querySelector(`[data-action="eraser"][data-canvas-id="${id}"]`);
    const saveBtn = document.querySelector(`[data-action="save"][data-canvas-id="${id}"]`);
    const recogBtn = document.querySelector(`[data-action="recognize"][data-canvas-id="${id}"]`);
    const saveStrokeBtn = document.querySelector(`[data-action="save-stroke-sample"][data-canvas-id="${id}"]`);

    const hidden = document.querySelector(`input[type="hidden"][data-canvas-id="${id}"]`);
    const targetInput = document.querySelector(`input[type="text"][data-canvas-id="${id}"]`);
    const sampleLabel = document.querySelector(`[data-role="sample-label"][data-canvas-id="${id}"]`);
    const sampleStatus = document.querySelector(`[data-role="sample-status"][data-canvas-id="${id}"]`);

    if (clearBtn) {
      clearBtn.addEventListener("click", function(e) {
        e.preventDefault();
        api.clear();
      });
    }

    if (undoBtn) {
      undoBtn.addEventListener("click", function(e) {
        e.preventDefault();
        api.undo();
      });
    }

    if (eraserBtn) {
      eraserBtn.addEventListener("click", function(e) {
        e.preventDefault();
        api.toggleEraser(eraserBtn);
      });
    }

    if (saveBtn) {
      saveBtn.addEventListener("click", function(e) {
        e.preventDefault();
        if (hidden) hidden.value = api.toDataURL();
        alert("Drawing saved.");
      });
    }

    if (saveStrokeBtn) {
      saveStrokeBtn.addEventListener("click", async function(e) {
        e.preventDefault();
        e.stopPropagation();

        const label = sampleLabel ? sampleLabel.value.trim() : "";
        const strokes = api.getStrokes();

        if (!label) {
          alert("Choose a label first.");
          return;
        }

        if (!strokes || strokes.length === 0) {
          alert("Draw one symbol first.");
          return;
        }

        saveStrokeBtn.disabled = true;
        const oldText = saveStrokeBtn.textContent;
        saveStrokeBtn.textContent = "Saving...";

        try {
          const resp = await fetch("/api/save_stroke_sample", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              label: label,
              strokes: strokes
            })
          });

          const data = await resp.json();

          if (!resp.ok || !data.ok) {
            alert(data.error || "Save failed.");
            if (sampleStatus) sampleStatus.textContent = data.error || "Save failed.";
            return;
          }

          const msg = "Saved " + label + ": " + data.path;
          if (sampleStatus) sampleStatus.textContent = msg;
          alert(msg);

          api.clear();

        } catch (err) {
          console.error(err);
          alert("Save failed.");
          if (sampleStatus) sampleStatus.textContent = "Save failed.";
        } finally {
          saveStrokeBtn.disabled = false;
          saveStrokeBtn.textContent = oldText;
        }
      });
    }

    if (recogBtn) {
      recogBtn.addEventListener("click", async function(e) {
        e.preventDefault();
        e.stopPropagation();

        recogBtn.disabled = true;
        const oldText = recogBtn.textContent;
        recogBtn.textContent = "Recognizing...";

        const png = api.toDataURL();
        if (hidden) hidden.value = png;

        try {
          const resp = await fetch("/api/recognize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              image_data_url: png,
              strokes: api.getStrokes()
            })
          });

          const data = await resp.json();
          console.log("Recognition response:", data);

          if (!resp.ok) {
            alert(data.error || "Recognition failed.");
            return;
          }

          if (data && data.text && targetInput) {
            targetInput.value = data.text;

            let msg = "Prediction: " + data.text;

            if (data.segments) {
              msg += "\n\nSegments:";
              data.segments.forEach(function(s) {
                msg += "\n" + s.index + ". " + s.label + " → " + s.text + " | confidence: " + Number(s.confidence).toFixed(4);
              });
            } else {
              if (data.label !== undefined) {
                msg += "\nLabel: " + data.label;
              }

              if (data.confidence !== undefined) {
                msg += "\nConfidence: " + Number(data.confidence).toFixed(4);
              }

              if (data.top_predictions) {
                msg += "\n\nTop predictions:";
                data.top_predictions.forEach(function(p) {
                  msg += "\n" + p.label + " → " + p.text + ": " + Number(p.confidence).toFixed(4);
                });
              }
            }

            alert(msg);
          } else {
            alert(data.error || "No recognition result.");
          }
        } catch (err) {
          console.error(err);
          alert("Recognition failed.");
        } finally {
          recogBtn.disabled = false;
          recogBtn.textContent = oldText;
        }
      });
    }
  });
}



function getProblemTypeKey() {
  if (window.pemdasTypeKey) {
    return window.pemdasTypeKey;
  }

  const stepSelect = document.querySelector("select[name='operations']");
  const steps = stepSelect ? stepSelect.value : "unknown";

  const neg = document.querySelector("input[name='neg']");
  const exp = document.querySelector("input[name='exp']");
  const abs = document.querySelector("input[name='abs']");
  const frac = document.querySelector("input[name='frac']");

  const n = neg && neg.checked ? "N1" : "N0";
  const e = exp && exp.checked ? "E1" : "E0";
  const a = abs && abs.checked ? "A1" : "A0";
  const f = frac && frac.checked ? "F1" : "F0";

  return `pemdas_steps_type_${steps}_${n}_${e}_${a}_${f}`;
}

function getCurrentProblemKey() {
  if (window.pemdasProblemKey) {
    return window.pemdasProblemKey;
  }

  const problemEl = document.getElementById("currentProblemText");
  if (problemEl) {
    return problemEl.innerText.trim();
  }

  const allText = document.body.innerText || "";
  const match = allText.match(/Problem:\s*(.+)/);
  if (match) {
    return match[1].trim().split("\n")[0];
  }

  return "unknown_problem";
}

function getTypeStats() {
  const key = getProblemTypeKey();
  const raw = localStorage.getItem(key);

  if (!raw) {
    return {
      done: 0,
      correct: 0,
      incorrect: 0,
      problemResults: {}
    };
  }

  try {
    const stats = JSON.parse(raw);
    stats.done = stats.done || 0;
    stats.correct = stats.correct || 0;
    stats.incorrect = stats.incorrect || 0;
    stats.problemResults = stats.problemResults || {};
    return stats;
  } catch (e) {
    return {
      done: 0,
      correct: 0,
      incorrect: 0,
      problemResults: {}
    };
  }
}

function saveTypeStats(stats) {
  localStorage.setItem(getProblemTypeKey(), JSON.stringify(stats));
}

function updateTypeStatsDisplay() {
  const box = document.getElementById("typeStatsBox");
  if (!box) return;

  const stats = getTypeStats();

  const doneEl = document.getElementById("statDone");
  const correctEl = document.getElementById("statCorrect");
  const incorrectEl = document.getElementById("statIncorrect");

  if (doneEl) doneEl.textContent = stats.done;
  if (correctEl) correctEl.textContent = stats.correct;
  if (incorrectEl) incorrectEl.textContent = stats.incorrect;
}

function recordProblemResult(isCorrect) {
  const stats = getTypeStats();
  const problemKey = getCurrentProblemKey();
  const oldResult = stats.problemResults[problemKey];

  if (!oldResult) {
    stats.done += 1;

    if (isCorrect) {
      stats.correct += 1;
      stats.problemResults[problemKey] = "correct";
    } else {
      stats.incorrect += 1;
      stats.problemResults[problemKey] = "incorrect";
    }
  } else if (oldResult === "incorrect" && isCorrect) {
    stats.incorrect = Math.max(0, stats.incorrect - 1);
    stats.correct += 1;
    stats.problemResults[problemKey] = "correct";
  }

  saveTypeStats(stats);
  updateTypeStatsDisplay();
}

function isLastStepBox(box) {
  const boxes = Array.from(document.querySelectorAll(".step-box"));
  return boxes.length > 0 && boxes[boxes.length - 1] === box;
}

function isPlainFinalNumberAnswer(answer) {
  let s = String(answer || "").trim();
  s = s.replace(/−/g, "-");

  // Allow final answer like (-7), but not expressions like (-7+4)*4.
  if (/^\(\s*-?\d+(?:\.\d+)?\s*\)$/.test(s)) {
    s = s.replace(/^\(\s*/, "").replace(/\s*\)$/, "");
  }

  // A final-number stat should be a single number only, not an expression or equation.
  // Examples counted: 78, -12, 1.9, -19/10
  // Examples not counted: 56*3+5, (1+12)*6, x=78, 78+0
  return /^-?(?:\d+(?:\.\d+)?|\.\d+)(?:\/-?(?:\d+(?:\.\d+)?|\.\d+))?$/.test(s);
}

function shouldRecordTypeStats(box, answer, isCorrect) {
  if (!isLastStepBox(box)) {
    return false;
  }

  // Only count a correct problem when the LAST step is a plain final number.
  if (isCorrect) {
    return isPlainFinalNumberAnswer(answer);
  }

  // Count incorrect only when the student is attempting the LAST step as a final number.
  // This prevents intermediate equivalent expressions from affecting the type stats.
  return isPlainFinalNumberAnswer(answer);
}

function resetTypeStats() {
  if (!confirm("Reset stats for this exact problem type?")) {
    return;
  }

  localStorage.removeItem(getProblemTypeKey());
  updateTypeStatsDisplay();
}


async function checkStepAnswer(btn) {
  const box = btn.closest(".step-box");
  const input = box ? box.querySelector("input[type='text']") : null;
  const feedback = box ? box.querySelector(".step-feedback") : null;

  if (!box || !input || !feedback) {
    alert("Check answer setup is missing on this step.");
    return;
  }

  const stepIndex = parseInt(box.dataset.stepIndex || "0", 10);
  const answer = input.value.trim();

  if (!answer) {
    feedback.innerHTML = "<div style='color:#b00020; margin-top:10px;'>Please type an answer or use drawing recognition first.</div>";
    return;
  }

  const oldText = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Checking...";

  try {
    const resp = await fetch("/api/check_step", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        step_index: stepIndex,
        answer: answer
      })
    });

    const data = await resp.json();

    if (!resp.ok || !data.ok) {
      feedback.innerHTML = "<div style='color:#b00020; margin-top:10px;'>" + (data.message || data.error || "Check failed.") + "</div>";
      return;
    }

    if (data.correct) {
      const isLast = isLastStepBox(box);
      const isFinalNumber = isPlainFinalNumberAnswer(answer);

      // For the final step, do not accept an expression like 25+11.
      // The student must write the single final number, like 36.
      if (isLast && !isFinalNumber) {
        feedback.innerHTML = `
          <div style="color:#b00020; margin-top:10px; font-weight:700;">
            ❌ Almost correct, but final answer must be one number
          </div>
          <div style="margin-top:6px; line-height:1.45;">
            Your expression is equivalent to the answer, but on the last step please simplify it to a single final number.
          </div>
          <div style="margin-top:6px; color:#555;">
            Final answer: <b>${data.expected}</b>
          </div>
        `;
        return;
      }

      if (data.count_stats) {
        recordProblemResult(true);
      }

      feedback.innerHTML = `
        <div style="color:#0a7a2f; margin-top:10px; font-weight:700;">
          ✅ Correct
        </div>
      `;
    } else {
      if (data.count_stats) {
        recordProblemResult(false);
      }

      feedback.innerHTML = `
        <div style="color:#b00020; margin-top:10px; font-weight:700;">
          ❌ Not correct
        </div>
        <div style="margin-top:6px; line-height:1.45;">
          ${data.message}
        </div>
        <div style="margin-top:6px; color:#555;">
          Correct value for the whole problem: <b>${data.expected}</b>
        </div>
      `;
    }
  } catch (err) {
    console.error(err);
    feedback.innerHTML = "<div style='color:#b00020; margin-top:10px;'>Check failed.</div>";
  } finally {
    btn.disabled = false;
    btn.textContent = oldText;
  }
}

document.addEventListener("DOMContentLoaded", function() {
  initDrawingUI();
  updateTypeStatsDisplay();
});
</script>

</body>
</html>
"""


def render_page(title: str, subtitle: str, active: str, content_html: str):
    return render_template_string(
        BASE_HTML,
        title=title,
        subtitle=subtitle,
        active=active,
        content=content_html,
        session=session
    )


# -------------------------
# Math helpers
# -------------------------
def normalize_math_text(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    s = s.replace("×", "*").replace("÷", "/")
    return s


def value_from_expression_or_equation(user_text: str):
    t = normalize_math_text(user_text)

    if "=" in t:
        left, right = t.split("=", 1)
        left = sp.simplify(sp.sympify(left))
        right = sp.simplify(sp.sympify(right))

        if sp.simplify(left - right) != 0:
            raise ValueError("Equation is not true.")

        return sp.nsimplify(left)

    expr = sp.simplify(sp.sympify(t))
    return sp.nsimplify(expr)


def check_equivalence(student_str: str, correct_str: str):
    try:
        s = sp.simplify(sp.sympify(student_str))
        c = sp.simplify(sp.sympify(correct_str))
        ok = (sp.simplify(s - c) == 0)
        return ok, None
    except Exception as e:
        return False, str(e)


def checker_form(topic_hint: str):
    return f"""
    <form method="POST">
      <label>Student Answer <span class="muted">(example: (x+1)^2)</span></label>
      <input name="student" placeholder="Enter student answer..." />

      <label>Correct Answer <span class="muted">(example: x^2 + 2*x + 1)</span></label>
      <input name="correct" placeholder="Enter correct answer..." />

      <button type="submit">Check</button>
      <p class="muted" style="margin-top:10px;">Tip: {topic_hint}</p>
    </form>
    """


def _simple_int(allow_negative: bool):
    n = random.randint(2, 12)
    if allow_negative and random.random() < 0.35:
        n = -n
    return n


def _operand(allow_negative: bool, allow_abs: bool, allow_fraction: bool):
    if allow_fraction and random.random() < 0.30:
        num = random.randint(1, 12)
        den = random.randint(2, 12)

        if allow_negative and random.random() < 0.35:
            num = -num

        expr = rf"\frac{{{num}}}{{{den}}}"
        val = sp.Rational(num, den)

        if allow_abs and random.random() < 0.20:
            expr = rf"\left|{expr}\right|"
            val = abs(val)

        return expr, val

    a = random.randint(2, 12)
    b = random.randint(2, 12)

    if allow_negative and random.random() < 0.35:
        a = -a

    if allow_negative and random.random() < 0.25:
        b = -b

    kind = random.choice(["num", "add", "sub", "mul"])

    if kind == "num":
        expr, val = str(a), a
    elif kind == "add":
        expr, val = f"({a} + {b})", a + b
    elif kind == "sub":
        expr, val = f"({a} - {b})", a - b
    else:
        expr, val = f"({a} × {b})", a * b

    if allow_abs and random.random() < 0.25:
        expr, val = f"|{expr}|", abs(val)

    return expr, val


def generate_pemdas_problem(
    operations: int,
    allow_negative: bool,
    allow_exponents: bool,
    allow_abs: bool,
    allow_fraction: bool,
):
    expr, val = _operand(allow_negative, allow_abs, allow_fraction)

    ops = ["+", "-", "×", "÷"]

    if allow_exponents:
        ops.append("^")

    used_pow = False
    saw_fraction = False

    for _ in range(operations):
        op = random.choice(ops)

        if op in ["+", "-"]:
            op_expr, op_val = _operand(allow_negative, allow_abs, allow_fraction)
            expr = f"({expr} {op} {op_expr})"

            if op == "+":
                val = val + op_val
            else:
                val = val - op_val

        elif op == "×":
            m = random.randint(2, 10)

            if allow_negative and random.random() < 0.2:
                m = -m

            expr = f"({expr} × {m})"
            val = val * m

        elif op == "÷":
            d = random.randint(2, 10)

            if allow_fraction:
                expr = rf"\frac{{{expr}}}{{{d}}}"
                val = sp.Rational(val, d)
                saw_fraction = True
            else:
                k = random.randint(2, 6)
                expr = f"(({expr} × {d*k}) ÷ {d})"
                val = (val * (d * k)) // d

        elif op == "^":
            if used_pow:
                continue

            try:
                if abs(val) > 40:
                    continue
            except Exception:
                if abs(float(val)) > 40:
                    continue

            used_pow = True
            e = random.choice([2, 3])
            expr = f"({expr})^{e}"
            val = val ** e

        if allow_fraction and isinstance(val, sp.Rational) and val.q != 1:
            saw_fraction = True

        try:
            too_big = abs(val) > 10000
        except Exception:
            too_big = abs(float(val)) > 10000

        if too_big:
            return generate_pemdas_problem(
                operations,
                allow_negative,
                allow_exponents,
                allow_abs,
                allow_fraction
            )

    if allow_fraction and not saw_fraction:
        d = random.randint(2, 10)
        expr = rf"\frac{{{expr}}}{{{d}}}"
        val = sp.Rational(val, d)

    return expr, val


def generate_pemdas_set(
    operations: int,
    n: int,
    allow_negative: bool,
    allow_exponents: bool,
    allow_abs: bool,
    allow_fraction: bool,
):
    problems = []

    for _ in range(n):
        s, a = generate_pemdas_problem(
            operations,
            allow_negative,
            allow_exponents,
            allow_abs,
            allow_fraction
        )
        problems.append({"expr": s, "ans": str(a)})

    return problems


def to_latex(expr_str: str) -> str:
    s = expr_str
    s = s.replace("×", r"\times")
    s = s.replace("÷", r"\div")

    out = []
    open_bar = True

    for ch in s:
        if ch == "|":
            if open_bar:
                out.append(r"\lvert")
            else:
                out.append(r"\rvert")
            open_bar = not open_bar
        else:
            out.append(ch)

    if not open_bar:
        out.append(r"\rvert")

    return "".join(out)


def _apply_op(current, op, k, allow_fraction: bool):
    if op == "+":
        return current + k

    if op == "-":
        return current - k

    if op == "×":
        return current * k

    if op == "÷":
        if k == 0:
            raise ValueError("Cannot divide by zero.")
        # Always do real division correctly. The allow_fraction setting should
        # control problem generation only; it should never change arithmetic.
        return sp.Rational(current, k)

    if op == "^":
        return current ** int(k)

    raise ValueError("Unknown op")


def generate_step_pemdas_problem(
    operations: int,
    allow_negative: bool,
    allow_exponents: bool,
    allow_abs: bool,
    allow_fraction: bool
):
    a = _simple_int(allow_negative)
    b = _simple_int(allow_negative)

    ops = ["+", "-", "×", "÷"]

    if allow_exponents:
        ops.append("^")

    op1 = random.choice(ops)

    if op1 == "^":
        b = random.choice([2, 3])

    expr = f"({a} {op1} {b})"
    val = sp.Rational(a, 1)
    val = _apply_op(val, op1, sp.Rational(b, 1), allow_fraction)

    if allow_abs and random.random() < 0.25:
        expr = f"|{expr}|"
        val = abs(val)

    steps = [str(val)]

    for _ in range(operations - 1):
        op = random.choice(ops)
        k = _simple_int(allow_negative)

        if op == "^":
            k = random.choice([2, 3])

        if op == "÷":
            k = random.randint(2, 10)

        expr = f"({expr} {op} {k})"
        val = _apply_op(val, op, sp.Rational(k, 1), allow_fraction)

        if allow_abs and random.random() < 0.15:
            expr = f"|{expr}|"
            val = abs(val)

        steps.append(str(val))

    return expr, steps


# -------------------------
# Routes: Auth
# -------------------------
@app.route("/")
def home():
    if session.get("user_id"):
        return redirect(url_for("algebra"))
    return redirect(url_for("login"))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if len(username) < 3:
            flash("Username must be at least 3 characters.")
            return redirect(url_for("signup"))

        if len(password) < 6:
            flash("Password must be at least 6 characters.")
            return redirect(url_for("signup"))

        pw_hash = generate_password_hash(password)

        try:
            conn = get_db()
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, pw_hash),
            )
            conn.commit()
            conn.close()
            flash("Account created. Please login.")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("That username is already taken.")
            return redirect(url_for("signup"))

    content = """
    <div class="center">
      <form method="POST">
        <label>Username</label>
        <input name="username" placeholder="Choose a username" />

        <label>Password</label>
        <input type="password" name="password" placeholder="Create a password" />

        <button type="submit">Create Account</button>
      </form>

      <p class="muted" style="margin-top:10px;">
        Already have an account? <a href="/login">Login</a>
      </p>
    </div>
    """

    return render_page(
        "Create Account",
        "Make an account to access the math tabs.",
        active="",
        content_html=content,
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        conn.close()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("algebra"))

        flash("Invalid username or password.")
        return redirect(url_for("login"))

    content = """
    <div class="center">
      <form method="POST">
        <label>Username</label>
        <input name="username" />

        <label>Password</label>
        <input type="password" name="password" />

        <button type="submit">Login</button>
      </form>

      <p class="muted" style="margin-top:10px;">
        New here? <a href="/signup">Create an account</a>
      </p>
    </div>
    """

    return render_page(
        "Login",
        "Login to access the math practice tabs.",
        active="",
        content_html=content,
    )


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.")
    return redirect(url_for("login"))


# -------------------------
# API routes
# -------------------------
@app.route("/api/recognize", methods=["POST"])
@login_required
def api_recognize():
    data = request.get_json(silent=True) or {}
    img_data_url = data.get("image_data_url", "")
    strokes = data.get("strokes", [])

    result = recognize_handwriting_from_data_url(
        img_data_url,
        strokes=strokes
    )

    if not result.get("ok"):
        return jsonify(result), 400

    return jsonify(result)


@app.route("/api/save_stroke_sample", methods=["POST"])
@login_required
def api_save_stroke_sample():
    data = request.get_json(silent=True) or {}

    label = str(data.get("label", "")).strip()
    strokes = data.get("strokes", [])

    allowed_labels = {
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "plus", "minus", "times", "divide", "equals",
        "left_paren", "right_paren"
    }

    if label not in allowed_labels:
        return jsonify({"ok": False, "error": f"Invalid label: {label}"}), 400

    if not strokes:
        return jsonify({"ok": False, "error": "No strokes received."}), 400

    folder = os.path.join("stroke_training_data", label)
    os.makedirs(folder, exist_ok=True)

    existing = [
        f for f in os.listdir(folder)
        if f.lower().endswith(".json")
    ]

    filename = f"user_{len(existing) + 1:05d}.json"
    path = os.path.join(folder, filename)

    payload = {
        "label": label,
        "source": "website_canvas",
        "strokes": strokes
    }

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    return jsonify({"ok": True, "path": path})


# ---------------------------------------------------------------------------
# /api/check_step
#
# Validates the student's answer for one step of the PEMDAS step-by-step
# problem against the expected step value from the session. Replies with a
# specific explanatory message so the front-end can show useful feedback
# rather than a bare "Check failed."
#
# Response shape (the JS in checkStepAnswer expects all of these):
#   {ok: bool, correct: bool, message: str, expected: str}
#   ok=false is only used for setup-level errors (no problem in session,
#   bad index). Recognised input that is wrong is still ok=true with
#   correct=false and a helpful message.
# ---------------------------------------------------------------------------
def _pretty_value(v):
    """Render a sympy value in a human form: 7/3, 0.5, 8, etc."""
    try:
        v = sp.nsimplify(v)
    except Exception:
        pass
    if isinstance(v, sp.Rational) and v.q != 1:
        return f"{v.p}/{v.q}"
    return str(v)


def _is_plain_final_number_text(answer: str) -> bool:
    """True only for a single final numeric answer, not an expression/equation."""
    s = str(answer or "").strip().replace("−", "-")
    if re.fullmatch(r"\(\s*-?\d+(?:\.\d+)?\s*\)", s):
        s = s.strip()[1:-1].strip()
    return re.fullmatch(r"-?(?:\d+(?:\.\d+)?|\.\d+)(?:/-?(?:\d+(?:\.\d+)?|\.\d+))?", s) is not None


@app.route("/api/check_step", methods=["POST"])
@login_required
def api_check_step():
    data = request.get_json(silent=True) or {}

    try:
        step_index = int(data.get("step_index", -1))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Invalid step index."}), 400

    answer = (data.get("answer") or "").strip()

    problem = session.get("pemdas_steps_problem") or {}
    steps = problem.get("steps", [])

    if not steps:
        return jsonify({
            "ok": False,
            "error": "No active step-by-step problem in your session. Click 'New Problem' to start one."
        }), 400

    if step_index < 0 or step_index >= len(steps):
        return jsonify({"ok": False, "error": "Invalid step index."}), 400

    if not answer:
        return jsonify({
            "ok": True,
            "correct": False,
            "count_stats": False,
            "expected": "",
            "message": "Please type an answer or use drawing recognition first."
        })

    expected_str = str(steps[-1])
    is_last_step = (step_index == len(steps) - 1)
    is_final_number = _is_plain_final_number_text(answer)

    try:
        expected = sp.nsimplify(sp.sympify(expected_str, rational=True))
    except Exception:
        return jsonify({
            "ok": False,
            "error": "Could not parse the final answer."
        }), 500

    expected_pretty = _pretty_value(expected)
    norm = normalize_math_text(answer)

    def wrong(msg: str, *, count_stats: bool = False):
        return jsonify({
            "ok": True,
            "correct": False,
            "count_stats": bool(count_stats),
            "expected": expected_pretty,
            "message": msg
        })

    def right(msg: str, *, count_stats: bool = False):
        return jsonify({
            "ok": True,
            "correct": True,
            "count_stats": bool(count_stats),
            "expected": expected_pretty,
            "message": msg
        })

    # Equation form, example: 56*3+5=173.
    # Equations can be accepted for earlier steps, but they should never count stats.
    if "=" in norm:
        try:
            lhs_str, rhs_str = norm.split("=", 1)
            lhs = sp.simplify(sp.sympify(lhs_str, rational=True))
            rhs = sp.simplify(sp.sympify(rhs_str, rational=True))
        except Exception:
            return wrong("I couldn't parse your equation. Make sure both sides are valid expressions.")

        if sp.simplify(lhs - rhs) != 0:
            return wrong(
                f"The two sides of your equation are not equal: "
                f"left = {_pretty_value(lhs)}, right = {_pretty_value(rhs)}."
            )

        if sp.simplify(lhs - expected) == 0:
            if is_last_step:
                return wrong(
                    "Your equation is true, but the final step must be one number only. "
                    f"Please enter {expected_pretty}.",
                    count_stats=False
                )
            return right("Correct. Your equation is true and matches the whole problem.", count_stats=False)

        return wrong(
            f"Your equation is true, but it equals {_pretty_value(lhs)}. "
            f"The whole problem should equal {expected_pretty}."
        )

    # Plain expression, example: (-7+4)*4 or a final number like -12.
    try:
        val = sp.simplify(sp.sympify(norm, rational=True))
    except Exception:
        return wrong(
            f"I couldn't parse \"{answer}\" as a math expression. "
            f"Use +, -, *, /, and parentheses."
        )

    equivalent = (sp.simplify(val - expected) == 0)

    # Last step rule: last step must be the single final number, not an expression.
    if is_last_step:
        if not is_final_number:
            if equivalent:
                return wrong(
                    "Almost correct, but the final step must be one number only. "
                    f"Please enter {expected_pretty}.",
                    count_stats=False
                )
            return wrong(
                f"You wrote {_pretty_value(val)}, but the final answer should be {expected_pretty}. "
                "On the last step, enter one number only.",
                count_stats=False
            )

        if equivalent:
            return right("Correct. Final answer is one number and matches the whole problem.", count_stats=True)

        diff = sp.simplify(val - expected)
        return wrong(
            f"You wrote {_pretty_value(val)}, but the final answer should be "
            f"{expected_pretty}. You are off by {_pretty_value(diff)}.",
            count_stats=True
        )

    # Earlier steps: equivalent expressions are correct, but stats are not counted.
    if equivalent:
        return right("Correct. Your expression is equivalent to the whole problem.", count_stats=False)

    diff = sp.simplify(val - expected)
    return wrong(
        f"You wrote {_pretty_value(val)}, but the whole problem should equal "
        f"{expected_pretty}. You are off by {_pretty_value(diff)}.",
        count_stats=False
    )


@app.route("/api/save_symbol_sample", methods=["POST"])
@login_required
def save_symbol_sample():
    data = request.get_json(silent=True) or {}

    label = data.get("label", "").strip()
    image_data_url = data.get("image_data_url", "")

    allowed_labels = {
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "plus", "minus", "times", "divide", "equals", "left_paren", "right_paren"
    }

    if label not in allowed_labels:
        return jsonify({"ok": False, "error": "Invalid label."}), 400

    if not image_data_url.startswith("data:image/png;base64,"):
        return jsonify({"ok": False, "error": "Invalid image data."}), 400

    folder = os.path.join("training_data", label)
    os.makedirs(folder, exist_ok=True)

    existing = [
        f for f in os.listdir(folder)
        if f.lower().endswith(".png")
    ]

    filename = f"sample_{len(existing) + 1:04d}.png"
    path = os.path.join(folder, filename)

    b64 = image_data_url.replace("data:image/png;base64,", "")

    try:
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64))
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    return jsonify({"ok": True, "path": path})


# -------------------------
# Routes: Main tabs
# -------------------------
@app.route("/algebra", methods=["GET"])
@login_required
def algebra():
    content = f"""
    <div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:12px;">
      <a class="tab" href="{url_for('pemdas')}">PEMDAS Practice</a>
      <a class="tab" href="{url_for('pemdas_steps')}">PEMDAS Step-by-Step</a>
      <a class="tab" href="{url_for('collect_symbols')}">Collect Symbol Data</a>
    </div>

    <p class="muted">
      Choose a practice set above, or collect handwriting samples for the symbol recognizer.
    </p>
    """

    return render_page(
        "Algebra Foundation",
        "Practice pages for arithmetic and algebra foundations.",
        "algebra",
        content
    )


@app.route("/algebra/pemdas", methods=["GET", "POST"])
@login_required
def pemdas():
    operations = int(request.values.get("operations", 5))
    n = int(request.values.get("n", 20))

    allow_negative = (request.values.get("neg", "0") == "1")
    allow_exponents = (request.values.get("exp", "0") == "1")
    allow_abs = (request.values.get("abs", "0") == "1")
    allow_fraction = (request.values.get("frac", "0") == "1")

    operations = max(1, min(operations, 10))

    if n not in (5, 10, 15, 20):
        n = 20

    settings_key = "pemdas_settings"
    problems_key = "pemdas_problems"

    if request.method == "GET":
        session[settings_key] = {
            "operations": operations,
            "n": n,
            "neg": allow_negative,
            "exp": allow_exponents,
            "abs": allow_abs,
            "frac": allow_fraction,
        }

        session[problems_key] = generate_pemdas_set(
            operations,
            n,
            allow_negative,
            allow_exponents,
            allow_abs,
            allow_fraction,
        )

    problems = session.get(problems_key, [])
    settings = session.get(settings_key, {
        "operations": operations,
        "n": n,
        "neg": allow_negative,
        "exp": allow_exponents,
        "abs": allow_abs,
        "frac": allow_fraction,
    })

    checked_neg = "checked" if settings.get("neg") else ""
    checked_exp = "checked" if settings.get("exp") else ""
    checked_abs = "checked" if settings.get("abs") else ""
    checked_frac = "checked" if settings.get("frac") else ""

    settings_html = f"""
    <form method="GET" style="margin-bottom:14px;">
      <div>
        <label><b>Operations (1–10)</b></label>
        <select name="operations">
          {''.join([f'<option value="{k}" {"selected" if settings.get("operations")==k else ""}>{k}</option>' for k in range(1, 11)])}
        </select>
      </div>

      <div style="margin-top:12px;">
        <label><b># Problems</b></label>
        <select name="n">
          <option value="5" {"selected" if settings.get("n")==5 else ""}>5</option>
          <option value="10" {"selected" if settings.get("n")==10 else ""}>10</option>
          <option value="15" {"selected" if settings.get("n")==15 else ""}>15</option>
          <option value="20" {"selected" if settings.get("n")==20 else ""}>20</option>
        </select>
      </div>

      <div style="margin-top:14px;">
        <label><input type="checkbox" name="neg" value="1" {checked_neg}/> Allow negatives</label><br>
        <label><input type="checkbox" name="exp" value="1" {checked_exp}/> Allow exponents</label><br>
        <label><input type="checkbox" name="abs" value="1" {checked_abs}/> Allow absolute values</label><br>
        <label><input type="checkbox" name="frac" value="1" {checked_frac}/> Allow fractions</label>
      </div>

      <div style="margin-top:16px;">
        <button type="submit">Generate</button>
      </div>
    </form>
    """

    problems_html = []

    for i, p in enumerate(problems):
        latex_expr = to_latex(p["expr"])
        problems_html.append(f"""
          <div style="display:flex; gap:12px; align-items:center; margin:14px 0;">
            <div style="width:30px;"><b>{i+1}.</b></div>
            <div style="flex:1; font-size:20px;">\\({latex_expr}\\)</div>
            <div style="width:220px;">
              <input name="a{i}" placeholder="Answer, example: 7/3" />
            </div>
          </div>
        """)

    result_html = ""

    if request.method == "POST":
        action = request.form.get("action", "check")
        reveal = (action == "submit")

        correct_count = 0
        rows = []

        for i, p in enumerate(problems):
            user_val = (request.form.get(f"a{i}", "") or "").strip()
            ok = False
            err = ""

            try:
                u = value_from_expression_or_equation(user_val)
                e = sp.nsimplify(sp.Rational(p["ans"]))
                ok = (sp.simplify(u - e) == 0)
            except Exception as ex:
                ok = False
                err = str(ex)

            if ok:
                correct_count += 1

            mark = "✅" if ok else "❌"

            if reveal:
                ans_html = f"<span class=muted>(Ans: \\({sp.latex(sp.Rational(p['ans']))}\\))</span>"
            else:
                ans_html = "<span class=muted>(Hidden)</span>"

            err_html = f"<div class='muted'>Error: {err}</div>" if (err and not ok) else ""

            latex_expr = to_latex(p["expr"])

            rows.append(
                f"<tr>"
                f"<td>{i+1}</td>"
                f"<td>\\({latex_expr}\\)</td>"
                f"<td>{user_val if user_val else '<span class=muted>(blank)</span>'}{err_html}</td>"
                f"<td>{mark} {ans_html}</td>"
                f"</tr>"
            )

        result_html = f"""
        <hr>
        <p><b>Score:</b> {correct_count}/{len(problems)}</p>
        <table style="width:100%; border-collapse:collapse;">
          <thead>
            <tr style="text-align:left; border-bottom:1px solid #eee;">
              <th>#</th><th>Problem</th><th>Your Answer</th><th>Result</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
        """

    content = f"""
    <div style="margin-bottom:12px;">
      <a class="tab" href="{url_for('algebra')}">← Back to Algebra</a>
    </div>

    <p class="muted">PEMDAS Practice — customize operations, problem count, and features.</p>

    {settings_html}

    <form method="POST">
      {''.join(problems_html)}
      <button type="submit" name="action" value="check">Check Answers (No Key)</button>
      <button type="submit" name="action" value="submit" style="margin-left:10px;">Submit & Show Answers</button>
    </form>

    {result_html}
    """

    return render_page(
        "Algebra Foundation — PEMDAS",
        "Random order-of-operations practice with answer checking.",
        "algebra",
        content
    )


@app.route("/collect-symbols", methods=["GET"])
@login_required
def collect_symbols():
    labels = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "plus", "minus", "times", "divide", "equals", "left_paren", "right_paren"
    ]

    options = "".join([
        f'<option value="{label}">{label}</option>'
        for label in labels
    ])

    content = f"""
    <p class="muted">
      Draw one symbol at a time. Choose the correct label, then save it.
      This will create your own handwriting dataset for PEMDAS.
    </p>

    <label><b>Symbol Label</b></label>
    <select id="symbol-label" style="padding:10px; border-radius:10px; margin-bottom:12px;">
      {options}
    </select>

    <div class="draw-wrap">
      <canvas class="draw-canvas" data-canvas-id="collect"></canvas>
      <input type="hidden" data-canvas-id="collect" value="" />

      <div class="draw-actions">
        <button type="button" class="btn-lite" data-action="clear" data-canvas-id="collect">Clear</button>
        <button type="button" id="save-symbol-btn">Save Training Sample</button>
      </div>
    </div>

    <p id="save-result" class="muted" style="margin-top:12px;"></p>

    <script>
    document.addEventListener("DOMContentLoaded", function() {{
      const saveBtn = document.getElementById("save-symbol-btn");
      const result = document.getElementById("save-result");

      saveBtn.addEventListener("click", async function(e) {{
        e.preventDefault();

        const label = document.getElementById("symbol-label").value;
        const api = window.canvasApis["collect"];

        if (!api) {{
          result.innerHTML = "Canvas is not ready. Refresh page.";
          return;
        }}

        const png = api.toDataURL();

        const resp = await fetch("/api/save_symbol_sample", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{
            label: label,
            image_data_url: png
          }})
        }});

        const data = await resp.json();

        if (data.ok) {{
          result.innerHTML = "Saved: " + data.path;
        }} else {{
          result.innerHTML = data.error || "Save failed.";
        }}
      }});
    }});
    </script>
    """

    return render_page(
        "Collect Symbol Data",
        "Build your own handwriting dataset for PEMDAS.",
        "algebra",
        content
    )


@app.route("/algebra/pemdas-steps", methods=["GET", "POST"])
@login_required
def pemdas_steps():
    operations = int(request.values.get("operations", 4))
    operations = max(2, min(operations, 8))

    allow_negative = (request.values.get("neg", "0") == "1")
    allow_exponents = (request.values.get("exp", "0") == "1")
    allow_abs = (request.values.get("abs", "0") == "1")
    allow_fraction = (request.values.get("frac", "0") == "1")

    key = "pemdas_steps_problem"

    if request.method == "GET" or request.form.get("action") == "new":
        expr, steps = generate_step_pemdas_problem(
            operations,
            allow_negative,
            allow_exponents,
            allow_abs,
            allow_fraction,
        )

        session[key] = {
            "expr": expr,
            "steps": steps,
            "settings": {
                "operations": operations,
                "neg": allow_negative,
                "exp": allow_exponents,
                "abs": allow_abs,
                "frac": allow_fraction,
            }
        }

    data = session.get(key)

    if not data:
        expr, steps = generate_step_pemdas_problem(
            operations,
            allow_negative,
            allow_exponents,
            allow_abs,
            allow_fraction,
        )

        data = {
            "expr": expr,
            "steps": steps,
            "settings": {
                "operations": operations,
                "neg": allow_negative,
                "exp": allow_exponents,
                "abs": allow_abs,
                "frac": allow_fraction,
            }
        }

        session[key] = data

    expr = data["expr"]
    steps = data["steps"]
    settings = data["settings"]
    session["pemdas_expected_steps"] = [str(x) for x in steps]

    checked_neg = "checked" if settings.get("neg") else ""
    checked_exp = "checked" if settings.get("exp") else ""
    checked_abs = "checked" if settings.get("abs") else ""
    checked_frac = "checked" if settings.get("frac") else ""

    result_html = ""

    if request.method == "POST" and request.form.get("action") in ["check", "submit"]:
        rows = []
        correct_count = 0
        reveal = request.form.get("action") == "submit"

        for i, expected in enumerate(steps):
            user_val = request.form.get(f"s{i}", "").strip()

            ok = False
            err = ""

            try:
                u = value_from_expression_or_equation(user_val)
                e = sp.nsimplify(sp.Rational(expected))
                ok = (sp.simplify(u - e) == 0)
            except Exception as ex:
                ok = False
                err = str(ex)

            if ok:
                correct_count += 1

            mark = "✅" if ok else "❌"

            if reveal:
                ans_html = f"<span class=muted>(Correct: \\({sp.latex(sp.Rational(expected))}\\))</span>"
            else:
                ans_html = "<span class=muted>(Answer hidden)</span>"

            err_html = f"<div class='muted'>Error: {err}</div>" if (err and not ok) else ""

            rows.append(
                f"<tr>"
                f"<td><b>Step {i+1}</b></td>"
                f"<td>{user_val if user_val else '<span class=muted>(blank)</span>'}{err_html}</td>"
                f"<td>{mark} {ans_html}</td>"
                f"</tr>"
            )

        result_html = f"""
        <hr>
        <p><b>Steps correct:</b> {correct_count}/{len(steps)}</p>
        <table style="width:100%; border-collapse:collapse;">
          <thead>
            <tr style="text-align:left; border-bottom:1px solid #eee;">
              <th>Step</th><th>Your result</th><th>Check</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
        """

    latex_expr = to_latex(expr)

    stats_type_key = (
        f"pemdas_steps_type_{settings.get('operations')}_"
        f"N{1 if settings.get('neg') else 0}_"
        f"E{1 if settings.get('exp') else 0}_"
        f"A{1 if settings.get('abs') else 0}_"
        f"F{1 if settings.get('frac') else 0}"
    )
    stats_problem_key = f"{stats_type_key}::{expr}"

    settings_html = f"""
    <form method="GET" style="margin-bottom:14px;">
      <div>
        <label><b>How many steps?</b> (2–8)</label>
        <select name="operations">
          {''.join([f'<option value="{k}" {"selected" if settings.get("operations")==k else ""}>{k}</option>' for k in range(2, 9)])}
        </select>
      </div>

      <div style="margin-top:14px;">
        <label><input type="checkbox" name="neg" value="1" {checked_neg}/> Allow negatives</label><br>
        <label><input type="checkbox" name="exp" value="1" {checked_exp}/> Allow exponents</label><br>
        <label><input type="checkbox" name="abs" value="1" {checked_abs}/> Allow absolute values</label><br>
        <label><input type="checkbox" name="frac" value="1" {checked_frac}/> Allow fractions</label>
      </div>

      <div style="margin-top:16px;">
        <button type="submit">Generate New Problem</button>
      </div>
    </form>
    """

    step_inputs = []

    for i in range(len(steps)):
        step_inputs.append(f"""
          <div class="step-box" data-step-index="{i}" style="margin:18px 0; padding:14px; border:1px solid #eee; border-radius:14px;">
            <div style="font-weight:700; margin-bottom:8px;">Step {i+1}</div>

            <input type="text" name="s{i}" data-canvas-id="step{i}"
                   placeholder="Type an equivalent expression or equation, example: (4/6)/3 or (4/6)/3=2/9" />

            <div class="muted" style="margin-top:10px;">Or draw below (optional):</div>

            <div class="draw-wrap">
              <canvas class="draw-canvas" data-canvas-id="step{i}"></canvas>

              <input type="hidden" name="draw{i}" data-canvas-id="step{i}" value="" />

              <div class="draw-actions">
                <button type="button" class="btn-lite" data-action="clear" data-canvas-id="step{i}">Clear</button>
                <button type="button" class="btn-lite" data-action="undo" data-canvas-id="step{i}">Undo Last Stroke</button>
                <button type="button" class="btn-lite" data-action="eraser" data-canvas-id="step{i}">Eraser</button>
                <button type="button" data-action="recognize" data-canvas-id="step{i}">Use Drawing (Recognize)</button>
                <button type="button" class="btn-lite" onclick="checkStepAnswer(this)">Check Answer</button>
              </div>

              <div class="step-feedback" style="margin-top:10px;"></div>

              <div style="margin-top:12px; padding:12px; border:1px dashed #ccc; border-radius:12px;">
                <div class="muted" style="margin-bottom:8px;">
                  Training save: draw <b>one symbol only</b>, choose its correct label, then save.
                </div>

                <select data-role="sample-label" data-canvas-id="step{i}"
                        style="padding:10px; border-radius:10px; border:1px solid #ddd;">
                  <option value="left_paren">left_paren  (</option>
                  <option value="right_paren">right_paren  )</option>
                  <option value="0">0</option>
                  <option value="1">1</option>
                  <option value="2">2</option>
                  <option value="3">3</option>
                  <option value="4">4</option>
                  <option value="5">5</option>
                  <option value="6">6</option>
                  <option value="7">7</option>
                  <option value="8">8</option>
                  <option value="9">9</option>
                  <option value="plus">plus  +</option>
                  <option value="minus">minus  -</option>
                  <option value="times">times  ×</option>
                  <option value="divide">divide  ÷</option>
                  <option value="equals">equals  =</option>
                </select>

                <button type="button" class="btn-lite"
                        data-action="save-stroke-sample"
                        data-canvas-id="step{i}">
                  Save Stroke Training Sample
                </button>

                <div class="muted" data-role="sample-status" data-canvas-id="step{i}"
                     style="margin-top:8px;"></div>
              </div>
            </div>
          </div>
        """)

    content = f"""
    <script>
      window.pemdasTypeKey = {json.dumps(stats_type_key)};
      window.pemdasProblemKey = {json.dumps(stats_problem_key)};
    </script>

    <div id="typeStatsBox" class="type-stats-box">
      <div class="type-stats-box-title">This Type Stats</div>
      <div>Done: <span id="statDone">0</span></div>
      <div>Correct: <span id="statCorrect">0</span></div>
      <div>Incorrect: <span id="statIncorrect">0</span></div>
      <button type="button" class="small-reset-btn" onclick="resetTypeStats()">Reset</button>
    </div>

    <div style="margin-bottom:12px;">
      <a class="tab" href="{url_for('algebra')}">← Back to Algebra</a>
    </div>

    <p class="muted">Solve the problem one layer at a time. Enter your result after each step.</p>

    {settings_html}

    <div style="font-size:22px; margin: 10px 0 16px 0;">
      <b>Problem:</b> \\({latex_expr}\\)
    </div>

    <form method="POST">
      {''.join(step_inputs)}
      <button type="submit" name="action" value="check">Check Steps (No Answers)</button>
      <button type="submit" name="action" value="submit" style="margin-left:10px;">Submit (Show Answers)</button>
      <button type="submit" name="action" value="new" style="margin-left:10px;">New Problem</button>
    </form>

    {result_html}
    """

    return render_page(
        "Algebra Foundation — PEMDAS Step-by-Step",
        "One problem at a time, with step checking.",
        "algebra",
        content
    )


@app.route("/geometry", methods=["GET"])
@login_required
def geometry():
    content = """
    <p><b>Coming next:</b></p>
    <ul>
      <li>Area / perimeter checker</li>
      <li>Pythagorean theorem</li>
      <li>Angles and triangles</li>
    </ul>
    <p class="muted">Geometry usually needs values or diagrams. Tell me what geometry problems you want first.</p>
    """

    return render_page(
        "Geometry",
        "Diagram-based problems.",
        "geometry",
        content
    )


@app.route("/trig", methods=["GET", "POST"])
@login_required
def trig():
    result_html = ""

    if request.method == "POST":
        ok, err = check_equivalence(
            request.form.get("student", ""),
            request.form.get("correct", "")
        )

        if err:
            result_html = f"<hr><p><b>Error:</b> {err}</p>"
        else:
            if ok:
                result_html = "<hr><p><b>Result:</b> ✅ Correct!</p>"
            else:
                result_html = "<hr><p><b>Result:</b> ❌ Incorrect</p>"

    content = checker_form(
        "Use sin(x), cos(x), tan(x). Example: 2*sin(x)*cos(x)."
    ) + result_html

    return render_page(
        "Trigonometry",
        "Trig identities and simplification checker.",
        "trig",
        content
    )


@app.route("/calculus", methods=["GET", "POST"])
@login_required
def calculus():
    result_html = ""

    if request.method == "POST":
        ok, err = check_equivalence(
            request.form.get("student", ""),
            request.form.get("correct", "")
        )

        if err:
            result_html = f"<hr><p><b>Error:</b> {err}</p>"
        else:
            if ok:
                result_html = "<hr><p><b>Result:</b> ✅ Correct!</p>"
            else:
                result_html = "<hr><p><b>Result:</b> ❌ Incorrect</p>"

    content = checker_form(
        "You can type derivatives/integrals as expressions. We can add diff()/integrate() checking next."
    ) + result_html

    return render_page(
        "Calculus",
        "Derivatives and integrals checker.",
        "calculus",
        content
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
