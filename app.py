from flask import Flask, request, render_template_string, redirect, url_for, session, flash
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import sympy as sp
from functools import wraps
import random
from sympy import Integer

app = Flask(__name__)
app.secret_key = "change-this-to-a-random-secret"  # needed for sessions

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
# UI templates
# -------------------------
BASE_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{{ title }}</title>

  <!-- MathJax for LaTeX rendering -->
  <script>
    window.MathJax = {
      tex: { inlineMath: [['\\\\(', '\\\\)'], ['$', '$']] }
    };
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

  <style>
    body { font-family: Arial, sans-serif; max-width: 980px; margin: 32px auto; padding: 0 16px; }
    .topbar { display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:14px; }
    .brand { font-weight:700; }
    .auth a { margin-left:10px; text-decoration:none; }
    .tabs { display: flex; gap: 10px; border-bottom: 1px solid #ddd; padding-bottom: 10px; margin-bottom: 18px; }
    .tab { text-decoration: none; padding: 10px 12px; border-radius: 10px; color: #111; }
    .tab:hover { background: #f3f3f3; }
    .active { background: #111; color: #fff; }
    .card { border: 1px solid #e5e5e5; border-radius: 14px; padding: 16px; }
    input { width: 100%; padding: 10px; margin: 6px 0 12px; border: 1px solid #ddd; border-radius: 10px; }
    button { padding: 10px 14px; border: 0; border-radius: 10px; background: #111; color: #fff; cursor: pointer; }
    button:hover { opacity: 0.9; }
    .muted { color: #666; }
    .flash { background:#fff4cc; border:1px solid #ffe08a; padding:10px; border-radius:10px; margin: 12px 0; }
    .center { max-width: 520px; margin: 40px auto; }
    .formgrid { display:grid; grid-template-columns: 220px 220px 1fr auto; gap:14px; align-items:end; }
    .field label { display:block; font-weight:700; margin-bottom:6px; }
    .field input, .field select { width:100%; padding:10px; border:1px solid #ddd; border-radius:10px; }
    .checks { display:flex; gap:16px; flex-wrap:wrap; align-items:center; padding-top:26px; }
    .checks label { display:flex; gap:8px; align-items:center; font-weight:600; }
    .checks input { width:auto; margin:0; }
    .actions { padding-top:26px; }
    /* ✅ Fix: don't apply the big input styling to checkboxes/radios */
    input[type="checkbox"], input[type="radio"] {
        width: auto !important;
        margin: 0 !important;
        padding: 0 !important;
    }
  </style>
</head>
<body>

  <div class="topbar">
    <div class="brand">Math Practice</div>
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
# Math checker helpers
# -------------------------
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

def check_equivalence(student_str: str, correct_str: str):
    x = sp.Symbol("x")
    try:
        s = sp.simplify(sp.sympify(student_str))
        c = sp.simplify(sp.sympify(correct_str))
        ok = (sp.simplify(s - c) == 0)
        return ok, None
    except Exception as e:
        return False, str(e)
    
def _safe_int(n: int) -> Integer:
    return Integer(int(n))

def _operand(allow_negative: bool, allow_abs: bool, allow_fraction: bool):
    """
    Returns (expr_str, value) where value is int or sympy.Rational.
    If allow_fraction=True, sometimes returns a true fraction like \\frac{7}{3}.
    """
    # With fractions enabled, sometimes start with a fractional operand
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
    """
    Returns (expr_string, answer_value)
    answer_value can be int or sympy.Rational when allow_fraction=True
    """
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
            val = val + op_val if op == "+" else val - op_val

        elif op == "×":
            m = random.randint(2, 10)
            if allow_negative and random.random() < 0.2:
                m = -m
            expr = f"({expr} × {m})"
            val = val * m

        elif op == "÷":
            d = random.randint(2, 10)

            if allow_fraction:
                # show as real fraction bar (readable)
                expr = rf"\frac{{{expr}}}{{{d}}}"
                val = sp.Rational(val, d)
                saw_fraction = True
            else:
                # force integer result
                k = random.randint(2, 6)
                expr = f"(({expr} × {d*k}) ÷ {d})"
                val = (val * (d * k)) // d

        elif op == "^":
            if used_pow:
                continue
            # avoid huge blow-ups
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

        # If fractions allowed, detect we have one
        if allow_fraction and isinstance(val, sp.Rational) and val.q != 1:
            saw_fraction = True

        # Keep numbers reasonable
        try:
            too_big = abs(val) > 10000
        except Exception:
            too_big = abs(float(val)) > 10000

        if too_big:
            return generate_pemdas_problem(
                operations, allow_negative, allow_exponents, allow_abs, allow_fraction
            )

    # Guarantee at least one visible fraction if option enabled
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
            operations, allow_negative, allow_exponents, allow_abs, allow_fraction
        )
        # IMPORTANT: store answer as a string so Flask session can serialize it
        problems.append({"expr": s, "ans": str(a)})
    return problems



def to_latex(expr_str: str) -> str:
    s = expr_str
    s = s.replace("×", r"\times")
    s = s.replace("÷", r"\div")

    # Convert |...| into \lvert...\rvert (no \left / \right at all)
    out = []
    open_bar = True
    for ch in s:
        if ch == "|":
            out.append(r"\lvert" if open_bar else r"\rvert")
            open_bar = not open_bar
        else:
            out.append(ch)

    # If odd number of |, close it to avoid MathJax breaking
    if not open_bar:
        out.append(r"\rvert")

    return "".join(out)


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
            (username,)
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
        <input name="username" placeholder="Enter your username" />
        <label>Password</label>
        <input type="password" name="password" placeholder="Enter your password" />
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
# Routes: Tabs (Login required)
# -------------------------
@app.route("/algebra", methods=["GET"])
@login_required
def algebra():
    content = f"""
    <div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:12px;">
      <a class="tab" href="{url_for('pemdas')}">PEMDAS Practice</a>
      <!-- add more child pages later -->
      <!-- <a class="tab" href="...">Equations</a> -->
      <!-- <a class="tab" href="...">Fractions</a> -->
    </div>

    <p class="muted">Choose a practice set above.</p>
    """
    return render_page(
        "Algebra Foundation",
        "Practice pages (PEMDAS, equations, factoring, etc.).",
        "algebra",
        content
    )

@app.route("/algebra/pemdas", methods=["GET", "POST"])
@login_required
def pemdas():
    # Defaults
    operations = int(request.values.get("operations", 5))
    n = int(request.values.get("n", 20))

    allow_negative = (request.values.get("neg", "0") == "1")
    allow_exponents = (request.values.get("exp", "0") == "1")  # default off unless checked
    allow_abs = (request.values.get("abs", "0") == "1")
    allow_fraction = (request.values.get("frac", "0") == "1")

    # Clamp to safe ranges
    operations = max(1, min(operations, 10))
    if n not in (5, 10, 15, 20):
        n = 20

    settings_key = f"pemdas_settings"
    problems_key = f"pemdas_problems"

    # If user clicked "Generate" (GET) or first time, generate new set
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
            operations=operations,
            n=n,
            allow_negative=allow_negative,
            allow_exponents=allow_exponents,
            allow_abs=allow_abs,
            allow_fraction=allow_fraction,
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

    result_html = ""
    if request.method == "POST":
        # use stored problems (so answers match what user saw)
        correct_count = 0
        rows_html = []

        for i, p in enumerate(problems):
            user_val = request.form.get(f"a{i}", "").strip()
            is_correct = False
            try:
                # allow integers or fractions like 7/3
                user_num = sp.Rational(user_val)  # works for "5" and "7/3"
                ans_num = sp.Rational(p["ans"])   # p["ans"] is stored as a string
                is_correct = (sp.simplify(user_num - ans_num) == 0)
            except:
                is_correct = False

            if is_correct:
                correct_count += 1

            mark = "✅" if is_correct else "❌"
            latex_expr = to_latex(p["expr"])
            rows_html.append(
                f"<tr>"
                f"<td>{i+1}</td>"
                f"<td>\\({latex_expr}\\)</td>"
                f"<td>{user_val if user_val else '<span class=muted>(blank)</span>'}</td>"
                f"<td>{mark} <span class=muted>(Ans: \\({sp.latex(sp.Rational(p['ans']))}\\))</span></td>"
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
            {''.join(rows_html)}
          </tbody>
        </table>
        """

    # Settings form (GET) + Problems form (POST)
    checked_neg = "checked" if settings.get("neg") else ""
    checked_exp = "checked" if settings.get("exp") else ""
    checked_abs = "checked" if settings.get("abs") else ""
    checked_frac = "checked" if settings.get("frac") else ""

    settings_html = f"""
<form method="GET" style="margin-bottom:14px;">

  <!-- Row 1: dropdowns -->
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

  <!-- Row 2: checkboxes -->
  <div style="margin-top:14px;">
    <label><input type="checkbox" name="neg" value="1" {checked_neg}/> Allow negatives</label><br>
    <label><input type="checkbox" name="exp" value="1" {checked_exp}/> Allow exponents</label><br>
    <label><input type="checkbox" name="abs" value="1" {checked_abs}/> Allow absolute values</label><br>
    <label><input type="checkbox" name="frac" value="1" {checked_frac}/> Allow fractions</label>
  </div>

  <!-- Row 3: button at bottom -->
  <div style="margin-top:16px;">
    <button type="submit">Generate</button>
  </div>

  <p class="muted" style="margin-top:10px;">Tip: click Generate to make a new set.</p>

</form>
"""

    problems_html = []
    for i, p in enumerate(problems):
        latex_expr = to_latex(p["expr"])
        problems_html.append(f"""
          <div style="display:flex; gap:12px; align-items:center; margin:14px 0;">
            <div style="width:30px;"><b>{i+1}.</b></div>
            <div style="flex:1; font-size:20px;">\\({latex_expr}\\)</div>
            <div style="width:180px;">
              <input name="a{i}" placeholder="Answer" />
            </div>
          </div>
        """)

    content = f"""
    <div style="margin-bottom:12px;">
      <a class="tab" href="{url_for('algebra')}">← Back to Algebra</a>
    </div>

    <p class="muted">PEMDAS Practice — customize operations, problem count, and features.</p>
    {settings_html}

    <form method="POST">
      {''.join(problems_html)}
      <button type="submit">Check Answers</button>
    </form>

    {result_html}
    """

    return render_page(
        title="Algebra Foundation — PEMDAS",
        subtitle="Random order-of-operations practice with answer checking.",
        active="algebra",
        content_html=content
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
    return render_page("Geometry", "Diagram-based problems (we’ll add geometry-specific inputs next).", "geometry", content)

@app.route("/trig", methods=["GET", "POST"])
@login_required
def trig():
    result_html = ""
    if request.method == "POST":
        ok, err = check_equivalence(request.form.get("student",""), request.form.get("correct",""))
        if err:
            result_html = f"<hr><p><b>Error:</b> {err}</p>"
        else:
            result_html = "<hr><p><b>Result:</b> ✅ Correct!</p>" if ok else "<hr><p><b>Result:</b> ❌ Incorrect</p>"

    content = checker_form("Use sin(x), cos(x), tan(x). Example: 2*sin(x)*cos(x).") + result_html
    return render_page("Trigonometry", "Trig identities and simplification (prototype checker).", "trig", content)

@app.route("/calculus", methods=["GET", "POST"])
@login_required
def calculus():
    result_html = ""
    if request.method == "POST":
        ok, err = check_equivalence(request.form.get("student",""), request.form.get("correct",""))
        if err:
            result_html = f"<hr><p><b>Error:</b> {err}</p>"
        else:
            result_html = "<hr><p><b>Result:</b> ✅ Correct!</p>" if ok else "<hr><p><b>Result:</b> ❌ Incorrect</p>"

    content = checker_form("You can type derivatives/integrals as expressions; we can add diff()/integrate() checking next.") + result_html
    return render_page("Calculus", "Derivatives/integrals (prototype checker).", "calculus", content)


if __name__ == "__main__":
    app.run(debug=True)