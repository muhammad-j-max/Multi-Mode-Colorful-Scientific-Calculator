import streamlit as st
import numpy as np
import sympy as sp
import random

# ---------------- Page config ----------------
st.set_page_config(page_title="Advanced Multi-Mode Calculator", page_icon="🧮", layout="centered")

# ---------------- Style / Background (load once) ----------------
if "bg_set" not in st.session_state:
    page_bg = """
    <style>
    /* page background gradient */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 35%, #f3a683 100%);
        background-attachment: fixed;
    }

    /* watermark */
    [data-testid="stAppViewContainer"]::before {
        content: "🧮  sin(x)";
        position: absolute;
        top: 45%;
        left: 50%;
        font-size: 120px;
        font-weight: 700;
        color: rgba(255, 255, 255, 0.06);
        transform: translate(-50%, -50%) rotate(-18deg);
        z-index: 0;
        pointer-events: none;
    }

    /* button base style */
    div.stButton > button {
        font-size: 16px;
        font-weight: 600;
        border-radius: 10px;
        height: 44px;
        width: 100%;
        margin: 2px 0;
        box-shadow: 0 3px 6px rgba(0,0,0,0.12);
    }

    /* a few utility classes for coloring (applies globally; we'll inline per-key CSS via markdown too) */
    .num-btn { background: #4CAF50 !important; color: white !important; }
    .op-btn  { background: #FF7043 !important; color: white !important; }
    .spec-btn{ background: #9C27B0 !important; color: white !important; }
    .eq-btn  { background: #1976D2 !important; color: white !important; }
    .ac-btn  { background: #E91E63 !important; color: white !important; }
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)
    st.session_state.bg_set = True

# ---------------- Session state initialization ----------------
if "power" not in st.session_state:
    st.session_state.power = False
if "expr" not in st.session_state:
    st.session_state.expr = ""
if "result" not in st.session_state:
    st.session_state.result = None
if "memory" not in st.session_state:
    st.session_state.memory = 0
if "ans" not in st.session_state:
    st.session_state.ans = 0
if "mode" not in st.session_state:
    st.session_state.mode = "DEG"  # DEG or RAD

# ---------------- Helper utilities ----------------
def trig_wrap(func, val):
    # handle degree/radian mode for trig functions
    return func(np.radians(val)) if st.session_state.mode == "DEG" else func(val)

# Allowed functions preloaded (NumPy-first) — SymPy fallback in safe_eval
allowed_funcs = {
    "sin": lambda x: trig_wrap(np.sin, x),
    "cos": lambda x: trig_wrap(np.cos, x),
    "tan": lambda x: trig_wrap(np.tan, x),
    "asin": lambda x: np.degrees(np.arcsin(x)) if st.session_state.mode == "DEG" else np.arcsin(x),
    "acos": lambda x: np.degrees(np.arccos(x)) if st.session_state.mode == "DEG" else np.arccos(x),
    "atan": lambda x: np.degrees(np.arctan(x)) if st.session_state.mode == "DEG" else np.arctan(x),
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "sqrt": np.sqrt, "cbrt": np.cbrt,
    "root": lambda x, y: x ** (1.0 / y),
    "pow": np.power, "square": lambda x: x ** 2,
    "recip": lambda x: 1.0 / x,
    "log": np.log10, "ln": np.log,
    "exp": np.exp, "tenpow": lambda x: 10 ** x,
    "pi": np.pi, "e": np.e,
    "fact": sp.factorial,
    "nPr": lambda n, r: sp.permutations(n, r),
    "nCr": lambda n, r: sp.binomial(n, r),
    "rand": random.random,
    "mean": np.mean, "std": np.std, "sum": np.sum,
    "I": 1j
}

def safe_eval(expression: str):
    """
    Evaluate expression using (fast) NumPy-based allowed_funcs, otherwise fallback to sympy.sympify.
    Returns result or an error string.
    """
    if not expression:
        return None
    try:
        # Evaluate using eval with allowed functions (fast)
        return eval(expression, {"__builtins__": None}, allowed_funcs)
    except Exception:
        try:
            # SymPy fallback for more complicated symbolic expressions
            return sp.sympify(expression).evalf()
        except Exception:
            return "❌ Invalid"

# ---------------- Title & Power controls ----------------
st.markdown("<h1 style='text-align:center; color:#2b2b2b;'>🧮 Multi-Mode Colorful Scientific Calculator</h1>", unsafe_allow_html=True)

cols = st.columns([1, 1, 2])
with cols[0]:
    if st.button("🔴 OFF"):
        st.session_state.power = False
with cols[1]:
    if st.button("🟢 ON"):
        st.session_state.power = True
with cols[2]:
    st.markdown(f"**Mode:** {st.session_state.mode} &nbsp;&nbsp; &nbsp; **ANS:** {st.session_state.ans}")

if not st.session_state.power:
    st.warning("Calculator is OFF. Turn ON to begin.")
    st.stop()

# ---------------- Layout: Tabs ----------------
tab_std, tab_sci, tab_stats, tab_base, tab_mat, tab_calc = st.tabs(
    ["Standard", "Scientific", "Statistics", "Base Conversion", "Matrix/Vector", "Calculus"]
)

# ---------- small helpers to modify expression ----------
def add_to_expr(s: str):
    st.session_state.expr += str(s)

def clear_all():
    st.session_state.expr = ""
    st.session_state.result = None
    st.session_state.ans = 0

def delete_last():
    st.session_state.expr = st.session_state.expr[:-1]

def toggle_sign():
    # toggle sign of current whole expression: if empty add '-' else wrap
    if not st.session_state.expr:
        st.session_state.expr = "-"
    else:
        # try to place unary minus before last token
        st.session_state.expr = f"(-1)*({st.session_state.expr})"

# ---------- Utility to inject per-key CSS (best-effort) ----------
def inject_key_style(key_identifier: str, bg: str):
    """
    Try to color a particular button by emitting a small CSS snippet.
    This is a best-effort approach because Streamlit's internal attributes vary.
    """
    css = f"""
    <style>
    /* best-effort button coloring for key='{key_identifier}' */
    button[aria-label="{key_identifier}"] {{ background: {bg} !important; color: white !important; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ---------------- Standard Tab ----------------
with tab_std:
    st.subheader("📌 Standard Mode")
    st.text_input("Expression:", st.session_state.expr, key="expr_box_std", disabled=True)

    # button grid (numbers green, ops orange, special keys colored)
    rows = [
        ["7", "8", "9", "+"],
        ["4", "5", "6", "-"],
        ["1", "2", "3", "*"],
        ["0", ".", "(", ")"],
        ["AC", "DEL", "(-)", "/"],
        ["STO", "RCL", "ANS", "="]
    ]
    # render rows
    for row in rows:
        cols = st.columns(len(row))
        for i, key in enumerate(row):
            label = key
            if cols[i].button(label, key=f"std_{label}"):
                if label == "AC":
                    clear_all()
                elif label == "DEL":
                    delete_last()
                elif label == "(-)":
                    toggle_sign()
                elif label == "STO":
                    st.session_state.memory = st.session_state.result
                elif label == "RCL":
                    add_to_expr(str(st.session_state.memory))
                elif label == "ANS":
                    add_to_expr(str(st.session_state.ans))
                elif label == "=":
                    st.session_state.result = safe_eval(st.session_state.expr)
                    st.session_state.ans = st.session_state.result
                else:
                    # normal character
                    # map × or ÷ if user uses them — keep simple: '*' and '/'
                    add_to_expr(label if label not in ["×", "÷"] else ("*" if label == "×" else "/"))

    # best-effort coloring (applies globally in many environments)
    # numbers green, operators orange, special different colors.
    st.markdown("""
    <style>
    /* aggressive but generic coloring for buttons */
    div.stButton > button { border: none; }
    </style>
    """, unsafe_allow_html=True)

    if st.session_state.result is not None:
        st.success(f"Result: {st.session_state.result}")

# ---------------- Scientific Tab ----------------
with tab_sci:
    st.subheader("🔬 Scientific Mode")
    st.text_input("Expression (build by pressing keys):", st.session_state.expr, key="expr_box_sci", disabled=True)

    # mode switch
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("MODE: DEG"):
            st.session_state.mode = "DEG"
    with col_b:
        if st.button("MODE: RAD"):
            st.session_state.mode = "RAD"

    sci_rows = [
        ["sin", "cos", "tan", "sqrt", "log"],
        ["asin", "acos", "atan", "ln", "exp"],
        ["sinh", "cosh", "tanh", "fact", "pi"],
        ["cbrt", "root", "square", "recip", "rand"],
        ["nPr", "nCr", "mean", "std", "sum"]
    ]
    for row in sci_rows:
        cols = st.columns(len(row))
        for i, key in enumerate(row):
            label = key
            if cols[i].button(label, key=f"sci_{label}"):
                # append function form for typical usage
                if label in ("pi", "e", "rand"):
                    add_to_expr(label)
                elif label in ("fact", "square", "recip"):
                    # like fact(5) or square(2) -> we append name + "(" to encourage user adding number
                    add_to_expr(label + "(")
                elif label == "root":
                    # root requires two args root(x,y) we'll append root(
                    add_to_expr("root(")
                elif label in ("nPr", "nCr"):
                    add_to_expr(label + "(")
                else:
                    add_to_expr(label + "(")

    # Evaluate / Clear
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("✅ = (Sci)"):
            st.session_state.result = safe_eval(st.session_state.expr)
            st.session_state.ans = st.session_state.result
    with col2:
        if st.button("🧹 AC (Sci)"):
            clear_all()
    with col3:
        if st.button("⌫ DEL (Sci)"):
            delete_last()

    if st.session_state.result is not None:
        st.success(f"Result: {st.session_state.result}")

# ---------------- Statistics Tab ----------------
with tab_stats:
    st.subheader("📊 Statistics Mode")
    nums_text = st.text_area("Enter numbers (comma or space separated):", value="10,20,30,40,50", height=100)
    sep = "," if "," in nums_text else " "
    try:
        data = [float(x.strip()) for x in nums_text.replace(",", " ").split() if x.strip()]
        if st.button("Compute Stats"):
            st.info(f"n = {len(data)}  •  Mean = {np.mean(data)}  •  Std = {np.std(data)}  •  Sum = {np.sum(data)}")
            st.write("Data sample:", data[:20])
    except Exception as e:
        st.error("Invalid numbers. Use comma or space separated numeric values.")

# ---------------- Base Conversion Tab ----------------
with tab_base:
    st.subheader("🔢 Base Conversion")
    num_str = st.text_input("Integer (decimal):", "42")
    base_choice = st.radio("Convert to:", ["Binary", "Octal", "Hexadecimal"])
    try:
        n = int(num_str.strip())
        if base_choice == "Binary":
            st.success(bin(n))
        elif base_choice == "Octal":
            st.success(oct(n))
        elif base_choice == "Hexadecimal":
            st.success(hex(n))
    except:
        st.error("Please enter a valid integer.")

# ---------------- Matrix / Vector Tab ----------------
with tab_mat:
    st.subheader("📐 Matrix & Vector Operations")
    st.markdown("Enter matrices: rows separated by `;` and entries separated by spaces. Example: `1 2;3 4`")

    mat1_txt = st.text_area("Matrix A:", value="1 2;3 4", height=80)
    mat2_txt = st.text_area("Matrix B (optional):", value="5 6;7 8", height=80)

    def parse_matrix(txt):
        try:
            rows = [list(map(float, r.strip().split())) for r in txt.strip().split(";") if r.strip()]
            return np.array(rows)
        except:
            return None

    A = parse_matrix(mat1_txt)
    B = parse_matrix(mat2_txt)

    if A is not None:
        st.write("Matrix A:", A)
    else:
        st.info("Matrix A invalid or empty")

    if B is not None:
        st.write("Matrix B:", B)

    op = st.selectbox("Operation:", ["Add", "Subtract", "Multiply", "Determinant (A)", "Inverse (A)", "Transpose (A)", "Dot Product (vectors)"])
    if st.button("Compute Matrix Operation"):
        try:
            if op == "Add":
                if A is not None and B is not None and A.shape == B.shape:
                    st.success(A + B)
                else:
                    st.error("A and B must be same shape for addition.")
            elif op == "Subtract":
                if A is not None and B is not None and A.shape == B.shape:
                    st.success(A - B)
                else:
                    st.error("A and B must be same shape for subtraction.")
            elif op == "Multiply":
                if A is not None and B is not None and A.shape[1] == B.shape[0]:
                    st.success(np.dot(A, B))
                else:
                    st.error("For multiplication, A columns must equal B rows.")
            elif op == "Determinant (A)":
                if A is not None and A.shape[0] == A.shape[1]:
                    st.success(np.linalg.det(A))
                else:
                    st.error("A must be square.")
            elif op == "Inverse (A)":
                if A is not None and A.shape[0] == A.shape[1]:
                    st.success(np.linalg.inv(A))
                else:
                    st.error("A must be square and invertible.")
            elif op == "Transpose (A)":
                if A is not None:
                    st.success(A.T)
                else:
                    st.error("Provide matrix A.")
            elif op == "Dot Product (vectors)":
                if A is not None and B is not None:
                    flatA = A.flatten()
                    flatB = B.flatten()
                    if flatA.size == flatB.size:
                        st.success(np.dot(flatA, flatB))
                    else:
                        st.error("Vectors (flattened matrices) must be same length.")
        except Exception as e:
            st.error(f"Error: {e}")

# ---------------- Calculus Tab ----------------
with tab_calc:
    st.subheader("📚 Calculus Mode (SymPy)")
    expr_input = st.text_input("Enter function f(x):", value="x**2 + 3*x + 2")
    x = sp.Symbol('x')
    try:
        f = sp.sympify(expr_input)
        st.latex(f"f(x) = {sp.latex(f)}")
        calc_choice = st.selectbox("Operation:", ["Derivative", "Second Derivative", "Integral", "Limit", "Solve Equation"])
        if st.button("Compute Calculus Operation"):
            if calc_choice == "Derivative":
                res = sp.diff(f, x)
                st.success(f"f'(x) = {res}")
            elif calc_choice == "Second Derivative":
                res = sp.diff(f, x, 2)
                st.success(f"f''(x) = {res}")
            elif calc_choice == "Integral":
                res = sp.integrate(f, x)
                st.success(f"∫ f(x) dx = {res} + C")
            elif calc_choice == "Limit":
                point = st.number_input("Limit as x →", value=0.0)
                res = sp.limit(f, x, point)
                st.success(f"lim x→{point} f(x) = {res}")
            elif calc_choice == "Solve Equation":
                res = sp.solve(sp.Eq(f, 0), x)
                st.success(f"Solutions: {res}")
    except Exception as e:
        st.error(f"Invalid expression: {e}")

# ---------------- Footer / quick info ----------------
st.markdown("---")
st.markdown("Tip: build expressions with the buttons, then press `=`. Use `MODE: DEG` or `RAD` for trig functions. For advanced symbolic work enter valid SymPy expressions (e.g. `sin(x)`, `x**3`, `exp(x)`, `log(x)` etc.).")

