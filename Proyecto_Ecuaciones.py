# pip install customtkinter sympy numpy matplotlib scipy pytesseract pillow
import os
import re
import json
import traceback
import datetime

import customtkinter as ctk
from tkinter import filedialog, messagebox
import tkinter as tk

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


SCIPY_DISPONIBLE = True
try:
    from scipy.integrate import solve_ivp
except ImportError:
    SCIPY_DISPONIBLE = False


OCR_DISPONIBLE = True
try:
    import pytesseract
    from PIL import Image, ImageOps

    RUTAS_TESSERACT_POSIBLES = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/opt/homebrew/bin/tesseract",
    ]
    for _ruta in RUTAS_TESSERACT_POSIBLES:
        if os.path.exists(_ruta):
            pytesseract.pytesseract.tesseract_cmd = _ruta
            break


except ImportError:
    OCR_DISPONIBLE = False

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


COLOR_FONDO_PANEL = "#1e1e1e"
COLOR_BORDE = "#333333"
COLOR_ACENTO = "#00ffcc"
COLOR_ACENTO_2 = "#00ffff"
COLOR_ERROR = "#ff5555"
COLOR_OK = "#33ff77"
COLOR_ADVERTENCIA = "#ffcc00"


EJEMPLOS = {
    "Integración directa": "y' = 3*x**2 + 2*x - 1",
    "Separable": "y' = x*y",
    "Lineal 1er orden": "y' + 2*x*y = x",
    "Bernoulli": "y' + y/x = x*y**2",
    "Exacta": "(2*x*y + 3) + (x**2 - 1)*y' = 0",
    "Homogénea (coef. variables)": "y' = (x**2 + y**2)/(x*y)",
    "Riccati (con solución particular conocida)": "y' = y**2 - y/x - 1/x**2",
    "2do orden homogénea (raíces reales distintas)": "y'' - 3*y' + 2*y = 0",
    "2do orden homogénea (raíz repetida)": "y'' - 4*y' + 4*y = 0",
    "2do orden homogénea (raíces complejas)": "y'' + 4*y = 0",
    "2do orden NO homogénea (coef. indeterminados)": "y'' - y = x**2",
    "2do orden NO homogénea (variación parámetros)": "y'' + y = tan(x)",
    "Euler-Cauchy": "x**2*y'' - 2*x*y' + 2*y = 0",
    "3er orden coef. constantes": "y''' - 6*y'' + 11*y' - 6*y = 0",
}


NOMBRES_TECNICOS = {
    "nth_algebraic": "Integración Directa",
    "separable": "Ecuación de Variables Separables",
    "1st_exact": "Ecuación Diferencial Exacta",
    "1st_linear": "Ecuación Lineal de Primer Orden",
    "Bernoulli": "Ecuación de Bernoulli",
    "Riccati_special_minus2": "Ecuación de Riccati (caso especial)",
    "1st_homogeneous_coeff_best": "Ecuación Homogénea (sustitución y = v·x)",
    "1st_homogeneous_coeff_subs_indep_div_dep": "Ecuación Homogénea (sustitución y = v·x)",
    "1st_homogeneous_coeff_subs_dep_div_indep": "Ecuación Homogénea (sustitución x = v·y)",
    "nth_linear_constant_coeff_homogeneous": "Lineal Homogénea de Coeficientes Constantes",
    "nth_linear_constant_coeff_undetermined_coefficients": "Lineal NO Homogénea — Coeficientes Indeterminados",
    "nth_linear_constant_coeff_variation_of_parameters": "Lineal NO Homogénea — Variación de Parámetros",
    "nth_linear_euler_eq_homogeneous": "Ecuación de Euler-Cauchy",
    "nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients": "Euler-Cauchy NO Homogénea (coef. indeterminados)",
    "nth_linear_euler_eq_nonhomogeneous_variation_of_parameters": "Euler-Cauchy NO Homogénea (variación de parámetros)",
}


def parsear_rhs(texto):
    x, y, t, P, v = sp.symbols("x y t P v")
    contexto = {
        "x": x, "y": y, "t": t, "P": P, "v": v,
        "sin": sp.sin, "sen": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "exp": sp.exp, "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt,
        "pi": sp.pi, "E": sp.E,
    }
    texto = texto.replace("^", "**")
    if "=" in texto:
        texto = texto.split("=", 1)[1]
    return sp.sympify(texto, locals=contexto), contexto


def analizar_campo(texto_rhs, puntos, xlim, ylim, var_x="x", var_y="y"):
    expr, ctx = parsear_rhs(texto_rhs)
    sx, sy = ctx[var_x], ctx[var_y]
    f = sp.lambdify((sx, sy), expr, modules=["numpy"])

    def f_seguro(xv, yv):
        with np.errstate(all="ignore"):
            val = np.asarray(f(xv, yv), dtype=complex)
        val = np.where(np.abs(val.imag) < 1e-9, val.real, np.nan)
        return val * np.ones_like(np.asarray(xv, dtype=float) + np.asarray(yv, dtype=float))

    curvas = []
    if SCIPY_DISPONIBLE:
        for (px, py) in puntos:
            tramos = []
            for destino in (xlim[1], xlim[0]):
                try:
                    sol = solve_ivp(
                        lambda a, b: [float(f_seguro(a, b[0]))],
                        (px, destino), [py], max_step=(xlim[1] - xlim[0]) / 400,
                        rtol=1e-8, atol=1e-10,
                    )
                    tramos.append((sol.t, sol.y[0]))
                except Exception:
                    tramos.append((np.array([]), np.array([])))
            curvas.append(((px, py), tramos))

    equilibrios = []
    try:
        soluciones = sp.solve(sp.Eq(expr, 0), sy)
        for s in soluciones:
            if not s.has(sx):
                equilibrios.append(float(s))
    except Exception:
        pass

    return f_seguro, curvas, sorted(set(equilibrios)), expr



class UniversalEDOSolver(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("EDO SOLVER PRO — Sistema Avanzado de Ecuaciones Diferenciales")
        self.geometry("1400x940")
        self.minsize(1150, 780)


        self.x = sp.Symbol("x")
        self.y = sp.Function("y")(self.x)


        self.ultima_edo = None
        self.ultima_solucion = None
        self.ultimas_clasificaciones = None
        self.ultimo_orden = None
        self.ultima_figura_grafica = None


        self.historial = []

        self.setup_layout()


    def setup_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.build_sidebar()
        self.build_tabview()


    def build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=330, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(12, weight=1)

        ctk.CTkLabel(
            self.sidebar, text="EDO SOLVER\nPRO",
            font=ctk.CTkFont(size=26, weight="bold", family="Courier"),
        ).grid(row=0, column=0, padx=20, pady=(30, 15))

        ctk.CTkLabel(
            self.sidebar, text="Ecuación Diferencial:",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=1, column=0, padx=20, sticky="w")

        self.entry_eq = ctk.CTkEntry(
            self.sidebar, placeholder_text="Ej: y' = 3*x**2 + 2*x - 1",
            font=ctk.CTkFont(size=14),
        )
        self.entry_eq.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.entry_eq.insert(0, "y' = 3*x**2 + 2*x - 1")
        self.entry_eq.bind("<Return>", lambda _evt: self.solve_equation())

        ctk.CTkLabel(
            self.sidebar, text="Notación admitida: y', y'', y''', d2y/dx2, x^2, "
            "sin, cos, tan, exp, log, sqrt.",
            font=ctk.CTkFont(size=10), text_color="#888888", wraplength=280,
            justify="left",
        ).grid(row=3, column=0, padx=20, pady=(0, 10), sticky="w")

        ctk.CTkLabel(
            self.sidebar, text="Ejemplos rápidos:", font=ctk.CTkFont(size=12),
        ).grid(row=4, column=0, padx=20, pady=(5, 0), sticky="w")

        self.combo_ejemplos = ctk.CTkOptionMenu(
            self.sidebar, values=list(EJEMPLOS.keys()),
            command=self.cargar_ejemplo,
        )
        self.combo_ejemplos.grid(row=5, column=0, padx=20, pady=(5, 15), sticky="ew")

        self.btn_solve = ctk.CTkButton(
            self.sidebar, text="PROCESAR ALGORITMO", command=self.solve_equation,
            font=ctk.CTkFont(weight="bold"), fg_color="#1f538d", height=38,
        )
        self.btn_solve.grid(row=6, column=0, padx=20, pady=6, sticky="ew")

        self.btn_ocr = ctk.CTkButton(
            self.sidebar, text="ESCANEAR IMAGEN (OCR)",
            command=self.load_and_scan_image,
            font=ctk.CTkFont(weight="bold"),
            fg_color="#006400" if OCR_DISPONIBLE else "#555555",
        )
        self.btn_ocr.grid(row=7, column=0, padx=20, pady=6, sticky="ew")
        if not OCR_DISPONIBLE:
            self.btn_ocr.configure(state="disabled")

        self.btn_export = ctk.CTkButton(
            self.sidebar, text="EXPORTAR REPORTE (.txt)",
            command=self.exportar_reporte, fg_color="#4b3b7a",
        )
        self.btn_export.grid(row=8, column=0, padx=20, pady=6, sticky="ew")

        self.btn_clear = ctk.CTkButton(
            self.sidebar, text="LIMPIAR TODO", command=self.clear_all,
            fg_color="#7a2222",
        )
        self.btn_clear.grid(row=9, column=0, padx=20, pady=6, sticky="ew")

        self.switch_tema = ctk.CTkSwitch(
            self.sidebar, text="Modo claro", command=self.toggle_tema,
        )
        self.switch_tema.grid(row=10, column=0, padx=20, pady=(15, 5), sticky="w")

        self.lbl_status = ctk.CTkLabel(
            self.sidebar, text="", font=ctk.CTkFont(size=11),
            text_color="#888888", wraplength=290, justify="left",
        )
        self.lbl_status.grid(row=13, column=0, padx=20, pady=(0, 15), sticky="sw")
        if not OCR_DISPONIBLE:
            self.lbl_status.configure(
                text="OCR no disponible: instala pytesseract, Pillow y el "
                     "binario de Tesseract-OCR."
            )
        elif not SCIPY_DISPONIBLE:
            self.lbl_status.configure(
                text="SciPy no disponible: la pestaña de PVI Numérico estará "
                     "limitada. Instala scipy para activarla por completo."
            )

    def toggle_tema(self):
        modo = "light" if self.switch_tema.get() else "dark"
        ctk.set_appearance_mode(modo)


    def build_tabview(self):
        self.tabview = ctk.CTkTabview(self, corner_radius=10)
        self.tabview.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        self.tab_resolver = self.tabview.add("Paso a Paso")
        self.tab_grafica = self.tabview.add("Gráfica")
        self.tab_pvi = self.tabview.add("PVI Numérico")
        self.tab_campo = self.tabview.add("Campo de Pendientes")
        self.tab_historial = self.tabview.add("Historial")
        self.tab_ayuda = self.tabview.add("Ayuda / Teoría")

        self.build_tab_resolver()
        self.build_tab_grafica()
        self.build_tab_pvi()
        self.build_tab_campo()
        self.build_tab_historial()
        self.build_tab_ayuda()


    def build_tab_resolver(self):
        tab = self.tab_resolver
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(2, weight=1)


        self.frame_banner = ctk.CTkFrame(
            tab, corner_radius=10, fg_color="#0d2b3d", border_width=1,
            border_color=COLOR_ACENTO_2,
        )
        self.frame_banner.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.frame_banner.grid_columnconfigure(0, weight=1)

        self.lbl_banner_tipo = ctk.CTkLabel(
            self.frame_banner, text="Aún no se ha resuelto ninguna ecuación.",
            font=ctk.CTkFont(size=18, weight="bold"), text_color=COLOR_ACENTO_2,
            anchor="w",
        )
        self.lbl_banner_tipo.grid(row=0, column=0, padx=15, pady=(10, 0), sticky="w")

        self.lbl_banner_metodo = ctk.CTkLabel(
            self.frame_banner, text="", font=ctk.CTkFont(size=13),
            text_color="#cccccc", anchor="w",
        )
        self.lbl_banner_metodo.grid(row=1, column=0, padx=15, pady=(0, 10), sticky="w")


        self.math_frame = ctk.CTkFrame(
            tab, corner_radius=10, height=110, fg_color=COLOR_FONDO_PANEL,
            border_width=1, border_color=COLOR_BORDE,
        )
        self.math_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        self.math_frame.pack_propagate(False)


        self.text_result = ctk.CTkTextbox(
            tab, font=ctk.CTkFont(family="Consolas", size=13),
            text_color=COLOR_ACENTO, fg_color=COLOR_FONDO_PANEL, border_width=1,
            border_color=COLOR_BORDE, wrap="word",
        )
        self.text_result.grid(row=2, column=0, sticky="nsew")


    def build_tab_grafica(self):
        tab = self.tab_grafica
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=0)

        self.graph_frame = ctk.CTkFrame(tab, corner_radius=10)
        self.graph_frame.grid(row=0, column=0, sticky="nsew")

        self.text_grafica_explicacion = ctk.CTkTextbox(
            tab, height=150, font=ctk.CTkFont(size=13),
            text_color="#dddddd", fg_color=COLOR_FONDO_PANEL, wrap="word",
        )
        self.text_grafica_explicacion.grid(row=1, column=0, sticky="ew", pady=(10, 0))


    def build_tab_pvi(self):
        tab = self.tab_pvi
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(4, weight=1)

        ctk.CTkLabel(
            tab, text="Problema de Valor Inicial (PVI) — Solución Numérica",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, columnspan=3, padx=10, pady=(10, 5), sticky="w")

        ctk.CTkLabel(
            tab, text="Úsalo cuando SymPy no encuentra fórmula exacta, o "
            "simplemente para ver la curva concreta que pasa por un punto dado. "
            "Funciona para EDOs de 1er y 2do orden.",
            font=ctk.CTkFont(size=12), text_color="#aaaaaa", wraplength=850,
            justify="left",
        ).grid(row=1, column=0, columnspan=3, padx=10, pady=(0, 15), sticky="w")


        ctk.CTkLabel(tab, text="x0:").grid(row=2, column=0, padx=10, sticky="e")
        self.entry_x0 = ctk.CTkEntry(tab, width=100)
        self.entry_x0.insert(0, "0")
        self.entry_x0.grid(row=2, column=0, padx=(50, 10), pady=5, sticky="w")

        ctk.CTkLabel(tab, text="y(x0):").grid(row=2, column=1, padx=10, sticky="e")
        self.entry_y0 = ctk.CTkEntry(tab, width=100)
        self.entry_y0.insert(0, "1")
        self.entry_y0.grid(row=2, column=1, padx=(70, 10), pady=5, sticky="w")

        ctk.CTkLabel(tab, text="y'(x0) [si aplica]:").grid(row=3, column=0, padx=10, sticky="e")
        self.entry_dy0 = ctk.CTkEntry(tab, width=100)
        self.entry_dy0.insert(0, "0")
        self.entry_dy0.grid(row=3, column=0, padx=(140, 10), pady=5, sticky="w")

        ctk.CTkLabel(tab, text="Rango de x:").grid(row=3, column=1, padx=10, sticky="e")
        self.entry_rango_pvi = ctk.CTkEntry(tab, width=140)
        self.entry_rango_pvi.insert(0, "-5, 5")
        self.entry_rango_pvi.grid(row=3, column=1, padx=(90, 10), pady=5, sticky="w")

        self.btn_resolver_pvi = ctk.CTkButton(
            tab, text="RESOLVER PVI NUMÉRICAMENTE (Runge-Kutta)",
            command=self.solve_pvi_numeric, fg_color="#1f538d",
        )
        self.btn_resolver_pvi.grid(row=2, column=2, rowspan=2, padx=15, pady=5, sticky="nsew")
        if not SCIPY_DISPONIBLE:
            self.btn_resolver_pvi.configure(state="disabled")

        self.graph_frame_pvi = ctk.CTkFrame(tab, corner_radius=10)
        self.graph_frame_pvi.grid(row=4, column=0, columnspan=3, sticky="nsew", padx=10, pady=10)


    def build_tab_historial(self):
        tab = self.tab_historial
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(tab, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        top.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            top, text="Historial de ecuaciones resueltas en esta sesión",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkButton(
            top, text="Borrar historial", width=140, fg_color="#7a2222",
            command=self.borrar_historial,
        ).grid(row=0, column=1, sticky="e")

        self.scroll_historial = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        self.scroll_historial.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.scroll_historial.grid_columnconfigure(0, weight=1)


    def build_tab_ayuda(self):
        tab = self.tab_ayuda
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        texto_ayuda = ctk.CTkTextbox(
            tab, font=ctk.CTkFont(size=13), wrap="word", fg_color=COLOR_FONDO_PANEL,
        )
        texto_ayuda.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        texto_ayuda.insert("end", self.texto_ayuda_completo())
        texto_ayuda.configure(state="disabled")

    def texto_ayuda_completo(self):
        return (
            "GUÍA RÁPIDA — CÓMO ESCRIBIR TU ECUACIÓN\n"
            "========================================\n\n"
            "1) La derivada de y respecto a x se escribe con comilla simple: y'\n"
            "   La segunda derivada: y''      La tercera derivada: y'''\n"
            "   También se acepta: d2y/dx2 (se convierte automáticamente).\n\n"
            "2) Multiplicación SIEMPRE explícita con *:\n"
            "   Correcto: 3*x**2 + 2*x*y      Incorrecto: 3x^2 + 2xy\n"
            "   (El programa intenta corregir automáticamente los casos más comunes, "
            "   pero es más seguro escribirlo bien desde el principio.)\n\n"
            "3) Potencias con doble asterisco: x**2  (no x^2, aunque x^2 también se "
            "   convierte automáticamente).\n\n"
            "4) Funciones disponibles: sin(x), cos(x), tan(x), exp(x), log(x), "
            "   sqrt(x), sinh(x), cosh(x).\n\n"
            "5) Si tu ecuación no tiene signo '=', se asume que todo lo escrito es "
            "   igual a y' (ej. escribir  3*x**2  equivale a  y' = 3*x**2 ).\n\n"
            "¿QUÉ HACE CADA PESTAÑA?\n"
            "========================================\n"
            "• Paso a Paso: identifica el TIPO de ecuación, te explica la TEORÍA "
            "  general de ese método (como si fuera una mini clase) y luego aplica "
            "  ese método a TU ecuación concreta, paso por paso, hasta la solución "
            "  general y = f(x) + C.\n\n"
            "• Gráfica: dibuja la familia de curvas solución (variando las "
            "  constantes C1, C2, ...) y te explica qué significa la forma de esa "
            "  gráfica (crecimiento, decrecimiento, oscilación, estabilidad, etc.).\n\n"
            "• PVI Numérico: para cuando SymPy no encuentra una fórmula exacta, o "
            "  cuando simplemente quieres la curva concreta que pasa por un punto "
            "  (x0, y0) dado. Usa un método numérico (Runge-Kutta, vía SciPy).\n\n"
            "• Historial: guarda cada ecuación que resuelves durante la sesión para "
            "  que puedas volver a consultarla sin escribirla de nuevo.\n\n"
            "CLASIFICACIÓN DE MÉTODOS (RESUMEN PARA REPASAR ANTES DE UN EXAMEN)\n"
            "========================================\n"
            "• Integración directa:      y' = f(x)\n"
            "• Separables:                y' = f(x)*g(y)\n"
            "• Lineales 1er orden:        y' + P(x)y = Q(x)   → factor integrante\n"
            "• Bernoulli:                 y' + P(x)y = Q(x)y^n  → v = y^(1-n)\n"
            "• Exactas:                   M dx + N dy = 0, con ∂M/∂y = ∂N/∂x\n"
            "• Homogéneas (coef. var.):   y' = f(y/x)  → y = v*x\n"
            "• Riccati:                   y' = P(x) + Q(x)y + R(x)y^2\n"
            "• Coef. constantes (n-ésimo orden): a_n y^(n) + ... + a_0 y = g(x)\n"
            "• Euler-Cauchy:              a x^2 y'' + b x y' + c y = 0\n\n"
            "Consejo: si el programa marca error de sintaxis, revisa primero los "
            "signos de multiplicación (*) y los paréntesis. Es la causa más común.\n"
        )


    def cargar_ejemplo(self, nombre):
        self.entry_eq.delete(0, "end")
        self.entry_eq.insert(0, EJEMPLOS[nombre])

    def clear_all(self):
        self.entry_eq.delete(0, "end")
        self.text_result.delete("1.0", "end")
        self.text_grafica_explicacion.delete("1.0", "end")
        for widget in self.math_frame.winfo_children():
            widget.destroy()
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        for widget in self.graph_frame_pvi.winfo_children():
            widget.destroy()
        self.lbl_banner_tipo.configure(text="Aún no se ha resuelto ninguna ecuación.")
        self.lbl_banner_metodo.configure(text="")
        self.ultima_edo = None
        self.ultima_solucion = None
        self.ultimas_clasificaciones = None
        self.ultimo_orden = None

    def borrar_historial(self):
        self.historial = []
        for widget in self.scroll_historial.winfo_children():
            widget.destroy()

    def agregar_historial(self, eq_str, clase_principal, solucion):
        entrada = {
            "hora": datetime.datetime.now().strftime("%H:%M:%S"),
            "ecuacion": eq_str,
            "tipo": NOMBRES_TECNICOS.get(clase_principal, clase_principal),
            "solucion": str(solucion.rhs) if solucion is not None else "N/D",
        }
        self.historial.append(entrada)
        self.render_item_historial(entrada)

    def render_item_historial(self, entrada):
        frame = ctk.CTkFrame(self.scroll_historial, corner_radius=8, fg_color="#222222")
        frame.grid(row=len(self.scroll_historial.winfo_children()), column=0,
                    sticky="ew", pady=4, padx=2)
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            frame, text=f"[{entrada['hora']}]  {entrada['ecuacion']}",
            font=ctk.CTkFont(size=13, weight="bold"), anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(6, 0))

        ctk.CTkLabel(
            frame, text=f"Tipo: {entrada['tipo']}", font=ctk.CTkFont(size=11),
            text_color="#aaaaaa", anchor="w",
        ).grid(row=1, column=0, sticky="w", padx=10)

        ctk.CTkLabel(
            frame, text=f"y(x) = {entrada['solucion']}", font=ctk.CTkFont(size=11),
            text_color=COLOR_ACENTO, anchor="w", wraplength=650, justify="left",
        ).grid(row=2, column=0, sticky="w", padx=10, pady=(0, 6))

        btn = ctk.CTkButton(
            frame, text="Reusar", width=70,
            command=lambda e=entrada: self._reusar_historial(e),
        )
        btn.grid(row=0, column=1, rowspan=3, padx=10, pady=6)

    def _reusar_historial(self, entrada):
        self.entry_eq.delete(0, "end")
        self.entry_eq.insert(0, entrada["ecuacion"])
        self.tabview.set("Paso a Paso")

    def exportar_reporte(self):
        contenido = self.text_result.get("1.0", "end").strip()
        if not contenido:
            messagebox.showinfo("Nada que exportar", "Primero resuelve una ecuación.")
            return
        ruta = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Archivo de texto", "*.txt")],
            initialfile="reporte_edo.txt",
        )
        if not ruta:
            return
        try:
            with open(ruta, "w", encoding="utf-8") as f:
                f.write("REPORTE DE ECUACIÓN DIFERENCIAL\n")
                f.write(f"Generado: {datetime.datetime.now().isoformat(timespec='seconds')}\n")
                f.write(f"Ecuación ingresada: {self.entry_eq.get()}\n")
                f.write("=" * 70 + "\n\n")
                f.write(contenido)
            messagebox.showinfo("Listo", f"Reporte guardado en:\n{ruta}")
        except Exception as e:
            messagebox.showerror("Error al guardar", str(e))


    def sanitizar_texto_matematico(self, texto_crudo):
        texto = texto_crudo.strip()
        texto = texto.replace("\n", " ")
        texto = re.sub(r"\s+", "", texto)


        superindices = {
            "⁰": "^0", "¹": "^1", "²": "^2", "³": "^3", "⁴": "^4",
            "⁵": "^5", "⁶": "^6", "⁷": "^7", "⁸": "^8", "⁹": "^9",
        }
        for sup, rep in superindices.items():
            texto = texto.replace(sup, rep)


        texto = re.sub(r"d\^?3y/?dx\^?3", "y'''", texto, flags=re.IGNORECASE)
        texto = re.sub(r"d\^?2y/?dx\^?2", "y''", texto, flags=re.IGNORECASE)
        texto = re.sub(r"dy/?dx", "y'", texto, flags=re.IGNORECASE)
        texto = texto.replace("y′′′", "y'''").replace("y′′", "y''").replace("y′", "y'")
        texto = texto.replace('y"', "y''").replace("y”", "y''")
        texto = texto.replace("y’", "y'")


        texto = re.sub(r"(?<=[\d\(\+\-\*/=])O(?=[\d\)\+\-\*/=]|$)", "0", texto)
        texto = re.sub(r"(?<=[\d\(\+\-\*/=])l(?=[\d\)\+\-\*/=]|$)", "1", texto)


        texto = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", texto)
        texto = re.sub(r"(\d)\(", r"\1*(", texto)
        texto = re.sub(r"(?<![a-zA-Z])([xy])\(", r"\1*(", texto)
        texto = re.sub(r"\)([a-zA-Z0-9])", r")*\1", texto)

        texto = re.sub(r"\^", "**", texto)
        texto = re.sub(r"[^a-zA-Z0-9\+\-\*\/\=\(\)\'\.]", "", texto)

        if texto.startswith("="):
            texto = "y'" + texto
        return texto

    def preprocesar_imagen_ocr(self, img):
        img = img.convert("L")
        w, h = img.size
        objetivo = 1000
        if max(w, h) < objetivo:
            factor = max(2, int(objetivo / max(w, h)))
            img = img.resize((w * factor, h * factor), Image.LANCZOS)
        img = ImageOps.autocontrast(img)
        img = img.point(lambda p: 255 if p > 140 else 0)
        return img

    def load_and_scan_image(self):
        if not OCR_DISPONIBLE:
            messagebox.showerror("OCR no disponible",
                                  "Instala pytesseract, Pillow y Tesseract-OCR.")
            return
        ruta_imagen = filedialog.askopenfilename(
            filetypes=[("Imágenes", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if not ruta_imagen:
            return

        self.tabview.set("Paso a Paso")
        self.text_result.delete("1.0", "end")
        self.text_result.insert("end", ">> ESCANEANDO IMAGEN (probando varios modos de lectura)...\n")
        self.update()

        try:
            img_original = Image.open(ruta_imagen)
        except Exception as e:
            self.text_result.insert("end", f"\n[!] No se pudo abrir la imagen: {e}\n")
            return

        try:
            img_procesada = self.preprocesar_imagen_ocr(img_original)
        except Exception:
            img_procesada = img_original

        configs = ["--psm 6", "--psm 7", "--psm 4", "--psm 11", "--psm 3"]
        variantes = [("procesada", img_procesada), ("original", img_original)]

        candidatos_crudos = []
        for _nombre_v, img_v in variantes:
            for cfg in configs:
                try:
                    txt = pytesseract.image_to_string(img_v, config=cfg)
                except Exception:
                    continue
                if txt and txt.strip():
                    candidatos_crudos.append(txt)

        if not candidatos_crudos:
            self.text_result.insert(
                "end",
                "\n[!] Tesseract no logró leer ningún texto en la imagen.\n"
                "    Sugerencias: usa una foto más nítida, bien iluminada, recorta la "
                "imagen para que solo salga la ecuación, y evita fondos con rayas o "
                "cuadrícula detrás de los símbolos.\n"
                "    También puedes escribir la ecuación manualmente en el campo de texto.\n",
            )
            return

        mejor_limpio = None
        mejor_crudo = None
        parseo_exitoso = False
        for crudo in candidatos_crudos:
            limpio = self.sanitizar_texto_matematico(crudo)
            if not limpio:
                continue
            if mejor_limpio is None:
                mejor_limpio, mejor_crudo = limpio, crudo
            try:
                self.parse_user_input(limpio)
                mejor_limpio, mejor_crudo = limpio, crudo
                parseo_exitoso = True
                break
            except Exception:
                if len(limpio) > len(mejor_limpio):
                    mejor_limpio, mejor_crudo = limpio, crudo

        if mejor_limpio is None:
            self.text_result.insert(
                "end",
                "\n[!] Se detectó texto en la imagen pero no quedó nada interpretable "
                "como ecuación tras limpiarlo. Revisa que la imagen tenga buena calidad.\n"
                f"    Texto crudo detectado: {candidatos_crudos[0]!r}\n",
            )
            return

        self.entry_eq.delete(0, "end")
        self.entry_eq.insert(0, mejor_limpio)

        if parseo_exitoso:
            self.text_result.insert(
                "end", f"\n>> Ecuación detectada: {mejor_limpio}\n>> Procesando automáticamente...\n"
            )
            self.update()
            self.solve_equation()
        else:
            self.text_result.insert(
                "end",
                f"\n>> Mejor intento de lectura: {mejor_limpio}\n"
                "[!] No logré interpretar esto como una ecuación válida automáticamente.\n"
                "    Corrige el texto en el campo de arriba (fíjate en signos, exponentes "
                "y paréntesis) y pulsa 'PROCESAR ALGORITMO'.\n",
            )


    def parse_user_input(self, eq_str):
        eq_str = eq_str.strip()
        if not eq_str:
            raise ValueError("La ecuación está vacía.")


        eq_str = eq_str.replace("^", "**")

        if "=" in eq_str:
            izq_str, der_str = eq_str.split("=", 1)
        else:
            izq_str, der_str = eq_str, "0"


        for viejo, nuevo in (
            ("y'''", "Derivative(y, x, 3)"),
            ("y''", "Derivative(y, x, 2)"),
            ("y'", "Derivative(y, x)"),
        ):
            izq_str = izq_str.replace(viejo, nuevo)
            der_str = der_str.replace(viejo, nuevo)

        contexto = {
            "y": self.y, "x": self.x, "Derivative": sp.Derivative,
            "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp,
            "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt,
            "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
            "pi": sp.pi, "E": sp.E,
        }
        try:
            lhs = sp.sympify(izq_str, locals=contexto)
            rhs = sp.sympify(der_str, locals=contexto)
        except Exception as e:
            raise ValueError(f"No se pudo interpretar la ecuación: {e}")

        return sp.Eq(lhs, rhs)

    def detectar_orden(self, edo):
        try:
            return sp.ode_order(edo.lhs - edo.rhs, self.y)
        except Exception:
            return None


    def render_math_display(self, eq_sympy):
        for widget in self.math_frame.winfo_children():
            widget.destroy()

        latex_str = sp.latex(eq_sympy)
        latex_str = latex_str.replace(r"\frac{d}{d x} y{\left(x \right)}", r"\frac{dy}{dx}")
        latex_str = latex_str.replace(
            r"\frac{d^{2}}{d x^{2}} y{\left(x \right)}", r"\frac{d^2y}{dx^2}"
        )
        latex_str = latex_str.replace(
            r"\frac{d^{3}}{d x^{3}} y{\left(x \right)}", r"\frac{d^3y}{dx^3}"
        )
        latex_str = latex_str.replace(r"y{\left(x \right)}", "y")

        fig = plt.Figure(figsize=(9, 1.3), facecolor=COLOR_FONDO_PANEL)
        ax = fig.add_subplot(111)
        ax.axis("off")
        try:
            ax.text(0.5, 0.5, f"${latex_str}$", fontsize=22, color="white",
                     ha="center", va="center")
        except Exception:
            ax.text(0.5, 0.5, str(eq_sympy), fontsize=16, color="white",
                     ha="center", va="center")

        canvas = FigureCanvasTkAgg(fig, master=self.math_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


    def polinomio_caracteristico(self, edo):
        r = sp.Symbol("r")
        expr_full = sp.expand(edo.lhs - edo.rhs)
        terminos = sp.Add.make_args(expr_full)
        homog_terms = [t for t in terminos if t.has(self.y)]
        homog_expr = sum(homog_terms) if homog_terms else expr_full

        orden = sp.ode_order(homog_expr, self.y)
        trial = sp.exp(r * self.x)
        expr = homog_expr
        for k in range(orden, 0, -1):
            expr = expr.subs(sp.Derivative(self.y, self.x, k), r ** k * trial)
        expr = expr.subs(self.y, trial)
        expr = sp.simplify(expr / trial)
        return sp.Poly(expr, r), r

    def analizar_raices(self, raices):
        texto = ""
        hay_repetidas = len(set(raices)) < len(raices)
        complejas = [rt for rt in raices if not rt.is_real]
        reales = [rt for rt in raices if rt.is_real]

        if hay_repetidas:
            texto += ("   > Hay raíces repetidas: para mantener independencia "
                       "lineal, la segunda solución de esa raíz se multiplica "
                       "por x (x^2 si se repite 3 veces, etc.).\n")
        if complejas:
            texto += ("   > Hay raíces complejas conjugadas (a ± bi): la "
                       "solución incluye e^(a x)[cos(b x) + sen(b x)] — "
                       "comportamiento oscilatorio.\n")
        if reales:
            negativas = [rt for rt in reales if rt.is_number and rt < 0]
            positivas = [rt for rt in reales if rt.is_number and rt > 0]
            if negativas and not positivas:
                texto += "   > Todas las raíces reales son negativas: el sistema es ESTABLE (decae a 0).\n"
            elif positivas:
                texto += "   > Hay raíces reales positivas: el sistema es INESTABLE (crece sin límite).\n"
        return texto


    def generate_graph_explanation(self, expresion, constantes):
        num_c = len(constantes)
        expr_str = str(expresion)
        explicacion = "ANÁLISIS DE LA GRÁFICA (explicado fácil):\n"

        if num_c == 1:
            explicacion += f"   > Cada curva de color es una 'Solución Particular', obtenida al fijar un valor distinto de {constantes[0]}.\n"
            explicacion += "   > Fíjate que las curvas NUNCA se cruzan entre sí: eso confirma en la práctica el Teorema de Existencia y Unicidad (por cada punto del plano pasa una sola curva solución).\n"
        elif num_c >= 2:
            explicacion += f"   > La familia de curvas depende de {num_c} constantes ({', '.join(map(str, constantes))}).\n"
            explicacion += f"   > Para poder dibujarlo en 2D, se fijó {constantes[1]}=1 y se varió {constantes[0]}.\n"
        else:
            explicacion += "   > Esta es la ÚNICA solución posible (no depende de constantes arbitrarias, así que no hay una 'familia' de curvas, solo una).\n"

        if "exp" in expr_str:
            explicacion += ("   > Hay términos exponenciales: si el exponente es positivo la curva "
                             "crece muy rápido (dispara hacia arriba); si es negativo, decae hacia "
                             "cero suavemente (como un enfriamiento o una descarga).\n")
        if "sin" in expr_str or "cos" in expr_str:
            explicacion += "   > Hay senos/cosenos: comportamiento oscilatorio, tipo resorte o péndulo, que sube y baja de forma periódica.\n"
        if re.search(r"x\*\*[2-9]", expr_str):
            explicacion += "   > Hay términos polinómicos de grado 2 o más: la curva se curva como una parábola o algo más pronunciado, no es una línea recta.\n"
        if "log" in expr_str:
            explicacion += "   > Hay un logaritmo: la curva crece muy lento a la larga y tiene un límite de dominio (x no puede ser negativo o cero, según el argumento).\n"

        return explicacion


    def teoria_general(self, clase_principal):
        if "nth_algebraic" in clase_principal:
            return (
                "   > ¿QUÉ ES?: Es el caso más simple de todos. La derivada y' ya viene "
                "sola, igualada directamente a una función que solo depende de x "
                "(no aparece y en el lado derecho).\n"
                "   > ¿CÓMO SE RECONOCE?: La ecuación se ve como  y' = f(x)  — no hay "
                "ninguna 'y' suelta del lado derecho, solo x, números y operaciones.\n"
                "   > FÓRMULA GENERAL:  dy/dx = f(x)  →  y(x) = ∫f(x)dx + C\n"
                "   > ALGORITMO PASO A PASO (piénsalo como una receta que siempre "
                "funciona para este tipo):\n"
                "      1) Verifica que y' esté sola en un lado y solo aparezca x en el otro.\n"
                "      2) 'Separa' multiplicando ambos lados por dx: dy = f(x)dx.\n"
                "      3) Integra ambos lados: ∫dy = ∫f(x)dx.\n"
                "      4) El lado izquierdo SIEMPRE da y. El derecho se resuelve con las "
                "reglas normales de integración (potencias, trigonométricas, exponenciales...).\n"
                "      5) No olvides sumar +C al final: sin la constante, la respuesta está incompleta.\n"
            )

        if "separable" in clase_principal:
            return (
                "   > ¿QUÉ ES?: Aquí SÍ aparece y en el lado derecho, pero la buena "
                "noticia es que se puede reacomodar de forma que todos los términos "
                "con y (junto a dy) queden de un lado, y todos los términos con x "
                "(junto a dx) queden del otro, sin que se mezclen entre sí.\n"
                "   > ¿CÓMO SE RECONOCE?: Puedes escribir y' como un producto o "
                "cociente de una función solo de x por una función solo de y: "
                "y' = f(x)·g(y).\n"
                "   > FÓRMULA GENERAL:  dy/g(y) = f(x)dx\n"
                "   > ALGORITMO:\n"
                "      1) Escribe y' como dy/dx (como si fuera una fracción normal).\n"
                "      2) Mueve todos los términos con y (y su dy) a la izquierda, y "
                "todos los términos con x (y su dx) a la derecha, usando álgebra "
                "normal (multiplicar, dividir, pasar términos).\n"
                "      3) Integra cada lado POR SEPARADO: ∫(1/g(y))dy = ∫f(x)dx.\n"
                "      4) Resuelve ambas integrales con las reglas normales.\n"
                "      5) Suma la constante C (basta con ponerla en un solo lado).\n"
                "      6) Si puedes, despeja y para dejar la solución explícita "
                "y(x) = ...; si no se puede despejar, se deja implícita (una "
                "ecuación en x e y sin despejar, que también es válida).\n"
            )

        if "Bernoulli" in clase_principal:
            return (
                "   > ¿QUÉ ES?: Se PARECE a una ecuación lineal, pero tiene un término "
                "extra con y elevada a una potencia distinta de 0 o 1, lo que la hace "
                "NO lineal a simple vista.\n"
                "   > ¿CÓMO SE RECONOCE?: Tiene la forma  y' + P(x)·y = Q(x)·y^n , con "
                "n distinto de 0 y de 1.\n"
                "   > FÓRMULA GENERAL:  y' + P(x)y = Q(x)y^n\n"
                "   > ALGORITMO (el truco es un cambio de variable):\n"
                "      1) Identifica P(x), Q(x) y el exponente n.\n"
                "      2) Divide TODA la ecuación entre y^n.\n"
                "      3) Haz el cambio de variable v = y^(1-n). Esto convierte la y "
                "elevada a una potencia rara en una nueva variable v que sí se "
                "comporta de forma lineal.\n"
                "      4) Deriva v respecto a x (regla de la cadena) y sustituye: la "
                "ecuación se transforma en v' + (1-n)P(x)v = (1-n)Q(x), que YA ES "
                "lineal en v (se resuelve con factor integrante).\n"
                "      5) Resuelve esa lineal para encontrar v(x).\n"
                "      6) 'Deshaz' el cambio: como v = y^(1-n), despeja y en función "
                "de v para obtener la solución final y(x).\n"
            )

        if "Riccati" in clase_principal:
            return (
                "   > ¿QUÉ ES?: Una ecuación de primer orden con un término cuadrático "
                "en y, de la forma  y' = P(x) + Q(x)y + R(x)y^2 . Es de las más "
                "difíciles de primer orden: en general NO tiene solución elemental "
                "a menos que ya conozcamos (o adivinemos) UNA solución particular.\n"
                "   > ¿CÓMO SE RECONOCE?: Aparece y al cuadrado multiplicada por una "
                "función de x, junto con términos lineales en y y un término libre.\n"
                "   > FÓRMULA GENERAL:  y' = P(x) + Q(x)y + R(x)y^2\n"
                "   > ALGORITMO (si se conoce una solución particular y1):\n"
                "      1) Verifica o busca una solución particular y1(x) que cumpla "
                "la ecuación (a veces se puede adivinar probando funciones simples "
                "como y1 = 1/x, constantes, etc.).\n"
                "      2) Haz el cambio y = y1 + 1/v (v es una nueva función de x).\n"
                "      3) Sustituye y deriva: la ecuación se transforma en una "
                "ecuación LINEAL de primer orden en v, que se resuelve con factor "
                "integrante.\n"
                "      4) Deshaz el cambio para obtener y(x) = y1(x) + 1/v(x).\n"
                "   SymPy automatiza todo este proceso quand puede identificar una "
                "solución particular internamente.\n"
            )

        if "1st_exact" in clase_principal:
            return (
                "   > ¿QUÉ ES?: Se escribe como M(x,y)dx + N(x,y)dy = 0, y existe una "
                "función F(x,y) 'escondida' tal que M y N son sus derivadas parciales "
                "respecto a x e y respectivamente.\n"
                "   > ¿CÓMO SE RECONOCE?: Se escribe en la forma M(x,y)dx + "
                "N(x,y)dy = 0, y se cumple la condición de exactitud: "
                "∂M/∂y = ∂N/∂x.\n"
                "   > FÓRMULA GENERAL:  M(x,y)dx + N(x,y)dy = 0 , con ∂M/∂y = ∂N/∂x\n"
                "   > ALGORITMO:\n"
                "      1) Identifica M(x,y) (lo que acompaña a dx) y N(x,y) (lo que "
                "acompaña a dy).\n"
                "      2) Comprueba ∂M/∂y = ∂N/∂x. Si NO se cumple, no es exacta tal "
                "cual (a veces se arregla con un 'factor integrante').\n"
                "      3) Integra M respecto a x (tratando y como constante) para "
                "obtener F(x,y), dejando una función desconocida h(y) en vez de "
                "constante.\n"
                "      4) Deriva ese F respecto a y, iguala a N(x,y), y despeja "
                "h'(y). Integra h'(y) respecto a y para hallar h(y).\n"
                "      5) La solución general queda de forma IMPLÍCITA: F(x,y) = C.\n"
            )

        if "1st_homogeneous_coeff" in clase_principal:
            return (
                "   > ¿QUÉ ES?: Si multiplicas x por t y y por t (escalas ambas "
                "variables) y la ecuación 'no cambia de forma', se dice que es "
                "homogénea de grado 0.\n"
                "   > ¿CÓMO SE RECONOCE?: y' = f(x,y) donde f(tx,ty) = f(x,y). Suele "
                "verse como un cociente de dos polinomios del mismo grado en x e y.\n"
                "   > FÓRMULA GENERAL:  y' = f(y/x)\n"
                "   > ALGORITMO:\n"
                "      1) Verifica que la ecuación se pueda escribir como función "
                "solo de (y/x).\n"
                "      2) Haz el cambio v = y/x, es decir y = v·x (v es función de x).\n"
                "      3) Deriva y = v·x con regla del producto: y' = v + x·v'.\n"
                "      4) Sustituye: ahora queda una ecuación SEPARABLE en v y x.\n"
                "      5) Resuélvela separando variables e integrando.\n"
                "      6) Deshaz el cambio: reemplaza v por y/x.\n"
            )

        if "1st_linear" in clase_principal:
            return (
                "   > ¿QUÉ ES?: y y su derivada y' aparecen solo elevadas a la "
                "primera potencia (sin multiplicarse entre sí ni con exponentes raros).\n"
                "   > ¿CÓMO SE RECONOCE?: Se puede acomodar en la forma "
                "y' + P(x)y = Q(x).\n"
                "   > FÓRMULA GENERAL:  y' + P(x)y = Q(x)\n"
                "   > ALGORITMO (método del factor integrante):\n"
                "      1) Acomoda la ecuación en la forma estándar y' + P(x)y = Q(x).\n"
                "      2) Identifica P(x) (lo que multiplica a y, no a y').\n"
                "      3) Calcula el factor integrante: u(x) = e^(∫P(x)dx).\n"
                "      4) Multiplica TODA la ecuación por u(x). El lado izquierdo se "
                "convierte automáticamente en la derivada de un producto: "
                "(u(x)·y)' = u(x)·Q(x).\n"
                "      5) Integra ambos lados respecto a x.\n"
                "      6) Despeja y dividiendo entre u(x). No olvides sumar +C.\n"
            )

        if "euler" in clase_principal.lower():
            return (
                "   > ¿QUÉ ES?: Ecuación lineal donde los coeficientes NO son "
                "constantes, sino potencias de x que coinciden exactamente con el "
                "orden de cada derivada.\n"
                "   > ¿CÓMO SE RECONOCE?: Forma  aₙxⁿy⁽ⁿ⁾ + ... + a₁xy' + a₀y = 0.\n"
                "   > FÓRMULA GENERAL (caso 2do orden):  ax²y'' + bxy' + cy = 0\n"
                "   > ALGORITMO:\n"
                "      1) Verifica que cada término tenga x elevado al mismo grado "
                "que el orden de la derivada que acompaña.\n"
                "      2) Propone una solución de prueba y = xᵐ.\n"
                "      3) Sustituye y = xᵐ y sus derivadas; todos los términos "
                "quedan con el factor común xᵐ, que se cancela.\n"
                "      4) Queda una ecuación polinómica en m (la 'ecuación "
                "indicial'). Resuélvela para hallar los valores de m.\n"
                "      5) Según el tipo de raíces se arma la solución general "
                "combinando xᵐ¹, xᵐ² (o xᵐ·ln(x) si se repiten, o "
                "x^a[cos(b·ln x)+sin(b·ln x)] si son complejas).\n"
            )

        if "nth_linear_constant_coeff" in clase_principal:
            es_no_homogenea = "undetermined" in clase_principal or "variation" in clase_principal
            base = (
                "   > ¿QUÉ ES?: y y sus derivadas de distintos órdenes se suman "
                "multiplicadas por NÚMEROS fijos (no por funciones de x).\n"
                "   > ¿CÓMO SE RECONOCE?: Forma  aₙy⁽ⁿ⁾ + ... + a₁y' + a₀y = g(x), "
                "donde todos los 'a' son constantes.\n"
                "   > FÓRMULA GENERAL:  aₙy⁽ⁿ⁾ + ... + a₁y' + a₀y = g(x)\n"
                "   > ALGORITMO (parte homogénea, siempre se hace primero):\n"
                "      1) Propone una solución de prueba y = e^(rx).\n"
                "      2) Sustituye y = e^(rx) y sus derivadas; el factor e^(rx) se "
                "cancela en toda la ecuación (nunca es cero).\n"
                "      3) Queda un POLINOMIO en r llamado 'ecuación característica'. "
                "Se resuelve (factorizando, fórmula general, etc.) para encontrar "
                "sus raíces.\n"
                "      4) Según el tipo de raíz se arma cada 'bloque' de la solución:\n"
                "         • Raíz real r (no repetida)      → aporta el término  C·e^(rx)\n"
                "         • Raíz real repetida k veces      → aporta C₁e^(rx), "
                "C₂x·e^(rx), ..., hasta Cₖx^(k-1)e^(rx)\n"
                "         • Par de raíces complejas a±bi     → aporta "
                "e^(ax)[C₁cos(bx) + C₂sen(bx)]\n"
                "      5) La solución homogénea y_h es la suma de todos esos "
                "bloques.\n"
            )
            if es_no_homogenea:
                base += (
                    "   > PARTE NO HOMOGÉNEA (cuando g(x) ≠ 0):\n"
                    "      6) Se busca una solución particular y_p que 'compense' el "
                    "término g(x) del lado derecho:\n"
                    "         • Coeficientes Indeterminados: si g(x) es polinomio, "
                    "exponencial, seno/coseno o combinación, se propone una y_p con "
                    "la MISMA forma (coeficientes por determinar) y se sustituye en "
                    "la ecuación original para hallarlos.\n"
                    "         • Variación de Parámetros: método más general (sirve "
                    "para cualquier g(x)), usa las soluciones de la parte homogénea "
                    "para construir y_p mediante integrales.\n"
                    "      7) La solución general final es siempre  y = y_h + y_p.\n"
                )
            return base

        return (
            "   > Este tipo de ecuación no tiene, en este programa, una explicación "
            "conceptual paso a paso propia; SymPy la resolvió con métodos simbólicos "
            "avanzados (series, funciones especiales, transformaciones, etc.) que van "
            "más allá de un algoritmo manual típico de un curso introductorio.\n"
        )


    def generate_universal_steps(self, edo, clasificaciones, solucion):
        ocultos = ("factorable", "lie_group", "power_series")
        candidatos = [
            c for c in clasificaciones
            if not any(o in c for o in ocultos) and not c.endswith("_Integral")
        ]
        clase_principal = candidatos[0] if candidatos else clasificaciones[0]

        nombre_mostrar = NOMBRES_TECNICOS.get(
            clase_principal, clase_principal.replace("_", " ").upper()
        )

        orden = self.detectar_orden(edo)
        self.lbl_banner_tipo.configure(text=f"TIPO DETECTADO: {nombre_mostrar}")
        self.lbl_banner_metodo.configure(
            text=f"Orden de la EDO: {orden}    |    "
                 f"Otros métodos aplicables: {', '.join(clasificaciones[:4])}"
        )

        output = f"1. TIPO DE ECUACIÓN DETECTADO:\n   > {nombre_mostrar}\n"
        output += f"   > Orden de la ecuación: {orden}\n"
        if len(clasificaciones) > 1:
            output += f"   > (SymPy también reconoce estos métodos aplicables: {', '.join(clasificaciones[:5])})\n"
        output += "\n"

        output += "2. TEORÍA GENERAL DEL MÉTODO (antes de aplicarlo a tu ecuación):\n"
        output += self.teoria_general(clase_principal)
        output += "\n" + "·" * 70 + "\n"
        output += "AHORA, APLICADO PASO A PASO A TU ECUACIÓN CONCRETA:\n\n"


        if "nth_algebraic" in clase_principal:
            try:
                aislado = sp.solve(edo, sp.Derivative(self.y, self.x))[0]
                integral_calculada = sp.integrate(aislado, self.x)
                output += "3. AISLAR y':\n   > Se despeja y' para dejar dy/dx = f(x).\n\n"
                output += "4. CÁLCULO DE LA INTEGRAL:\n"
                output += f"   > ∫ dy = ∫ ({sp.pretty(aislado, use_unicode=False)}) dx\n"
                output += f"   > y(x) = {sp.pretty(integral_calculada, use_unicode=False)} + C\n\n"
            except Exception:
                output += "3-5. INTEGRACIÓN:\n   > Se aplican reglas de integración en ambos lados.\n\n"


        elif "separable" in clase_principal:
            output += ("3. SEPARACIÓN DE VARIABLES:\n   > Se reordenan los términos "
                       "algebraicamente para dejar todo lo que depende de y (con dy) de "
                       "un lado, y todo lo que depende de x (con dx) del otro: "
                       "g(y) dy = f(x) dx.\n\n")
            output += ("4. INTEGRACIÓN:\n   > Se integra cada lado por separado:\n"
                       "   > ∫ g(y) dy = ∫ f(x) dx + C\n\n")


        elif "Bernoulli" in clase_principal:
            output += ("3. IDENTIFICACIÓN:\n   > La ecuación tiene forma "
                       "y' + P(x)y = Q(x)y^n (no lineal por el término y^n).\n\n")
            output += ("4. SUSTITUCIÓN:\n   > Se define v = y^(1-n). Esto convierte la "
                       "ecuación en una EDO LINEAL en v: "
                       "v' + (1-n)P(x)v = (1-n)Q(x).\n\n")
            output += ("5. RESOLUCIÓN Y REGRESO A y:\n   > Se resuelve la lineal en v "
                       "con factor integrante y luego se deshace la sustitución "
                       "v = y^(1-n).\n\n")


        elif "Riccati" in clase_principal:
            output += ("3. IDENTIFICACIÓN:\n   > La ecuación tiene un término y^2: "
                       "y' = P(x) + Q(x)y + R(x)y^2.\n\n")
            output += ("4. SUSTITUCIÓN CON SOLUCIÓN PARTICULAR:\n   > SymPy busca (o usa) "
                       "una solución particular y1(x) y aplica el cambio y = y1 + 1/v "
                       "para reducirla a una ecuación lineal en v.\n\n")
            output += ("5. RESOLUCIÓN Y REGRESO A y:\n   > Se resuelve la lineal en v y "
                       "se deshace el cambio para obtener y(x).\n\n")


        elif "1st_exact" in clase_principal:
            output += ("3. VERIFICACIÓN DE EXACTITUD:\n   > Se escribe como "
                       "M(x,y)dx + N(x,y)dy = 0 y se comprueba que "
                       "∂M/∂y = ∂N/∂x.\n\n")
            output += ("4. CONSTRUCCIÓN DE F(x,y):\n   > Se integra M respecto a x (o N "
                       "respecto a y) y se completa con la función faltante que dependa "
                       "de la otra variable.\n\n")
            output += "5. SOLUCIÓN IMPLÍCITA:\n   > La solución general queda como F(x,y) = C.\n\n"


        elif "1st_homogeneous_coeff" in clase_principal:
            output += ("3. IDENTIFICACIÓN:\n   > y' = f(x,y) donde f(tx,ty) = f(x,y): "
                       "la ecuación es homogénea en el sentido de que ambos lados "
                       "escalan igual.\n\n")
            output += ("4. SUSTITUCIÓN:\n   > Se hace y = v·x (con v función de x), de "
                       "modo que y' = v + x·v'. Esto separa las variables v y x.\n\n")
            output += "5. RESOLUCIÓN Y REGRESO A y:\n   > Se integra en v y x, y luego se reemplaza v = y/x.\n\n"


        elif "1st_linear" in clase_principal:
            try:
                dydx = sp.Derivative(self.y, self.x)
                expr = sp.expand(edo.lhs - edo.rhs)
                coef_y = expr.coeff(self.y)
                coef_dy = expr.coeff(dydx)
                P = sp.simplify(coef_y / coef_dy) if coef_dy != 0 else coef_y
                factor = sp.exp(sp.integrate(P, self.x))
                output += f"3. FORMA ESTÁNDAR:\n   > y' + P(x)y = Q(x), con P(x) = {sp.pretty(P, use_unicode=False)}\n\n"
                output += f"4. FACTOR INTEGRANTE:\n   > u(x) = e^(∫P dx) = {sp.pretty(sp.simplify(factor), use_unicode=False)}\n\n"
                output += "5. INTEGRACIÓN:\n   > Se multiplica toda la ecuación por u(x); el lado izquierdo se convierte en (u·y)', y se integra en ambos lados.\n\n"
            except Exception:
                output += "3-6. FACTOR INTEGRANTE:\n   > Se calcula u(x) = e^(∫P(x)dx) y se integra (u·y)'.\n\n"


        elif "euler" in clase_principal.lower():
            output += ("3. IDENTIFICACIÓN:\n   > Ecuación de la forma "
                       "x^n·y^(n) + ... = 0: los coeficientes son potencias de x que "
                       "combinan con el orden de la derivada.\n\n")
            output += ("4. SUSTITUCIÓN:\n   > Se propone y = x^m (o el cambio de "
                       "variable x = e^t), obteniendo una ecuación característica en m.\n\n")
            output += f"5. SOLUCIÓN GENERAL:\n   > y(x) = {solucion.rhs}\n\n"


        elif "nth_linear_constant_coeff" in clase_principal:
            try:
                poly, r = self.polinomio_caracteristico(edo)
                raices = sp.roots(poly, r)
                lista_raices = []
                for raiz, mult in raices.items():
                    lista_raices.extend([raiz] * mult)
                output += f"3. ECUACIÓN CARACTERÍSTICA:\n   > {sp.Eq(poly.as_expr(), 0)}\n\n"
                output += f"4. RAÍCES:\n   > r = {lista_raices}\n"
                output += self.analizar_raices(lista_raices)
                output += "\n"
                if "undetermined" in clase_principal or "variation" in clase_principal:
                    metodo = ('coeficientes indeterminados' if 'undetermined' in clase_principal
                              else 'variación de parámetros')
                    output += ("5. SOLUCIÓN PARTICULAR (término no homogéneo):\n   > Como "
                               "el lado derecho no es cero, se suma una solución "
                               "particular y_p a la solución homogénea y_h. SymPy la "
                               f"calculó por {metodo}.\n\n")
            except Exception:
                output += "3-5. ECUACIÓN CARACTERÍSTICA:\n   > Se plantea el polinomio característico y se hallan sus raíces.\n\n"

        else:
            output += ("3. DESARROLLO ANALÍTICO:\n   > Este tipo de EDO no tiene un "
                       "desglose algebraico manual estándar implementado aquí; SymPy "
                       "resolvió la ecuación internamente con métodos simbólicos "
                       "avanzados.\n\n")

        output += f"SOLUCIÓN GENERAL FINAL:\n   > y(x) = {solucion.rhs}\n\n"
        output += "-" * 70 + "\n"

        constantes = sorted(
            [s for s in solucion.rhs.free_symbols if str(s).startswith("C")],
            key=str,
        )
        explicacion_grafica = self.generate_graph_explanation(solucion.rhs, constantes)
        output += explicacion_grafica

        self.text_grafica_explicacion.delete("1.0", "end")
        self.text_grafica_explicacion.insert("end", explicacion_grafica)

        return output, clase_principal


    def solve_equation(self):
        eq_str = self.entry_eq.get().strip()
        self.tabview.set("Paso a Paso")
        self.text_result.delete("1.0", "end")
        self.lbl_banner_tipo.configure(text="Procesando...")
        self.lbl_banner_metodo.configure(text="")

        if not eq_str:
            self.text_result.insert("end", "[!] Escribe una ecuación primero.\n")
            self.lbl_banner_tipo.configure(text="Sin ecuación ingresada.")
            return

        try:
            edo = self.parse_user_input(eq_str)
        except Exception as e:
            self.text_result.insert(
                "end",
                "[!] ERROR DE SINTAXIS al interpretar la ecuación.\n"
                "    Usa notación explícita: 3*x en vez de 3x, x**2 en vez de x^2,\n"
                "    y' para dy/dx, y'' para d²y/dx².\n"
                f"    Detalle técnico: {e}\n",
            )
            self.lbl_banner_tipo.configure(text="Error de sintaxis.")
            return

        try:
            self.render_math_display(edo)
        except Exception:
            pass

        try:
            clasificaciones = sp.classify_ode(edo, self.y)
            if not clasificaciones:
                self.text_result.insert(
                    "end", "[!] SymPy no pudo clasificar esta ecuación con los métodos disponibles.\n"
                )
                self.lbl_banner_tipo.configure(text="No se pudo clasificar la ecuación.")
                return
        except Exception as e:
            self.text_result.insert("end", f"[!] No se pudo clasificar la ecuación: {e}\n")
            self.lbl_banner_tipo.configure(text="Error al clasificar.")
            return

        self.ultima_edo = edo
        self.ultimas_clasificaciones = clasificaciones
        self.ultimo_orden = self.detectar_orden(edo)

        try:
            solucion = sp.dsolve(edo, self.y)
            if isinstance(solucion, list):
                solucion = solucion[0]
        except NotImplementedError:
            self.text_result.insert(
                "end",
                "[!] Esta ecuación NO tiene una solución analítica cerrada conocida por "
                "SymPy (esto le pasa a muchísimas EDOs reales, no es un fallo del "
                "programa).\n"
                f"    Clasificación detectada: {clasificaciones}\n"
                "    Alternativa: ve a la pestaña 'PVI Numérico', pon una condición "
                "inicial y resuélvela con Runge-Kutta.\n",
            )
            self.lbl_banner_tipo.configure(
                text="Sin solución analítica cerrada — usa la pestaña PVI Numérico."
            )
            return
        except Exception as e:
            self.text_result.insert("end", f"[!] ERROR al resolver: {e}\n{traceback.format_exc(limit=1)}\n")
            self.lbl_banner_tipo.configure(text="Error al resolver la ecuación.")
            return

        self.ultima_solucion = solucion
        clase_principal = None
        try:
            paso_a_paso, clase_principal = self.generate_universal_steps(edo, clasificaciones, solucion)
            self.text_result.insert("end", paso_a_paso)
        except Exception as e:
            self.text_result.insert(
                "end", f"y(x) = {solucion.rhs}\n\n[!] (No se pudo generar la explicación detallada: {e})\n"
            )

        try:
            self.plot_graph(solucion.rhs)
        except Exception as e:
            self.text_result.insert("end", f"\n[!] No se pudo graficar: {e}\n")

        if clase_principal:
            self.agregar_historial(eq_str, clase_principal, solucion)


    def plot_graph(self, expresion):
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#2b2b2b")
        ax.set_facecolor(COLOR_FONDO_PANEL)
        x_vals = np.linspace(-3, 3, 400)

        constantes = sorted(
            [s for s in expresion.free_symbols if str(s).startswith("C")], key=str
        )
        colores = ["#ff00ff", "#00ffff", "#00ff00", "#ffff00", "#ff3333"]
        valores_c = [-2, -1, 0, 1, 2]

        curvas_y = []

        def evaluar(expr_num):
            try:
                f_num = sp.lambdify(self.x, expr_num, modules=["numpy"])
                y_vals = f_num(x_vals)
                if np.isscalar(y_vals) or isinstance(y_vals, (int, float)):
                    y_vals = np.full_like(x_vals, float(y_vals))
                y_vals = np.array(y_vals, dtype=complex)
                y_vals = np.where(np.abs(y_vals.imag) < 1e-6, y_vals.real, np.nan)
                return y_vals
            except Exception:
                return None

        try:
            if len(constantes) == 1:
                for idx, c in enumerate(valores_c):
                    y_vals = evaluar(expresion.subs(constantes[0], c))
                    if y_vals is not None:
                        ax.plot(x_vals, y_vals, label=f"{constantes[0]}={c}",
                                color=colores[idx], linewidth=2)
                        curvas_y.append(y_vals)
            elif len(constantes) >= 2:
                for idx, c in enumerate(valores_c):
                    y_vals = evaluar(expresion.subs({constantes[0]: c, constantes[1]: 1}))
                    if y_vals is not None:
                        ax.plot(x_vals, y_vals,
                                label=f"{constantes[0]}={c}, {constantes[1]}=1",
                                color=colores[idx], linewidth=2)
                        curvas_y.append(y_vals)
            else:
                y_vals = evaluar(expresion)
                if y_vals is not None:
                    ax.plot(x_vals, y_vals, color="#00ffff", linewidth=2)
                    curvas_y.append(y_vals)

            ax.axhline(0, color="#ffffff", linewidth=1)
            ax.axvline(0, color="#ffffff", linewidth=1)
            ax.grid(True, linestyle=":", alpha=0.3, color="#00ffff")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("Familia de soluciones", color="white")

            if curvas_y:
                todo = np.concatenate([c[np.isfinite(c)] for c in curvas_y if c is not None])
                todo = todo[np.isfinite(todo)]
                if todo.size:
                    ymin, ymax = np.percentile(todo, [2, 98])
                    margen = max((ymax - ymin) * 0.15, 1)
                    ax.set_ylim(ymin - margen, ymax + margen)

            if constantes:
                ax.legend(facecolor="#2b2b2b", edgecolor=COLOR_ACENTO_2, loc="upper right", fontsize=8)

            self.ultima_figura_grafica = fig
            canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        except Exception:
            lbl = ctk.CTkLabel(self.graph_frame, text="No fue posible graficar esta solución.")
            lbl.pack(pady=20)


    def _construir_funcion_numerica(self, edo, orden):
        deriv_alta = sp.Derivative(self.y, self.x, orden) if orden > 1 else sp.Derivative(self.y, self.x)
        despejes = sp.solve(sp.Eq(edo.lhs - edo.rhs, 0), deriv_alta)
        if not despejes:
            raise ValueError("No se pudo despejar la derivada de mayor orden.")
        expr_deriv_alta = despejes[0]

        if orden == 1:
            simbolos_estado = [self.y]
        else:
            simbolos_estado = [self.y] + [sp.Derivative(self.y, self.x, k) for k in range(1, orden)]

        f_lamb = sp.lambdify((self.x, simbolos_estado), expr_deriv_alta, modules=["numpy"])

        def sistema(x_val, estado):
            derivs = list(estado[1:]) if orden > 1 else []
            try:
                nueva_deriv_alta = f_lamb(x_val, estado)
            except Exception:
                nueva_deriv_alta = np.nan
            return derivs + [nueva_deriv_alta]

        return sistema

    def build_tab_campo(self):
        tab = self.tab_campo
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(3, weight=1)

        ctk.CTkLabel(
            tab, text="Campo de pendientes e isoclinas",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, padx=10, pady=(10, 2), sticky="w")

        marco = ctk.CTkFrame(tab, fg_color="transparent")
        marco.grid(row=1, column=0, sticky="ew", padx=10)
        marco.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(marco, text="dy/dx =").grid(row=0, column=0, sticky="e", padx=5, pady=3)
        self.entry_campo_f = ctk.CTkEntry(marco)
        self.entry_campo_f.insert(0, "x**2 - y**2")
        self.entry_campo_f.grid(row=0, column=1, sticky="ew", padx=5)

        ctk.CTkLabel(marco, text="Puntos (x,y; x,y; ...):").grid(row=1, column=0, sticky="e", padx=5, pady=3)
        self.entry_campo_puntos = ctk.CTkEntry(marco)
        self.entry_campo_puntos.insert(0, "0,1")
        self.entry_campo_puntos.grid(row=1, column=1, sticky="ew", padx=5)

        ctk.CTkLabel(marco, text="x mín, x máx, y mín, y máx:").grid(row=2, column=0, sticky="e", padx=5, pady=3)
        self.entry_campo_ventana = ctk.CTkEntry(marco)
        self.entry_campo_ventana.insert(0, "-3, 3, -3, 3")
        self.entry_campo_ventana.grid(row=2, column=1, sticky="ew", padx=5)

        ctk.CTkLabel(marco, text="Isoclinas (pendientes c):").grid(row=3, column=0, sticky="e", padx=5, pady=3)
        self.entry_campo_iso = ctk.CTkEntry(marco)
        self.entry_campo_iso.insert(0, "-2, -1, 0, 1, 2")
        self.entry_campo_iso.grid(row=3, column=1, sticky="ew", padx=5)

        self.var_iso = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(marco, text="Dibujar isoclinas", variable=self.var_iso).grid(row=4, column=1, sticky="w", padx=5, pady=3)

        botones = ctk.CTkFrame(tab, fg_color="transparent")
        botones.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        ctk.CTkButton(botones, text="DIBUJAR CAMPO Y CURVAS", command=self.dibujar_campo,
                      fg_color="#1f538d").pack(side="left", padx=5)
        ctk.CTkButton(botones, text="Usar ecuación del panel lateral", command=self.campo_desde_panel,
                      fg_color="#4b3b7a").pack(side="left", padx=5)

        self.graph_frame_campo = ctk.CTkFrame(tab, corner_radius=10)
        self.graph_frame_campo.grid(row=3, column=0, sticky="nsew", padx=10, pady=5)

        self.text_campo = ctk.CTkTextbox(tab, height=140, wrap="word",
                                         fg_color=COLOR_FONDO_PANEL, font=ctk.CTkFont(size=13))
        self.text_campo.grid(row=4, column=0, sticky="ew", padx=10, pady=(5, 10))

    def campo_desde_panel(self):
        txt = self.entry_eq.get().strip()
        if "=" in txt:
            txt = txt.split("=", 1)[1].strip()
        self.entry_campo_f.delete(0, "end")
        self.entry_campo_f.insert(0, txt)

    def dibujar_campo(self):
        for w in self.graph_frame_campo.winfo_children():
            w.destroy()
        self.text_campo.delete("1.0", "end")
        try:
            texto_f = self.entry_campo_f.get()
            ventana = [float(sp.sympify(p)) for p in self.entry_campo_ventana.get().split(",")]
            xmin, xmax, ymin, ymax = ventana
            puntos = []
            for par in self.entry_campo_puntos.get().split(";"):
                par = par.strip()
                if par:
                    a, b = par.split(",")
                    puntos.append((float(sp.sympify(a)), float(sp.sympify(b))))
            pendientes = []
            if self.var_iso.get():
                pendientes = [float(sp.sympify(c)) for c in self.entry_campo_iso.get().split(",") if c.strip()]

            expr_prev, ctx = parsear_rhs(texto_f)
            libres = {str(s) for s in expr_prev.free_symbols}
            var_x, var_y = "x", "y"
            if "t" in libres and "x" not in libres:
                var_x = "t"
            if "P" in libres:
                var_y = "P"
            if "v" in libres:
                var_y = "v"

            f, curvas, equilibrios, expr = analizar_campo(
                texto_f, puntos, (xmin, xmax), (ymin, ymax), var_x, var_y)

            plt.style.use("dark_background")
            fig, ax = plt.subplots(figsize=(8, 5), facecolor="#2b2b2b")
            ax.set_facecolor(COLOR_FONDO_PANEL)

            gx, gy = np.meshgrid(np.linspace(xmin, xmax, 25), np.linspace(ymin, ymax, 25))
            pend = f(gx, gy)
            norma = np.sqrt(1 + pend ** 2)
            ax.quiver(gx, gy, 1 / norma, pend / norma, color="#7fbfff", pivot="mid",
                      angles="xy", scale=45, width=0.003, headwidth=0, headlength=0, headaxislength=0)

            if pendientes:
                fx, fy = np.meshgrid(np.linspace(xmin, xmax, 300), np.linspace(ymin, ymax, 300))
                fz = f(fx, fy)
                cs = ax.contour(fx, fy, fz, levels=sorted(pendientes), colors="#ffcc00",
                                linewidths=0.9, linestyles="--")
                ax.clabel(cs, fmt="c=%g", fontsize=7)

            paleta = ["#00ffcc", "#ff00ff", "#00ff00", "#ff3333", "#ffff00", "#ff9900", "#00aaff"]
            for i, ((px, py), tramos) in enumerate(curvas):
                color = paleta[i % len(paleta)]
                for (tt, yy) in tramos:
                    if tt.size:
                        ax.plot(tt, yy, color=color, linewidth=2.2)
                ax.scatter([px], [py], color=color, zorder=5, s=35, edgecolor="white")

            for eq in equilibrios:
                if ymin <= eq <= ymax:
                    ax.axhline(eq, color="#ff5555", linestyle=":", linewidth=1.3)

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.axhline(0, color="white", linewidth=0.6)
            ax.axvline(0, color="white", linewidth=0.6)
            ax.set_xlabel(var_x)
            ax.set_ylabel(var_y)
            ax.set_title(f"Campo de pendientes:  d{var_y}/d{var_x} = {expr}", color="white", fontsize=10)
            ax.grid(True, linestyle=":", alpha=0.2)

            canvas = FigureCanvasTkAgg(fig, master=self.graph_frame_campo)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

            self.text_campo.insert("end", self.interpretar_campo(expr, curvas, equilibrios, var_x, var_y, (xmin, xmax)))
        except Exception as e:
            self.text_campo.insert("end", f"[!] No se pudo dibujar el campo: {e}\n{traceback.format_exc(limit=2)}")

    def interpretar_campo(self, expr, curvas, equilibrios, var_x, var_y, xlim):
        txt = "LECTURA DEL CAMPO (explicada fácil):\n"
        txt += "   > Cada rayita azul muestra hacia dónde va la solución en ese punto: es la pendiente que dicta la ecuación.\n"
        txt += "   > Las líneas amarillas punteadas son isoclinas: puntos donde la pendiente es siempre la misma (c).\n"
        if equilibrios:
            txt += f"   > Soluciones de equilibrio (rayas rojas): {var_y} = {', '.join(f'{e:g}' for e in equilibrios)}. Ahí la pendiente es 0 y la solución no cambia.\n"
            txt += "     Si las rayitas cercanas apuntan hacia esa línea es estable (atrae); si se alejan, es inestable (repele).\n"
        for (px, py), tramos in curvas:
            tt, yy = tramos[0]
            if tt.size:
                fin = yy[-1]
                if not np.isfinite(fin) or abs(fin) > 1e6 or tt[-1] < xlim[1] - 1e-6:
                    txt += f"   > Curva desde ({px:g}, {py:g}): se dispara o deja de existir antes de x={xlim[1]:g} (explosión en tiempo finito o dominio restringido).\n"
                else:
                    txt += f"   > Curva desde ({px:g}, {py:g}): llega a {var_y} ≈ {fin:.4g} en {var_x}={tt[-1]:g}.\n"
        return txt


    def solve_pvi_numeric(self):
        for widget in self.graph_frame_pvi.winfo_children():
            widget.destroy()

        if not SCIPY_DISPONIBLE:
            messagebox.showerror("SciPy no disponible", "Instala scipy para usar esta pestaña.")
            return

        eq_str = self.entry_eq.get().strip()
        if not eq_str:
            messagebox.showinfo("Falta ecuación", "Escribe una ecuación en el panel lateral primero.")
            return

        try:
            edo = self.parse_user_input(eq_str)
            orden = self.detectar_orden(edo)
            if orden is None or orden > 2:
                messagebox.showinfo(
                    "Orden no soportado",
                    "El solver numérico integrado admite ecuaciones de orden 1 o 2. "
                    "Para órdenes mayores necesitarías extender el sistema de estado.",
                )
                return

            x0 = float(sp.sympify(self.entry_x0.get()))
            y0 = float(sp.sympify(self.entry_y0.get()))
            rango_txt = self.entry_rango_pvi.get()
            partes = [float(sp.sympify(p)) for p in rango_txt.split(",")]
            x_ini, x_fin = partes[0], partes[1]

            if orden == 1:
                estado0 = [y0]
            else:
                dy0 = float(sp.sympify(self.entry_dy0.get()))
                estado0 = [y0, dy0]

            sistema = self._construir_funcion_numerica(edo, orden)

            x_eval = np.linspace(x_ini, x_fin, 400)
            sol = solve_ivp(sistema, (x_ini, x_fin), estado0, t_eval=x_eval, method="RK45", dense_output=True)

            if not sol.success:
                messagebox.showerror("No se pudo integrar", sol.message)
                return

            plt.style.use("dark_background")
            fig, ax = plt.subplots(figsize=(8, 4.2), facecolor="#2b2b2b")
            ax.set_facecolor(COLOR_FONDO_PANEL)
            ax.plot(sol.t, sol.y[0], color=COLOR_ACENTO, linewidth=2, label="y(x) numérica (RK45)")
            ax.scatter([x0], [y0], color=COLOR_ADVERTENCIA, zorder=5, label=f"Condición inicial (x0={x0}, y0={y0})")
            ax.axhline(0, color="#ffffff", linewidth=1)
            ax.axvline(0, color="#ffffff", linewidth=1)
            ax.grid(True, linestyle=":", alpha=0.3, color=COLOR_ACENTO_2)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("Solución numérica del PVI (Runge-Kutta 45)", color="white")
            ax.legend(facecolor="#2b2b2b", edgecolor=COLOR_ACENTO_2, fontsize=8)

            canvas = FigureCanvasTkAgg(fig, master=self.graph_frame_pvi)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

        except Exception as e:
            messagebox.showerror("Error en el PVI numérico", f"{e}\n\n{traceback.format_exc(limit=2)}")


if __name__ == "__main__":
    app = UniversalEDOSolver()
    app.mainloop()
