# ============================================================
# app.py (coeur SymPy)
# Fichier destiné à être importé par Flask (routes séparées).
# Toutes les sorties passent par st.latex()/st.info()/st.error().
# ============================================================

from flask import Flask, request, jsonify, render_template
from sympy import S
import sympy as sp
from sympy import (
    sin, cos, tan,
    asin, acos, atan,
    sinh, cosh, tanh,
    asinh, acosh, atanh,
    exp, log, ln, sqrt,
    pi, E, Abs, sign,
    erf, gamma, zeta,
    floor, ceiling
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sympy.calculus.util import continuous_domain
from sympy import S
from sympy.assumptions import Q, ask
from sympy.core.relational import Relational, Equality
import io
import base64

# ------------------------------------------------------------
# Symboles globaux
# ------------------------------------------------------------
k = sp.symbols("k", integer=True, positive=True)


# ============================================================
# Faux Streamlit : st.latex(), st.info(), st.error(), st.pyplot()
# ============================================================
class _StreamlitCompat:
    def __init__(self):
        self._out = []

    def set_out(self, out):
        self._out = out

    def latex(self, msg):
        self._out.append({"type": "latex", "content": msg})

    def info(self, msg):
        self._out.append({"type": "info", "content": str(msg)})

    def error(self, msg):
        self._out.append({"type": "error", "content": str(msg)})

    def pyplot(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        self._out.append({"type": "plot", "content": b64})


st = _StreamlitCompat()




def facteur(expression):# 
    #retourne le facteur multiplicatif rationnel de l'expression et retourne 1 s'il ny'en a pas de visible ou si le facteur est irrationnel
    # Pour les rationnels
    facteur , reste =  expression.as_coeff_Mul()# Je sépare le coefficient rationnel et le reste de   l'expression (qui est une fonction de n)
    return facteur

def facteur2(expression) : 
    expression_approx = expression.evalf()# On evalue expression en fonction de n, toutes les constantes irrationnelles y sont évaluées( n lui n'est pas evalué)
    facteur_approx = facteur(expression_approx)
    quotient = sp.simplify(expression_approx/expression)# Ce quotient est purement réel mathématiquement (expression.approx est au numérateur pour eviter les inversions des approximations) 
    # On remarquera que quotient = facteur_approx/facteur aussi donc
    facteur = sp.simplify(facteur_approx / quotient)  # on obtient 1.0*facteur
    return facteur 





# ============================================================
# Fonctions bornées connues
# ============================================================
BORNES_CONNUES = {
    sp.sin: (-1, 1),
    sp.cos: (-1, 1),
    sp.sign: (-1, 1),
    sp.Abs: (0, None),
}


# ============================================================
# Outils de signe / encadrements
# ============================================================
def signe_asymptotique(expr, n):
    # 1) direct
    if expr.is_positive:
        return 1
    if expr.is_negative:
        return -1

    # 2) limite
    try:
        L = sp.limit(expr, n, sp.oo)
        if L.is_positive:
            return 1
        if L.is_negative:
            return -1
    except:
        pass

    # 3) terme dominant
    try:
        lead = sp.simplify(expr.as_leading_term(n))
        if lead.is_positive:
            return 1
        if lead.is_negative:
            return -1
    except:
        pass

    return 0


def min_borne(a, b):
    d = sp.simplify(a - b)
    if d.is_nonpositive:
        return a
    if d.is_nonnegative:
        return b
    return a


def max_borne(a, b):
    d = sp.simplify(a - b)
    if d.is_nonnegative:
        return a
    if d.is_nonpositive:
        return b
    return b


def borne_constante_variable(expr, n):
    if expr.is_Number:
        return expr, expr
    if expr == n:
        return expr, expr
    # (-1)**(quelquechose en n) : borné dans [-1,1]
    if expr.is_Pow and expr.base == -1 and expr.exp.has(n):
        return sp.Integer(-1), sp.Integer(1)
    return None


def borne_fonction(expr):
    if expr.is_Function and expr.func in BORNES_CONNUES:
        bmin, bmax = BORNES_CONNUES[expr.func]
        if bmin is not None and bmax is not None:
            return sp.sympify(bmin), sp.sympify(bmax)
        return expr, expr
    return None


def encadrements(expr, n):
    # constantes / n / (-1)^n
    r = borne_constante_variable(expr, n)
    if r is not None:
        return r

    # sin/cos/...
    r = borne_fonction(expr)
    if r is not None:
        return r

    # somme
    if expr.is_Add:
        mi, ma = 0, 0
        for t in expr.args:
            tmi, tma = encadrements(t, n)
            mi += tmi
            ma += tma
        return sp.simplify(mi), sp.simplify(ma)

    # produit
    if expr.is_Mul:
        mi, ma = 1, 1
        for f in expr.args:
            fmi, fma = encadrements(f, n)
            c1 = min_borne(mi * fmi, mi * fma)
            c2 = min_borne(ma * fmi, ma * fma)
            mi = min_borne(c1, c2)
            c3 = max_borne(mi * fmi, mi * fma)
            c4 = max_borne(ma * fmi, ma * fma)
            ma = max_borne(c3, c4)
        return sp.simplify(mi), sp.simplify(ma)

    # puissance : uniquement cas sûrs (entiers, -1)
    if expr.is_Pow:
        base, exp = expr.as_base_exp()
        bmin, bmax = encadrements(base, n)

        if exp == -1:
            # inversion seulement si signe contrôlé
            if bmin.is_positive:
                return sp.simplify(1 / bmax), sp.simplify(1 / bmin)
            if bmax.is_negative:
                return sp.simplify(1 / bmin), sp.simplify(1 / bmax)
            return expr, expr

        if exp.is_Integer and exp % 2 == 0:
            if bmin.is_nonnegative:
                return sp.simplify(bmin**exp), sp.simplify(bmax**exp)
            if bmax.is_nonpositive:
                return sp.simplify(bmax**exp), sp.simplify(bmin**exp)
            return sp.Integer(0), sp.simplify(max(abs(bmin), abs(bmax)) ** exp)

        if exp.is_Integer and exp % 2 == 1:
            return sp.simplify(bmin**exp), sp.simplify(bmax**exp)

        return expr, expr

    # fraction : seulement si signe dénominateur contrôlé à l'infini
    try:
        num, den = expr.as_numer_denom()
        if den != 1:
            nmin, nmax = encadrements(num, n)
            dmin, dmax = encadrements(den, n)
            s = signe_asymptotique(den, n)
            if s == 1:
                return (
                    sp.simplify(min_borne(nmin / dmax, nmax / dmax)),
                    sp.simplify(max_borne(nmin / dmin, nmax / dmin)),
                )
            if s == -1:
                return (
                    sp.simplify(min_borne(nmax / dmax, nmin / dmax)),
                    sp.simplify(max_borne(nmax / dmin, nmin / dmin)),
                )
    except:
        pass

    return expr, expr


# ============================================================
# Graphe (optionnel)
# ============================================================
def graphe(u_n, n, c):
    try:
        domaine = continuous_domain(u_n, n, S.Reals)
    except:
        domaine = S.Reals

    x_, y_ = [], []

    if isinstance(u_n, sp.Sum):
        expr = u_n.function
        (kk, n_min, _) = u_n.limits[0]
        f = sp.lambdify(kk, expr, "numpy")

        for N in range(0, 400):
            if domaine.contains(N):
                if N < int(n_min):
                    continue
                try:
                    s = 0.0
                    for j in range(int(n_min), N + 1):
                        s += float(f(j))
                    # On ajoute x et y ensemble uniquement si tout a réussi
                    x_.append(N)
                    y_.append(s)
                except:
                    pass
    else:
        f = sp.lambdify(n, u_n, "numpy")

        for N in range(0, 1500):
            if domaine.contains(N):
                try:
                    # IMPORTANT : on calcule d'abord y ; si ça échoue, on n'ajoute rien
                    y_val = float(f(N))
                    x_.append(N)
                    y_.append(y_val)
                except:
                    pass

    # Si rien n'est traçable, on évite de faire planter matplotlib
    if len(x_) == 0 or len(y_) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x_, y_, color=c, label=r"$" + sp.latex(u_n) + r"$")
    ax.set_xlabel("n")
    ax.set_ylabel("$" + sp.latex(u_n) + "$")
    ax.legend()
    st.pyplot(fig)




def diverge(u_n, n):
    try:
        lim = sp.limit_seq(u_n, n)

        if lim is not None and lim != 0:
            st.latex(
                r"\lim_{n \to \infty} " + sp.latex(u_n) + " = " + sp.latex(lim)
            )
            st.latex(
                r"\text{Le terme général ne tend pas vers } 0,\text{ la série est donc divergente.}"
            )
            return True

        return False

    except:
        return False






# ============================================================
# Positivité + décroissance (Leibniz)
# ============================================================
def positivite(u_n, n, n_min):
    """
    Vérifie (quand possible) u_n >= 0 pour n >= n_min, et affiche une justification.
    """
    try:
        x = sp.symbols("x", real=True)
        domaine = S.Reals.intersect(sp.Interval(n_min, sp.oo))
        ineq = sp.solve_univariate_inequality(
            sp.simplify(u_n.subs(n, x)) >= 0,
            x,
            relational=False,
            domain=domaine,
        )
        ok = (u_n.is_nonnegative is True) or (ineq != sp.EmptySet)
        if ok:
            msg = (
                r"\text{On vérifie que } "
                + sp.latex(u_n)
                + r"\ge 0 \text{ pour } n\ge "
                + sp.latex(n_min)
                + r"."
            )
            return True, msg
        return None, None
    except:
        if u_n.is_nonnegative is True:
            msg = r"\text{On a } " + sp.latex(u_n) + r"\ge 0."
            return True, msg
        return None, None


def decroissance(a_n, n, n_min):
    try:
        # Définition de la suite
        st.latex(
            r"\text{On considère la suite } (a_n)_{n\in\mathbb{N}} "
            r"\text{ définie par } a_n = " + sp.latex(a_n) + r"."
        )

        a_n1 = sp.simplify(a_n.subs(n, n + 1))

        # ============================================================
        # 1) Étude directe de a_{n+1} - a_n
        # ============================================================
        try:
            delta = sp.simplify(a_n1 - a_n)
            if delta.is_nonpositive:
                st.latex(
                    r"\text{Pour tout } n \ge " + sp.latex(n_min) + r",\quad "
                    r"a_{n+1}-a_n = "
                    + sp.latex(a_n1) + r" - " + sp.latex(a_n)
                    + r" = " + sp.latex(delta) + r"\le 0."
                )
                st.latex(
                    r"\text{Ainsi, la suite } (a_n) \text{ est décroissante à partir de } "
                    + sp.latex(n_min) + r"."
                )
                return True, None
        except:
            pass

        # ============================================================
        # 2) Étude par le quotient a_{n+1}/a_n
        # ============================================================
        try:
            q = sp.simplify(a_n1 / a_n)

            # On impose la positivité pour que la comparaison ait un sens
            pos_a_n, _ = positivite(a_n, n, n_min)
            pos_a_n1, _ = positivite(a_n1, n, n_min)

            if pos_a_n and pos_a_n1:
                test_q = sp.simplify(q - 1)
                if test_q.is_nonpositive:
                    st.latex(
                        r"\text{Pour tout } n \ge " + sp.latex(n_min) + r",\quad "
                        r"\frac{a_{n+1}}{a_n} = " + sp.latex(q) + r"\le 1."
                    )
                    st.latex(
                        r"\text{Comme } a_n \ge 0,\text{ la suite } (a_n) "
                        r"\text{ est décroissante à partir de } "
                        + sp.latex(n_min) + r"."
                    )
                    return True, None
        except:
            pass
        
        # ============================================================
        # 3) Étude par la fonction associée et sa dérivée
        # ============================================================
        try:
            x = sp.symbols('x', real=True)

            f = a_n.subs(n, x)

            st.latex(
                r"\text{On associe à la suite la fonction } "
                r"f : \mathbb{R} \to \mathbb{R} \text{ telle que } "
                r"f(n)=a_n \text{ pour tout entier } n."
            )
            st.latex(
                r"\text{On a donc, pour tout réel } x,\quad f(x) = "
                + sp.latex(f) + r"."
            )

            # Domaine de dérivabilité, restreint à x ≥ n_min
            domaine = continuous_domain(f, x, S.Reals)
            domaine = domaine.intersect(sp.Interval(n_min, sp.oo))

            st.latex(
                r"\text{La fonction } f \text{ est dérivable sur } "
                + sp.latex(domaine) + r"."
            )

            f_prime = sp.diff(f, x)
            st.latex(
                r"\text{Sa dérivée est donnée par } f'(x) = "
                + sp.latex(f_prime) + r"."
            )

            # --------------------------------------------------------
            # Test direct du signe de la dérivée
            # --------------------------------------------------------
            if f_prime.is_negative:
                st.latex(
                    r"\text{On a } f'(x) < 0 \text{ pour tout } x \in "
                    + sp.latex(domaine) + r"."
                )
                st.latex(
                    r"\text{La fonction } f \text{ est strictement décroissante sur ce domaine.}"
                )
                st.latex(
                    r"\text{Par conséquent, la suite } (a_n) \text{ est décroissante pour tout } n \ge "
                    + sp.latex(n_min) + r"."
                )
                return True, None

            if f_prime.is_nonpositive:
                st.latex(
                    r"\text{On a } f'(x) \le 0 \text{ pour tout } x \in "
                    + sp.latex(domaine) + r"."
                )
                st.latex(
                    r"\text{La fonction } f \text{ est décroissante sur ce domaine.}"
                )
                st.latex(
                    r"\text{Par conséquent, la suite } (a_n) \text{ est décroissante pour tout } n \ge "
                    + sp.latex(n_min) + r"."
                )
                return True, None

            # --------------------------------------------------------
            # 4) Étude complète du signe de f' par inéquation
            # --------------------------------------------------------
            try:
                domaine_signe = sp.solve_univariate_inequality(
                    f_prime <= 0, x, relational=False
                )

                domaine_utile = domaine.intersect(domaine_signe)

                if domaine_utile.sup == sp.oo:
                    st.latex(
                        r"\text{On étudie le signe de } f'(x) = "
                        + sp.latex(f_prime) + r"."
                    )
                    st.latex(
                        r"\text{On obtient } f'(x) \le 0 \text{ pour } x \in "
                        + sp.latex(domaine_utile) + r"."
                    )
                    st.latex(
                        r"\text{La fonction } f \text{ est donc décroissante sur cet intervalle.}"
                    )
                    st.latex(
                        r"\text{Il existe alors un entier } N \ge "
                        + sp.latex(n_min) + r" \text{ tel que pour tout } n \ge N,\ a_{n+1} \le a_n."
                    )
                    st.latex(
                        r"\text{Ainsi, la suite } (a_n) \text{ est décroissante à partir d'un certain rang.}"
                    )
                    return True, None
            except:
                pass

        except:
            pass

    except Exception as e:
        return None, f"Détail technique (décroissance) : {e}"

    return False, None


# ============================================================
# Alternance / extraction partie positive
# ============================================================
def detecte_alternance(expr, n):
    for p in expr.atoms(sp.Pow):
        base, exp = p.args
        if base == -1 and exp.has(n):
            return True
    return False


def extraire_partie_positive(expr, n, n_min):
    """
    Si expr = (-1)^n * q(n), renvoie q(n) si q(n) >= 0 (ou -q si q<=0).
    """
    for p in expr.atoms(sp.Pow):
        base, exp = p.args
        if base == -1 and exp.has(n):
            q = sp.simplify(expr / p)
            if q.is_nonnegative or (positivite(q, n, n_min)[0] is True):
                return sp.simplify(q)
            if q.is_nonpositive:
                return sp.simplify(-q)
            return None
    return None


# ============================================================
# DL : on n'affiche que l'équivalence (~) et la mention "d'après un DL"
# ============================================================
def dl_equivalent(u_n, n):
    try:
        x = sp.symbols("x")
        u_x = sp.simplify(u_n.subs(n, 1 / x))
        ser = sp.series(u_x, x, 0, 6)
        eq = sp.simplify(ser.as_leading_term(x).subs(x, 1 / n))

        if sp.simplify(eq - u_n).equals(0):
            return None

        st.latex(
            r"\text{On obtient } "
            + sp.latex(u_n)
            + r"\sim "
            + sp.latex(eq)
            + r"\ \text{(d'après un développement limité en } \frac{1}{n}\text{).}"
        )
        return eq
    except:
        return None


# ============================================================
# Critère de Riemann (1/n^p)
# ============================================================
def prepare_Riemann(expression, n):
    expression_approx = expression.evalf()
    facteur_val = facteur(expression_approx)
    if facteur_val == 0:
        facteur_val = 1
    expression_approx = expression_approx / facteur_val
    
    numerateur = expression_approx.as_numer_denom()[0]
    try:
        num = int(numerateur)
    except (TypeError, ValueError):
        return None
    
    dict = expression.as_numer_denom()[1].as_powers_dict()
    return num, dict

def test_Riemann(u_n, n, n_min):
    S_n = sp.Sum(u_n.subs(n, k), (k, n_min, n))

    P = prepare_Riemann(u_n, n)
    if P is None:
        return None

    num, d = P
    if num == 1 and len(d) == 1 and list(d.keys())[0] == n:
        exposant = d[n]
        
        if exposant > 1:
            st.latex(
                r"\text{Comme } "
                + sp.latex(exposant)
                + r">1,\ \sum \frac{1}{n^{"
                + sp.latex(exposant)
                + r"}} \text{ converge par le critère de Riemann. Donc } "
                + sp.latex(S_n)
                + r"\text{ converge.}"
            )
            return True
        else:
            st.latex(
                r"\text{Comme } "
                + sp.latex(exposant)
                + r"\le 1,\ \sum \frac{1}{n^{"
                + sp.latex(exposant)
                + r"}} \text{ diverge (critère de Riemann). Donc } "
                + sp.latex(S_n)
                + r"\text{ diverge.}"
            )
            return False

    return None


# ============================================================
# Bertrand (1/(n (log n)^a))
# ============================================================
def facteur(expr):
    c, _ = expr.as_coeff_Mul()
    return c


def test_bertrand(u_n, n, n_min):
    S_n = sp.Sum(u_n.subs(n, k), (k, n_min, n))

    u_s = sp.simplify(u_n / facteur(u_n))
    num, den = u_s.as_numer_denom()

    # On ne traite que num = 1
    try:
        if int(num) != 1:
            return None
    except:
        return None

    facs = list(den.args) if den.is_Mul else [den]
    if n not in facs:
        return None

    others = [f for f in facs if f != n]
    exposant_log = S.Zero

    for f in others:
        if f == sp.log(n):
            exposant_log += 1
            continue
        if f.is_Pow:
            base, exp = f.as_base_exp()
            if base == sp.log(n):
                exposant_log += exp
                continue
        return None

    st.latex(
        r"\text{On reconnaît } "
        + sp.latex(u_n)
        + r"=\frac{1}{n(\log n)^{"
        + sp.latex(exposant_log)
        + r"}}."
    )

    if exposant_log > 1:
        st.latex(
            r"\text{Comme } "
            + sp.latex(exposant_log)
            + r">1,\ \sum \frac{1}{n(\log n)^{"
            + sp.latex(exposant_log)
            + r"}} \text{ converge (critère de Bertrand). Donc } "
            + sp.latex(S_n)
            + r"\text{ converge.}"
        )
        return True

    st.latex(
        r"\text{Comme } "
        + sp.latex(exposant_log)
        + r"\le 1,\ \sum \frac{1}{n(\log n)^{"
        + sp.latex(exposant_log)
        + r"}} \text{ diverge (critère de Bertrand). Donc } "
        + sp.latex(S_n)
        + r"\text{ diverge.}"
    )
    return False


# ============================================================
# Critère des logs itérés (famille n^p * log * loglog * ...)
# ============================================================
def _is_iterated_log_of_n(expr, n, max_depth=8):
    """
    True si expr = log(log(...log(n)...)) (au moins 1 log).
    """
    depth = 0
    t = expr
    while depth < max_depth:
        if t == n:
            return depth >= 1
        if t.func != sp.log or len(t.args) != 1:
            return False
        t = t.args[0]
        depth += 1
    return False


def parse_polylog_term(u_n, n):
    """
    Reconnaît :  1 / ( n^p * Π (log^{∘i}(n))^{beta_i} )
    Renvoie : (p, [(log_i, beta_i) trié par profondeur])
    """
    u_s = sp.simplify(u_n)
    num, den = u_s.as_numer_denom()
    if num != 1:
        return None

    den_factors = list(den.args) if den.is_Mul else [den]

    exposant_n = S.Zero
    logs = []  # (depth, base, expo)

    for f in den_factors:
        # n^p
        if f == n:
            exposant_n += 1
            continue
        if f.is_Pow and f.base == n:
            exposant_n += f.exp
            continue

        # log^{...}
        base, expo = None, None
        if f == sp.log(n):
            base, expo = sp.log(n), 1
        elif f.is_Pow:
            b, e = f.as_base_exp()
            if _is_iterated_log_of_n(b, n):
                base, expo = b, e

        if base is None:
            return None

        # profondeur
        depth = 0
        t = base
        while t != n:
            t = t.args[0]
            depth += 1
        logs.append((depth, base, expo))

    logs.sort(key=lambda x: x[0])
    return exposant_n, [(b, e) for _, b, e in logs]


def test_polylog(u_n, n, n_min):
    """
    Décision standard :
    - si p>1 : converge
    - si p<1 : diverge
    - si p=1 : on regarde les exposants des logs itérés :
        premier exposant != 1 : >1 => converge, <=1 => diverge
        tous = 1 => diverge
    """
    S_n = sp.Sum(u_n.subs(n, k), (k, n_min, n))
    parsed = parse_polylog_term(u_n, n)
    if parsed is None:
        return None

    exposant_n, logs = parsed

    # On annonce la forme (sans lettre non définie : on affiche l'exposant réel rencontré)
    st.latex(
        r"\text{On met } "
        + sp.latex(u_n)
        + r"\text{ sous la forme } \frac{1}{n^{"
        + sp.latex(exposant_n)
        + r"}}\cdot\frac{1}{\prod (\log^{\circ i}(n))^{\beta_i}}\text{ (avec les exposants affichés ci-dessous).}"
    )

    # p>1 / p<1 / p=1
    if exposant_n > 1:
        st.latex(
            r"\text{Comme } "
            + sp.latex(exposant_n)
            + r">1,\ \sum \frac{1}{n^{"
            + sp.latex(exposant_n)
            + r"}} \text{ converge (Riemann), et } "
            + sp.latex(u_n)
            + r"\le \frac{1}{n^{"
            + sp.latex(exposant_n)
            + r"}}\text{ pour } n\ge 2.\ \text{Donc } "
            + sp.latex(S_n)
            + r"\text{ converge (comparaison).}"
        )
        return True

    if exposant_n < 1:
        st.latex(
            r"\text{Comme } "
            + sp.latex(exposant_n)
            + r"<1,\ \sum \frac{1}{n^{"
            + sp.latex(exposant_n)
            + r"}} \text{ diverge (Riemann), et } "
            + sp.latex(u_n)
            + r"\ge \frac{1}{n^{"
            + sp.latex(exposant_n)
            + r"}}\text{ pour } n\ge 2.\ \text{Donc } "
            + sp.latex(S_n)
            + r"\text{ diverge (comparaison).}"
        )
        return False

    # cas critique p=1
    if logs == []:
        st.latex(r"\text{Ici } " + sp.latex(u_n) + r"=\frac{1}{n},\ \sum \frac{1}{n} \text{ diverge. Donc } " + sp.latex(S_n) + r"\text{ diverge.}")
        return False

    # on parcourt les logs dans l'ordre
    produit_critique = []
    for base, expo in logs:
        produit_critique.append(base)

        st.latex(
            r"\text{On considère l'exposant de } "
            + sp.latex(base)
            + r"\text{ : } \beta="
            + sp.latex(expo)
            + r"."
        )

        if sp.simplify(expo - 1).equals(0):
            continue

        if expo > 1:
            st.latex(
                r"\text{Au premier rang où } \beta\neq 1,\ \beta="
                + sp.latex(expo)
                + r">1,\ \text{donc } "
                + sp.latex(S_n)
                + r"\text{ converge (critère des logs itérés).}"
            )
            return True
        else:
            st.latex(
                r"\text{Au premier rang où } \beta\neq 1,\ \beta="
                + sp.latex(expo)
                + r"\le 1,\ \text{donc } "
                + sp.latex(S_n)
                + r"\text{ diverge (critère des logs itérés).}"
            )
            return False

    st.latex(
        r"\text{Tous les exposants rencontrés valent } 1,\ \text{on est dans un cas critique de type } "
        r"\sum \frac{1}{n\log n\log\log n\cdots} \text{ qui diverge. Donc } "
        + sp.latex(S_n)
        + r"\text{ diverge.}"
    )
    return False


# ============================================================
# d'Alembert (ratio)
# ============================================================
def test_Alembert(u_n, n, n_min):
    S_n = sp.Sum(u_n.subs(n, k), (k, n_min, n))
    u_s = sp.simplify(u_n)

    if u_s == 0:
        return None

    try:
        u_n1 = sp.simplify(u_s.subs(n, n + 1))
        q = sp.simplify(sp.Abs(u_n1 / u_s))
        L = sp.limit_seq(q, n)
        if L is None or L.is_real is False:
            return None

        if L < 1:
            st.latex(
                r"\text{On calcule } \left|\frac{u_{n+1}}{u_n}\right|="
                + sp.latex(q)
                + r"\ \text{et } \lim_{n\to\infty}\left|\frac{u_{n+1}}{u_n}\right|="
                + sp.latex(L)
                + r"<1."
            )
            st.latex(r"\text{Donc } " + sp.latex(S_n) + r"\text{ converge (critère d'Alembert).}")
            return True

        if L > 1:
            st.latex(
                r"\text{On calcule } \left|\frac{u_{n+1}}{u_n}\right|="
                + sp.latex(q)
                + r"\ \text{et } \lim_{n\to\infty}\left|\frac{u_{n+1}}{u_n}\right|="
                + sp.latex(L)
                + r">1."
            )
            st.latex(r"\text{Donc } " + sp.latex(S_n) + r"\text{ diverge (critère d'Alembert).}")
            return False

    except:
        pass

    return None


# ============================================================
# Cauchy (racine n-ième)
# ============================================================
def test_Cauchy_racine(u_n, n, n_min):
    S_n = sp.Sum(u_n.subs(n, k), (k, n_min, n))
    try:
        v = sp.simplify(sp.Abs(u_n))
        r = sp.simplify(v ** (1 / n))
        L = sp.limit_seq(r, n)
        if L is None or L.is_real is False:
            return None

        if L < 1:
            st.latex(
                r"\text{On calcule } \sqrt[n]{|u_n|}="
                + sp.latex(r)
                + r"\ \text{et } \lim_{n\to\infty}\sqrt[n]{|u_n|}="
                + sp.latex(L)
                + r"<1."
            )
            st.latex(r"\text{Donc } " + sp.latex(S_n) + r"\text{ converge absolument (critère de Cauchy).}")
            return True

        if L > 1:
            st.latex(
                r"\text{On calcule } \sqrt[n]{|u_n|}="
                + sp.latex(r)
                + r"\ \text{et } \lim_{n\to\infty}\sqrt[n]{|u_n|}="
                + sp.latex(L)
                + r">1."
            )
            st.latex(r"\text{Donc } " + sp.latex(S_n) + r"\text{ diverge (critère de Cauchy).}")
            return False

    except:
        pass

    return None


# ============================================================
# Leibniz
# ============================================================
def test_Leibniz(u_n, n, n_min):
    S_n = sp.Sum(u_n.subs(n, k), (k, n_min, n))
    u_s = sp.simplify(u_n)

    if not (detecte_alternance(u_s, n) or detecte_alternance(u_s, n + 1)):
        return None

    a_n = extraire_partie_positive(u_s, n, n_min)
    
    if a_n is None:
        return None
    
    lim = sp.limit_seq(a_n, n)
    if lim != 0:
        return None
    
    ok_pos, msg_pos = positivite(a_n, n, n_min)
    if ok_pos is not True:
        return None

    ok_dec, msg_dec = decroissance(a_n, n, n_min)
    if ok_dec is not True:
        return None

    

    st.latex(msg_pos)
    st.latex(msg_dec)
    st.latex(r"\text{De plus } \lim_{n\to\infty} " + sp.latex(a_n) + r"=0.")
    st.latex(r"\text{Donc } " + sp.latex(S_n) + r"\text{ converge (critère de Leibniz).}")
    return True


# ============================================================
# Comparaison (générale + cas bornés n^p +/- borné)
# ============================================================
def _detect_main_power(den, n):
    """
    Si den contient un terme n ou n^p (p réel), renvoie ce terme (sympy).
    Priorité : n^p si présent, sinon n.
    """
    if not den.is_Add:
        return None
    # cherche n^p
    for t in den.args:
        if t.is_Pow and t.base == n:
            return t
    # sinon n
    for t in den.args:
        if t == n:
            return n
    return None


def _comparison_den_power_plus_bounded(u_s, n, n_min):
    """
    Cas u_s = 1/(n^p + b(n)) avec b(n) bornée via encadrements.
    Décision (si p est comparable) :
    - p>1 => convergence (comparaison à 1/n^p)
    - p=1 => divergence (comparaison à 1/(n+C) ~ 1/n)
    - 0<p<1 => divergence (comparaison à 1/n^p)
    - p<=0 => les termes ne tendent pas vers 0 => divergence
    """
    num, den = u_s.as_numer_denom()
    if num != 1 or not den.is_Add:
        return None

    main = _detect_main_power(den, n)
    if main is None:
        return None

    perturb = sp.simplify(den - main)
    bmin, bmax = encadrements(perturb, n)

    # on n'accepte que des bornes numériques (sin/cos/(-1)^n etc => OK)
    if not (bmin.is_Number and bmax.is_Number):
        return None

    M = sp.Max(sp.Abs(bmin), sp.Abs(bmax))
    p = S.One if main == n else main.exp

    # on affiche l'écriture + encadrement
    st.latex(
        r"\text{On écrit } "
        + sp.latex(den)
        + r"="
        + sp.latex(main)
        + r"+"
        + sp.latex(perturb)
        + r"\ \text{avec } "
        + sp.latex(bmin)
        + r"\le "
        + sp.latex(perturb)
        + r"\le "
        + sp.latex(bmax)
        + r"."
    )

    # p<=0 : termes ne vont pas vers 0
    if p <= 0:
        st.latex(
            r"\text{Ici l'exposant sur } n \text{ vaut } "
            + sp.latex(p)
            + r"\le 0,\ \text{donc } "
            + sp.latex(main)
            + r"\text{ ne tend pas vers } +\infty.\ \text{Ainsi } "
            + sp.latex(u_s)
            + r"\text{ ne tend pas vers }0,\ \text{donc la série diverge.}"
        )
        return False

    # p>1 : convergence
    if p > 1:
        # pour n>=2, on a n^p - M >= (1/2)n^p dès que n^p >= 2M
        # (on reste sur une justification lisible, sans introduire une lettre non expliquée)
        st.latex(
            r"\text{Pour } n \text{ assez grand, on a } "
            + sp.latex(main)
            + r"-"
            + sp.latex(M)
            + r"\ge \frac{1}{2}"
            + sp.latex(main)
            + r"\ \Rightarrow\ "
            + sp.latex(den)
            + r"\ge \frac{1}{2}"
            + sp.latex(main)
            + r"."
        )
        st.latex(
            r"\text{Donc } "
            + sp.latex(u_s)
            + r"\le \frac{2}{"
            + sp.latex(main)
            + r"}=\frac{2}{n^{"
            + sp.latex(p)
            + r"}}."
        )
        st.latex(
            r"\text{Or } \sum \frac{1}{n^{"
            + sp.latex(p)
            + r"}} \text{ converge (Riemann), donc } \sum "
            + sp.latex(u_s)
            + r"\text{ converge (comparaison).}"
        )
        return True

    # p=1 : divergence
    if sp.simplify(p - 1).equals(0):
        st.latex(
            r"\text{On a } "
            + sp.latex(den)
            + r"\le "
            + sp.latex(main)
            + r"+"
            + sp.latex(M)
            + r"\ \Rightarrow\ "
            + sp.latex(u_s)
            + r"\ge \frac{1}{n+"
            + sp.latex(M)
            + r"}."
        )
        st.latex(
            r"\text{Or } \sum \frac{1}{n+"
            + sp.latex(M)
            + r"} \text{ diverge (comparable à } \sum \frac{1}{n}\text{), donc } \sum "
            + sp.latex(u_s)
            + r"\text{ diverge (comparaison).}"
        )
        return False

    # 0<p<1 : divergence
    if p < 1:
        st.latex(
            r"\text{On a } "
            + sp.latex(den)
            + r"\le "
            + sp.latex(main)
            + r"+"
            + sp.latex(M)
            + r"\ \Rightarrow\ "
            + sp.latex(u_s)
            + r"\ge \frac{1}{n^{"
            + sp.latex(p)
            + r"}+"
            + sp.latex(M)
            + r"}."
        )
        st.latex(
            r"\text{Pour } n \text{ assez grand, } n^{"
            + sp.latex(p)
            + r"}+"
            + sp.latex(M)
            + r"\le 2n^{"
            + sp.latex(p)
            + r"}\ \Rightarrow\ "
            + sp.latex(u_s)
            + r"\ge \frac{1}{2}\cdot\frac{1}{n^{"
            + sp.latex(p)
            + r"}}."
        )
        st.latex(
            r"\text{Or } \sum \frac{1}{n^{"
            + sp.latex(p)
            + r"}} \text{ diverge (Riemann, exposant }\le 1\text{), donc } \sum "
            + sp.latex(u_s)
            + r"\text{ diverge (comparaison).}"
        )
        return False

    return None


def test_comparaison(u_n, n, n_min):
    S_n = sp.Sum(u_n.subs(n, k), (k, n_min, n))
    u_s = sp.simplify(u_n)

    # ------------------------------------------------------------
    # Cas : 1/(log(n))^q (sans n) => divergence par comparaison à 1/n
    # ------------------------------------------------------------
    try:
        num, den = u_s.as_numer_denom()
        if num == 1:
            if den == sp.log(n):
                st.latex(
                    r"\text{Pour } n\ge 3,\ \log(n)<n\ \Rightarrow\ "
                    + sp.latex(u_s)
                    + r"=\frac{1}{\log(n)}\ge \frac{1}{n}."
                )
                st.latex(
                    r"\text{Or } \sum \frac{1}{n} \text{ diverge, donc } "
                    + sp.latex(S_n)
                    + r"\text{ diverge (comparaison).}"
                )
                return False

            if den.is_Pow:
                base, exp = den.as_base_exp()
                if base == sp.log(n) and exp.is_positive:
                    st.latex(
                        r"\text{Pour } n\ge 3,\ \log(n)<n\ \Rightarrow\ "
                        + sp.latex(u_s)
                        + r"=\frac{1}{(\log(n))^{"
                        + sp.latex(exp)
                        + r"}}\ge \frac{1}{n}."
                    )
                    st.latex(
                        r"\text{Or } \sum \frac{1}{n} \text{ diverge, donc } "
                        + sp.latex(S_n)
                        + r"\text{ diverge (comparaison).}"
                    )
                    return False
    except:
        pass

    # ------------------------------------------------------------
    # Cas : 1/(n^p + borné)  (sin/cos/(-1)^n/combinaisons bornées)
    # ------------------------------------------------------------
    try:
        num, den = u_s.as_numer_denom()
        if num == 1 and den.is_Add:
            r = _comparison_den_power_plus_bounded(u_s, n, n_min)
            if r is not None:
                return r
    except:
        pass

    # ------------------------------------------------------------
    # Sinon : MAJORATION automatique uniquement (pour conclure la convergence).
    # (On n'utilise pas une minoration automatique pour conclure la divergence.)
    # ------------------------------------------------------------
    try:
        a, b = encadrements(u_n, n)
        max_un = max_borne(a, b)

        # majoration -> convergence
        try:
            test_sup = critere(max_un, n, n_min)
            if test_sup is True:
                st.latex(
                    r"\text{On a } "
                    + sp.latex(u_n)
                    + r"\le "
                    + sp.latex(max_un)
                    + r"\ \text{pour } n\ge "
                    + sp.latex(n_min)
                    + r"."
                )
                st.latex(
                    r"\text{Comme } \sum "
                    + sp.latex(max_un)
                    + r"\text{ converge, } "
                    + sp.latex(S_n)
                    + r"\text{ converge (comparaison).}"
                )
                return True
        except:
            pass

    except:
        pass

    return None


# ============================================================
# Équivalence
# ============================================================
def test_equivalence(u_n, n, n_min):
    S_n = sp.Sum(u_n.subs(n, k), (k, n_min, n))
    eq = dl_equivalent(u_n, n)
    if eq is None:
        return None

    C = critere(eq, n, n_min)
    if C is True:
        st.latex(r"\text{Donc } " + sp.latex(S_n) + r"\text{ converge par équivalence.}")
        return True
    if C is False:
        st.latex(r"\text{Donc } " + sp.latex(S_n) + r"\text{ diverge par équivalence.}")
        return False

    return None


# ============================================================
# Convergence absolue
# ============================================================
def convergence_absolue(u_n, n, n_min):
    v = sp.Abs(u_n)

    # critères "propres" sur |u_n|, sans sur-traitement
    for f in (test_Riemann, test_polylog, test_bertrand, test_Cauchy_racine, test_Alembert):
        r = f(v, n, n_min)
        if r is True:
            st.latex(r"\text{Donc la série } \sum " + sp.latex(u_n) + r"\text{ converge absolument.}")
            return True
        if r is False:
            # divergence de la valeur absolue ne conclut pas sur la série initiale
            return None

    return None

def divergence_grossiere(u_n , n) :
    if diverge(u_n, n):
            lim = sp.limit_seq(u_n, n)
            if lim is not None and not getattr(lim, "is_zero", False):
                st.latex(r"\text{La condition nécessaire n’est pas vérifiée : divergence grossiere.}")
                return True
    return False






# ============================================================
# Orchestrateur
# ============================================================
def critere(u_n, n, n_min):
    if not divergence_grossiere(u_n, n):
        # protection contre appels circulaires
        if not hasattr(critere, "_pile"):
            critere._pile = []

        u_s = sp.simplify(u_n)
        for old in critere._pile:
            if sp.simplify(old - u_s).equals(0):
                return None

        critere._pile.append(u_s)

        try:
            # ordre stratégique : alternée -> Riemann -> logs itérés -> Bertrand -> racine -> ratio -> comparaison -> équivalence
            for f in (
                test_Leibniz,
                test_Riemann,
                test_polylog,
                test_bertrand,
                test_Cauchy_racine,
                test_Alembert,
                test_comparaison,
            ):
                r = f(u_n, n, n_min)
                if r is not None:
                    return r

            # absolue (si ça conclut)
            if convergence_absolue(u_n, n, n_min) is True:
                return True

            # équivalence en dernier recours
            r = test_equivalence(u_n, n, n_min)
            if r is not None:
                return r

            return None

        finally:
            critere._pile.pop()


import numpy as np
import sympy as sp
from sympy import S
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify


# =============================================================
# 0) Faux Streamlit (fallback si jamais "st" n'existe pas déjà)
#    IMPORTANT : dans TON projet, "st" existe déjà (core SymPy).
# =============================================================
try:
    st  # noqa: F821
except NameError:
    import io, base64

    class _StreamlitCompat:
        def __init__(self):
            self._out = []

        def set_out(self, out):
            self._out = out

        def latex(self, msg):
            self._out.append({"type": "latex", "content": msg})

        def info(self, msg):
            self._out.append({"type": "info", "content": str(msg)})

        def error(self, msg):
            self._out.append({"type": "error", "content": str(msg)})

        def pyplot(self, fig):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode("utf-8")
            self._out.append({"type": "plot", "content": b64})

    st = _StreamlitCompat()


# =============================================================
# 1) LaTeX : messages propres
# =============================================================
def _latex_rouge(msg: str) -> str:
    safe = (msg or "").replace("\\", r"\textbackslash ").replace("_", r"\_")
    return r"\textcolor{red}{\text{" + safe + r"}}"


def _latex_info(msg: str) -> str:
    safe = (msg or "").replace("\\", r"\textbackslash ").replace("_", r"\_")
    return r"\text{" + safe + r"}"


def _latex_intervalle(intervalle) -> str:
    if intervalle == S.Reals:
        return r"\mathbb{R}"
    if intervalle == S.EmptySet:
        return r"\varnothing"
    try:
        return sp.latex(intervalle)
    except Exception:
        return r"\varnothing"


# =============================================================
# 2) Parsing intervalle depuis payload (compatible app.js)
# =============================================================
def _parse_interval_payload(data):
    borne_gauche = (data.get("interval_left") or "").strip()
    borne_droite = (data.get("interval_right") or "").strip()
    ferme_gauche = bool(data.get("interval_left_closed", False))
    ferme_droite = bool(data.get("interval_right_closed", False))

    if not borne_gauche and not borne_droite:
        return S.Reals

    def _vers_borne(texte):
        texte = (texte or "").strip()
        if not texte:
            return None
        if texte in ("oo", "+oo", "∞", "+∞"):
            return sp.oo
        if texte in ("-oo", "-∞"):
            return -sp.oo
        try:
            return sp.sympify(texte)
        except Exception:
            return None

    a = _vers_borne(borne_gauche)
    b = _vers_borne(borne_droite)

    if a is None:
        a = -sp.oo
    if b is None:
        b = sp.oo

    if a == -sp.oo and b == sp.oo:
        return S.Reals

    left_open = True if a == -sp.oo else (not ferme_gauche)
    right_open = True if b == sp.oo else (not ferme_droite)

    try:
        if (a.is_real is True) and (b.is_real is True):
            if sp.simplify(a - b).is_positive is True:
                return S.EmptySet
            if sp.simplify(a - b).equals(0):
                if left_open or right_open:
                    return S.EmptySet
    except Exception:
        pass

    if a == -sp.oo:
        return sp.Interval(-sp.oo, b, left_open=True, right_open=right_open)
    if b == sp.oo:
        return sp.Interval(a, sp.oo, left_open=left_open, right_open=True)

    return sp.Interval(a, b, left_open=left_open, right_open=right_open)


def _intervalle_est_vide(intervalle):
    try:
        return intervalle == S.EmptySet
    except Exception:
        return False


# =============================================================
# 3) Limite simple (suite/série) : robuste
# =============================================================
def _limite_simple(expr, n):
    try:
        L = sp.limit_seq(expr, n)
        if L is None or isinstance(L, sp.Limit):
            return None, False
        return sp.simplify(L), True
    except Exception:
        return None, False


# =============================================================
# 4) Numérique robuste : eval vecteur
# =============================================================
def _eval_vecteur(x, expr, xs):
    if expr is None:
        return np.full(xs.shape, np.nan, dtype=float)
    try:
        f_num = sp.lambdify(x, expr, "numpy")
        with np.errstate(all="ignore"):
            y = f_num(xs)
        if np.isscalar(y):
            return np.full(xs.shape, float(y), dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.size != xs.size:
            return np.full(xs.shape, np.nan, dtype=float)
        return y
    except Exception:
        return np.full(xs.shape, np.nan, dtype=float)


def _bornes_intervalle_pour_graphe(intervalle):
    if intervalle == S.Reals:
        return -5.0, 5.0
    if isinstance(intervalle, sp.Interval):
        try:
            xmin = float(sp.N(intervalle.inf))
            xmax = float(sp.N(intervalle.sup))
            if (not np.isfinite(xmin)) or (not np.isfinite(xmax)) or (xmin == xmax):
                return -5.0, 5.0
            if (xmin == -np.inf) or (xmax == np.inf):
                return -5.0, 5.0
            return xmin, xmax
        except Exception:
            return -5.0, 5.0
    return -5.0, 5.0


# =============================================================
# 5) Graphes suites de fonctions / séries de fonctions
# =============================================================
def _plot_famille_et_limite(f_n, f_lim, x, intervalle, n, n_min=1, n_max=20):
    def _remplacer_n(expr, valeur):
        try:
            subs_map = {s: valeur for s in expr.free_symbols if getattr(s, "name", "") == "n"}
            if subs_map:
                return sp.simplify(expr.subs(subs_map))
            return sp.simplify(expr.subs(n, valeur))
        except Exception:
            return expr

    xmin, xmax = _bornes_intervalle_pour_graphe(intervalle)
    xs = np.linspace(xmin, xmax, 1600)
    fig, ax = plt.subplots(figsize=(18, 5))

    cmap = plt.get_cmap("tab20")
    couleurs = [cmap(i) for i in range(20)]

    k = 0
    for i in range(int(n_min), int(n_max) + 1):
        f_i = _remplacer_n(f_n, i)
        y_i = _eval_vecteur(x, f_i, xs)
        ax.plot(xs, y_i, color=couleurs[k % 20], linewidth=1.6, alpha=0.95, label=rf"$f_{{{i}}}(x)$")
        k += 1

    if f_lim is not None:
        y_lim = _eval_vecteur(x, f_lim, xs)
        ax.plot(xs, y_lim, color="black", linewidth=3.0, label=r"$f(x)=\lim_{n\to\infty} f_n(x)$")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, alpha=0.25)
    st.pyplot(fig)


def _plot_series_partielles(u_n, x, intervalle, n, n_min=1, N_max=20):
    xmin, xmax = _bornes_intervalle_pour_graphe(intervalle)
    xs = np.linspace(xmin, xmax, 1400)

    N_max = int(N_max)
    n_min = int(n_min)
    if N_max < n_min:
        N_max = n_min

    fig, ax = plt.subplots(figsize=(18, 5))
    cmap = plt.get_cmap("tab20")

    for N in range(n_min, N_max + 1):
        Sx = np.zeros_like(xs, dtype=float)
        for k in range(n_min, N + 1):
            try:
                terme_k = sp.simplify(u_n.subs(n, k))
                Sx += _eval_vecteur(x, terme_k, xs)
            except Exception:
                pass
        ax.plot(xs, Sx, color=cmap((N - 1) % 20), linewidth=1.8, label=rf"$S_{{{N}}}(x)$")

    ax.set_xlabel("x")
    ax.set_ylabel("Sommes partielles")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, alpha=0.25)
    st.pyplot(fig)


def etude_suite_de_fonctions(f_n, n, x, intervalle, n_min=1, n_max=20):
    if _intervalle_est_vide(intervalle):
        st.latex(_latex_rouge("L’intervalle est vide."))
        return

    try:
        f_n = sp.simplify(sp.sympify(f_n))
    except Exception:
        st.latex(_latex_rouge("Expression invalide pour f_n."))
        return

    latex_I = _latex_intervalle(intervalle)

    st.latex(r"\text{Suite de fonctions définie par }\quad f_n(x)=" + sp.latex(f_n) + r"\quad \text{sur }" + latex_I)

    f_lim, ok = _limite_simple(f_n, n)
    if ok:
        st.latex(r"\forall x\in " + latex_I + r",\quad \lim_{n\to+\infty} f_n(x)=" + sp.latex(f_lim))
    else:
        st.latex(r"\text{La limite simple de la suite }(f_n)\text{ n’existe pas.}")
        f_lim = None

    _plot_famille_et_limite(f_n, f_lim, x, intervalle, n, n_min=int(n_min), n_max=int(n_max))


def etude_serie_de_fonctions(u_n, n, x, intervalle, n_min=1, N_graphe=20):
    if _intervalle_est_vide(intervalle):
        st.latex(_latex_rouge("L’intervalle est vide."))
        return

    try:
        u_n = sp.simplify(sp.sympify(u_n))
    except Exception:
        st.latex(_latex_rouge("Expression invalide pour u_n(x)."))
        return

    latex_I = _latex_intervalle(intervalle)
    n_min_int = int(n_min)
    N_graphe_int = int(N_graphe)

    st.latex(
        r"\text{Série de fonctions }\sum_{n\ge " + sp.latex(n_min_int) + r"} u_n(x),\quad "
        r"u_n(x)=" + sp.latex(u_n) + r"\quad \text{sur }" + latex_I
    )

    u_lim, ok = _limite_simple(u_n, n)
    if ok:
        st.latex(r"\forall x\in " + latex_I + r",\quad \lim_{n\to+\infty} u_n(x)=" + sp.latex(u_lim))
    else:
        st.latex(r"\text{La limite simple du terme général }u_n(x)\text{ n’existe pas.}")

    _plot_series_partielles(u_n, x, intervalle, n, n_min=n_min_int, N_max=max(n_min_int, N_graphe_int))


# =============================================================
# 6) Série entière : graphes + reconnaissance a_n(z(x))^n
# =============================================================
def _trace_sommes_partielles_et_limite(u_n_x, x, intervalle, n, n_min=0, N_max=20, f_lim=None):
    xmin, xmax = _bornes_intervalle_pour_graphe(intervalle)
    xs = np.linspace(xmin, xmax, 1600)

    fig, ax = plt.subplots(figsize=(18, 10))
    cmap = plt.get_cmap("tab20")

    Y_MAX = 20 # limite verticale d'affichage

    n_min = int(n_min)
    N_max = int(N_max)
    if N_max < n_min:
        N_max = n_min

    for N in range(n_min, N_max + 1):
        Sx = np.zeros_like(xs, dtype=float)
        for j in range(n_min, N + 1):
            try:
                terme_j = sp.simplify(u_n_x.subs(n, j))
                Sx += _eval_vecteur(x, terme_j, xs)
            except Exception:
                pass

        # Écrêtage VISUEL uniquement
        Sx = np.clip(Sx, -Y_MAX, Y_MAX)

        ax.plot(
            xs, Sx,
            color=cmap((N - n_min) % 20),
            linewidth=1.7,
            alpha=0.98,
            label=rf"$S_{{{N}}}(x)$"
        )

    if f_lim is not None:
        y_lim = _eval_vecteur(x, f_lim, xs)

        # Écrêtage VISUEL de la limite aussi
        y_lim = np.clip(y_lim, -Y_MAX, Y_MAX)

        ax.plot(
            xs, y_lim,
            color="black",
            linewidth=3.2,
            label=r"$f(x)$"
        )

    ax.set_xlabel("x")
    ax.set_ylabel("Sommes partielles")
    ax.set_ylim(-Y_MAX, Y_MAX)
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, alpha=0.25)

    st.pyplot(fig)


def _extraire_forme_puissance(expr, n, x):
    expr = sp.simplify(expr)

    try:
        for p in expr.atoms(sp.Pow):
            base, exp_ = p.as_base_exp()
            if sp.simplify(exp_ - n).equals(0) and base.has(x) and (not base.has(n)):
                a_n = sp.simplify(expr / p)
                return sp.simplify(a_n), sp.simplify(base)
    except Exception:
        pass

    try:
        if expr.has(x**n):
            a_n = sp.simplify(expr / (x**n))
            return sp.simplify(a_n), x
    except Exception:
        pass

    return None, None


# =============================================================
# 7) Rayon : Cauchy–Hadamard (robuste sans limit_sup)
# =============================================================
def _limsup_racine_absolue(a_n, n):
    """
    Renvoie limsup (|a_n|)^(1/n) si SymPy arrive à donner une borne d'accumulation.
    Utilise limit_seq, qui peut renvoyer AccumulationBounds(min,max).
    """
    try:
        from sympy.calculus.accumulationbounds import AccumulationBounds
        seq = sp.Abs(a_n) ** (sp.Rational(1, 1) / n)
        L = sp.limit_seq(seq, n)
        if L is None or isinstance(L, sp.Limit):
            return None
        if isinstance(L, AccumulationBounds):
            return sp.simplify(L.max)
        return sp.simplify(L)
    except Exception:
        return None


def _lim_racine_absolue(a_n, n):
    try:
        seq = sp.Abs(a_n) ** (sp.Rational(1, 1) / n)
        L = sp.limit_seq(seq, n)
        if L is None or isinstance(L, sp.Limit):
            return None
        return sp.simplify(L)
    except Exception:
        return None


def _rayon_depuis_L(L):
    if L is None:
        return None
    try:
        if sp.simplify(L).equals(0):
            return sp.oo
        if L == sp.oo:
            return sp.Integer(0)
        return sp.simplify(1 / L)
    except Exception:
        return None


import sympy as sp
import sympy as sp

import sympy as sp

import sympy as sp

import sympy as sp

def _extraire_forme_puissance(expr, n, x):
    try:
        expr = sp.powsimp(sp.expand_power_exp(expr), combine='all')
        
        if not expr.has(x):
            return expr, sp.Integer(1)

        res_an, res_z = None, None

        a_n_simple = sp.simplify(expr / (x**n))
        if not a_n_simple.has(x):
            res_an, res_z = a_n_simple, x

        if res_an is None:
            for p in expr.atoms(sp.Pow):
                base, exp_ = p.as_base_exp()
                if not base.has(x) or base.has(n):
                    continue

                exp_s = sp.simplify(exp_)
                k_mul = sp.simplify(exp_s / n)

                if k_mul.is_integer and not k_mul.equals(0):
                    try:
                        c = facteur2(base)
                    except:
                        c = base.as_coeff_Mul()[0]

                    z_raw = sp.simplify(base / c)
                    
                    k_z = 1
                    if z_raw.is_Pow:
                        b, e = z_raw.as_base_exp()
                        if b == x and e.is_integer and e > 0:
                            k_z = e
                        else: continue
                    elif z_raw == x:
                        k_z = 1
                    else:
                        continue

                    a_n = sp.simplify((expr / p) * (c**exp_s))
                    if not a_n.has(x):
                        res_an, res_z = a_n, x**(k_z * k_mul)
                        break

        if res_an is None:
            try:
                coeff_indep, x_part = expr.as_independent(x, as_Add=False)
                if x_part.has(x):
                    expo_x = sp.simplify(sp.log(x_part, x))
                    k_final = sp.simplify(expo_x / n)
                    if not k_final.has(x) and k_final > 0:
                        res_an, res_z = coeff_indep, x**k_final
            except:
                pass

        if res_an is not None and res_z is not None:
            try:
                base_z, exp_z = res_z.as_base_exp()
                try:
                    c_final = facteur2(base_z)
                except:
                    c_final = base_z.as_coeff_Mul()[0]
                
                if c_final != 1:
                    res_an = sp.simplify(res_an * (c_final**(exp_z * n)))
                    res_z = sp.simplify((base_z / c_final)**exp_z)
            except:
                pass
            return sp.simplify(res_an), res_z

    except:
        pass
    
    return None, None



def _rayon_cauchy_hadamard(a_n, n):
    Lsimple = _lim_racine_absolue(a_n, n)
    Lsup = _limsup_racine_absolue(a_n, n)
    R = _rayon_depuis_L(Lsup if Lsup is not None else Lsimple)
    return R, Lsimple, Lsup


def _rayon_dalembert(a_n, n):
    try:
        q = sp.simplify(sp.Abs(a_n.subs(n, n + 1) / a_n))
        L = sp.limit(q, n, sp.oo)
        if L is None or isinstance(L, sp.Limit):
            return None, None
        L = sp.simplify(L)
        if sp.simplify(L).equals(0):
            return sp.oo, L
        if L == sp.oo:
            return sp.Integer(0), L
        return sp.simplify(1 / L), L
    except Exception:
        return None, None


# =============================================================
# 8) Somme fermée (si possible)
# =============================================================
def _nettoyer_expression_somme(expr):
    try:
        e = sp.simplify(expr)
    except Exception:
        e = expr

    try:
        if isinstance(e, sp.Piecewise) and len(e.args) >= 1:
            e = e.args[0][0]
    except Exception:
        pass

    try:
        e = e.rewrite(sp.factorial)
    except Exception:
        pass

    try:
        e = sp.hyperexpand(e)
    except Exception:
        pass

    try:
        e = sp.simplify(e)
    except Exception:
        pass

    try:
        if e.has(sp.I):
            e = sp.re(e)
            e = sp.simplify(e)
    except Exception:
        pass

    return e


def _serie_vers_fonction(a_n, n, z, n_min=0):
    try:
        t = sp.Symbol("t", real=True)
        S_formelle = sp.Sum(a_n * (t ** n), (n, int(n_min), sp.oo))
        f_t = S_formelle.doit()
        f_t = _nettoyer_expression_somme(f_t)
        f_z = sp.simplify(f_t.subs(t, z))
        return f_z
    except Exception:
        return None


def _ratio_geometrique(u_n_x, n, x):
    try:
        q = sp.simplify(u_n_x.subs(n, n + 1) / u_n_x)
        if (not q.has(n)) and q.has(x):
            return sp.simplify(q)
        return None
    except Exception:
        return None


# =============================================================
# 9) Étude : série entière / série de puissances sur I
# =============================================================
def etude_serie_entiere_convergence(expr_terme, n, x, intervalle, n_min=1, N_graphe=20):
    # ==========================================================
    # 0) Vérification de l’intervalle
    # ==========================================================
    if _intervalle_est_vide(intervalle):
        st.latex(_latex_rouge(
            "L’intervalle d’étude choisi est vide. Veuillez corriger l’intervalle."
        ))
        return

    # ==========================================================
    # 1) Lecture et validation du terme général
    # ==========================================================
    try:
        u_n_x = sp.simplify(sp.sympify(expr_terme))
    except Exception:
        st.latex(_latex_rouge(
            "L’expression du terme général est invalide. Veuillez la corriger."
        ))
        return

    try:
        n_min = int(n_min)
        N_graphe = int(N_graphe)
    except Exception:
        st.latex(_latex_rouge(
            "L’indice de départ doit être un entier valide."
        ))
        return

    latex_I = _latex_intervalle(intervalle)

    # ==========================================================
    # 2) Domaine de définition en n (INDICE)
    # ==========================================================
    try:
        Dom_n = sp.calculus.util.continuous_domain(u_n_x, n, S.Reals)
        if not Dom_n.contains(n_min):
            st.latex(_latex_rouge(
                "La série n’est pas définie pour l’indice de départ choisi. "
                "Veuillez corriger l’indice de départ."
            ))
            return
    except Exception:
        pass
    # ==========================================================
    # 3) Domaine de définition en x (VARIABLE)
    # ==========================================================
    try:
        Dom_x = sp.calculus.util.continuous_domain(u_n_x, x, S.Reals)
        if intervalle != S.Reals:
            intervalle_eff = intervalle.intersect(Dom_x)
        else:
            intervalle_eff = Dom_x

        if intervalle_eff == S.EmptySet:
            st.latex(_latex_rouge(
                "Aucune valeur de l’intervalle choisi ne respecte le domaine "
                "de définition de la fonction. Veuillez corriger l’intervalle."
            ))
            return
    except Exception:
        intervalle_eff = intervalle  # fallback sûr

    # ==========================================================
    # 4) Énoncé mathématique clair
    # ==========================================================
    st.latex(
        r"\text{On considère la série entière}\quad "
        r"\sum u_n(x)\quad \text{définie sur}\quad I=" + _latex_intervalle(intervalle_eff) + r"."
    )
    st.latex(
        r"\text{Le terme général est donné par}\quad u_n(x)=" + sp.latex(u_n_x) + r"."
    )

    # ==========================================================
    # 5) Reconnaissance série géométrique (accélération)
    # ==========================================================
    f_geo = None
    try:
        qx = _ratio_geometrique(u_n_x, n, x)
        if qx is not None:
            u0 = sp.simplify(u_n_x.subs(n, n_min))
            f_geo = sp.simplify(u0 / (1 - qx))

            st.latex(
                r"\text{Le rapport }\frac{u_{n+1}(x)}{u_n(x)}=" + sp.latex(qx)
                + r"\text{ est indépendant de l’indice.}"
            )
            st.latex(
                r"\text{La série est donc géométrique, et elle converge pour }"
                r"|" + sp.latex(qx) + r"|<1."
            )
    except Exception:
        f_geo = None

    # ==========================================================
    # 6) Mise sous la forme a_n (z(x))^n
    # ==========================================================
    a_n, z = _extraire_forme_puissance(u_n_x, n, x)

    if a_n is None or z is None:
        st.latex(
            r"\text{La série ne peut pas être mise sous la forme d’une série entière standard.}"
        )
        st.latex(r"\text{On procède néanmoins à une visualisation des sommes partielles.}")
        _trace_sommes_partielles_et_limite(
            u_n_x, x, intervalle_eff, n,
            n_min=n_min, N_max=max(N_graphe, n_min),
            f_lim=f_geo
        )
        return

    st.latex(
        r"\text{On écrit le terme général sous la forme}\quad "
        r"u_n(x)=a_n\,\bigl(z(x)\bigr)^n."
    )
    st.latex(r"u_n(x)" + r" =" + sp.latex(a_n) + sp.latex(z**n))
    # ==========================================================
    # 7) Calcul du rayon – Cauchy–Hadamard puis d’Alembert
    # ==========================================================
    R_prime = None

    Lsup = _limsup_racine_absolue(a_n, n)
    if Lsup is not None:
        st.latex(r"\text{On determine R' = } R(\sum a_nx^n) \text{ par la formule de Cauchy - Hadamard }")
        st.latex(
            r"\ell=\limsup_{n\to+\infty}\sqrt[n]{|a_n|}="+ r"\limsup_{n\to+\infty}\sqrt[n]{" +sp.latex(sp.Abs(a_n))+r"}=" + sp.latex(Lsup)
        )
        R_prime = _rayon_depuis_L(Lsup)
        st.latex(
            r"R'=\frac{1}{\ell}=" + sp.latex(R_prime)
        )
    else:
        L = _lim_racine_absolue(a_n, n)
        if L is not None:
            st.latex(r"\text{On determine R' = } R(\sum a_nz_n) \text{ par la formule de Cauchy - Hadamard }")
            st.latex(
                r"\lim_{n\to+\infty}\sqrt[n]{|a_n|}=" + sp.latex(L)
            )
            R_prime = _rayon_depuis_L(L)
            st.latex(
                r"R'=\frac{1}{\ell}=" + sp.latex(R_prime)
            )
        else:
            R2, Ld = _rayon_dalembert(a_n, n)
            if R2 is None:
                st.latex(_latex_rouge(
                    "Impossible de déterminer le rayon de convergence avec les critères classiques."
                ))
                return
            R_prime = R2
            st.latex(r"\text{On determine R' = } R(\sum a_nz^n) \text{ par le critère d'Alembert}")
            st.latex(
                r"\lim_{n\to+\infty}\left|\frac{a_{n+1}}{a_n}\right|="
                + sp.latex(Ld)
            )
            st.latex(
                r"R'=\frac{1}{\ell}=" + sp.latex(R_prime)
            )

    # ==========================================================
    # 8) Ajustement du rayon si z(x)=x^k avec k>1
    # ==========================================================
    k_val = 1
    try:
        if z.is_Pow and z.base == x and z.exp.is_integer and z.exp > 1:
            k_val = int(z.exp)
    except Exception:
        k_val = 1

    if k_val != 1:
        R = sp.simplify(R_prime ** (sp.Rational(1, k_val)))
        st.latex(
            r"\text{Puisque la variable apparaît sous la forme d’une puissance.}" + r"( z(x)= "+ sp.latex(z)+ r")"
        )
        st.latex(
            r"R=\sqrt[" + sp.latex(k_val) + r"]{R'}=" + sp.latex(R)
        )
    else:
        R = R_prime
        st.latex(
            r"\text{Le rayon de convergence est}\quad R=" + sp.latex(R)
        )

    # ==========================================================
    # 9) Conclusion générale (réelle et complexe)
    # ==========================================================
    st.latex(r"\text{Conclusion générale :}")
    st.latex(
        sp.latex(sp.Abs(x))+ r"<" + sp.latex(R)+ r"\ \Longleftrightarrow\ \text{convergence absolue de la série de terme général } a_n" + sp.latex(x**(k_val*n))
    )
    #"|z(x)|"+ "=" +
    st.latex(
        sp.latex(sp.Abs(x))+ r">" + sp.latex(R)+ r"\ \Longleftrightarrow\ \text{divergence grossière de la série de terme général } a_n" + sp.latex(x**(k_val*n))
    )
    st.latex(r"\text{Cela vaut aussi  pour tout complexe } z \text{ à la place du réel }x")

    st.latex(r"\begin{aligned}"
    r"&\text{Dans le plan complexe, la série entière de coefficients } (a_n) \text{ converge normalement (donc uniformément) }\\"
    r"&\text{sur tout disque fermé } \{ z \in \mathbb{C} \mid |z| \le r \},\ \forall r < "+ sp.latex(R)+
    r"\end{aligned}"
)

    # ==========================================================
    # 10) Ensembles sur l’intervalle effectif
    # ==========================================================
    try:
        E_int = sp.solveset(sp.Abs(x) < R, x, domain=S.Reals).intersect(intervalle_eff)
        E_ext = sp.solveset(sp.Abs(x) > R, x, domain=S.Reals).intersect(intervalle_eff)
        E_bord = sp.solveset(sp.Eq(sp.Abs(x), R), x, domain=S.Reals).intersect(intervalle_eff)

        st.latex(r"\text{Sur l'intervalle de départ I = }" + sp.latex(intervalle) + r"\text{ On a :}")

        if E_int != S.EmptySet:
            st.latex(r"\text{Convergence absolue sur }\ " + sp.latex(E_int) + r".")
        if E_ext != S.EmptySet:
            st.latex(r"\text{Divergence sur }\ " + sp.latex(E_ext) + r".")
        if E_bord != S.EmptySet:
            st.latex(
                r"\text{Sur le bord, une étude spécifique est nécessaire sur }\ "
                + sp.latex(E_bord) + r"."
            )
    except Exception:
        pass

    # ==========================================================
    # 11) Étude du bord via critères numériques
    # ==========================================================
    if R not in (0, sp.oo):
        try:
            bord = sp.solveset(sp.Eq(sp.Abs(x), R), x, domain=S.Reals).intersect(intervalle_eff)
            if isinstance(bord, sp.FiniteSet):
                i = 1
                for x0 in bord:
                    u_bord = sp.simplify(u_n_x.subs(x, x0))
                    S_bord = sp.Sum(u_bord.subs(n, k), (k, n_min, n))
                    
                    st.latex(sp.latex(i) +
                        r"\text{) Au point }" + sp.latex(x0)
                        + r",\ \text{on étudie la série numérique}"+ sp.latex(S_bord)
                    )
                    i = i + 1
                    try:
                        r_dec = critere(u_bord, n, n_min)
                        if r_dec is True:
                            st.latex(r"\text{Conclusion : convergence.}")
                        elif r_dec is False:
                            st.latex(r"\text{Conclusion : divergence.}")
                    except Exception:
                        pass
        except Exception:
            pass

    # ==========================================================
    # 12) Fonction somme (si possible)
    # ==========================================================
    f_lim = f_geo
    if f_lim is None:
        try:
            f_tmp = _serie_vers_fonction(a_n, n, z, n_min=n_min)
            if f_tmp is not None:
                f_lim = f_tmp
        except Exception:
            pass

    if f_lim is not None:
        st.latex(
            r"\text{Lorsque la série converge, elle définit la fonction somme}\quad "
            r"f(x)=" + sp.latex(f_lim) + r"."
        )

    # ==========================================================
    # 13) Graphe (TOUJOURS)
    # ==========================================================
    st.latex(r"\text{Visualisation des sommes partielles :}")
    _trace_sommes_partielles_et_limite(
        u_n_x, x, intervalle_eff, n,
        n_min=n_min, N_max=max(N_graphe, n_min),
        f_lim=f_lim
    )
# =============================================================
# 10) Développement en série entière
# =============================================================
def developpement_en_serie_entiere(f_expr, x, ordre=10):
    try:
        f = sp.simplify(sp.sympify(f_expr))
    except Exception:
        st.latex(_latex_rouge("Impossible : l’expression de la fonction est invalide."))
        return None, None, None

    ordre = int(ordre)
    st.latex(r"\text{On considère la fonction}\quad f(x)=" + sp.latex(f) + r".")

    somme = None
    n = sp.Symbol("n", integer=True, nonnegative=True)

    try:
        fps = sp.fps(f, x, 0)
        somme = fps.infinite
    except Exception:
        somme = None

    if somme is None:
        try:
            coeffs = [sp.simplify(sp.diff(f, x, k).subs(x, 0) / sp.factorial(k)) for k in range(0, ordre + 1)]
            somme = sp.Sum(coeffs[n] * x**n, (n, 0, ordre))
        except Exception:
            somme = None

    if somme is not None:
        st.latex(r"\text{Développement en série entière au voisinage de }0:\quad " + sp.latex(somme))

    ser_poly = None
    try:
        ser_poly = sp.series(f, x, 0, ordre + 1).removeO()
    except Exception:
        ser_poly = None

    R = None
    try:
        if isinstance(somme, sp.Sum):
            term = sp.simplify(somme.function)
            a_n = sp.simplify(term / (x**n))
            if not a_n.has(x):
                try:
                    R = sp.convergence_radius(a_n, n, x)
                except Exception:
                    R = None
    except Exception:
        R = None

    if R is not None:
        if R == sp.oo:
            st.latex(r"\text{Rayon de convergence : }R=+\infty\ \text{(validité sur }\mathbb{R}\text{).}")
        elif R == 0:
            st.latex(r"\text{Rayon de convergence : }R=0.")
        else:
            st.latex(r"\text{Rayon de convergence : }R=" + sp.latex(R) + r"\quad \Rightarrow\quad x\in(-R,R).")

    return f, ser_poly, R


def _trace_fonction_et_tronques(f, x, intervalle, ordre_max=10):
    xmin, xmax = _bornes_intervalle_pour_graphe(intervalle)
    xs = np.linspace(xmin, xmax, 1600)

    fig, ax = plt.subplots(figsize=(18, 5))

    y_f = _eval_vecteur(x, f, xs)
    ax.plot(xs, y_f, color="black", linewidth=3.2, label=r"$f(x)$")

    ordre_max = int(ordre_max)
    cmap = plt.get_cmap("tab20")
    for m in range(1, ordre_max + 1):
        try:
            tronque = sp.series(f, x, 0, m + 1).removeO()
            y_t = _eval_vecteur(x, tronque, xs)
            ax.plot(xs, y_t, color=cmap((m - 1) % 20), linewidth=1.6, alpha=0.98, label=rf"$T_{{{m}}}(x)$")
        except Exception:
            pass

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, alpha=0.25)
    st.pyplot(fig)


# =============================================================
# 11) Flask app (compatible app.js + index.html)
# =============================================================
app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/compute", methods=["POST"])
def compute():
    data = request.get_json(force=True) or {}
    out = []
    st.set_out(out)

    type_input = (data.get("type_input", "") or "").strip()
    user_input1 = (data.get("user_input1") or "").strip()
    user_input2 = (data.get("user_input2") or "").strip()

    try:
        # IMPORTANT : n doit pouvoir valoir 0 si la série commence à 0
        n = sp.symbols("n", integer=True, nonnegative=True)

        locals_dict = {
            "n": n,
            "pi": sp.pi,
            "e": sp.E,
            "E": sp.E,
            "I": sp.I,
            "sqrt": sp.sqrt,
            "abs": sp.Abs,
            "Abs": sp.Abs,
            "sign": sp.sign,
            "exp": sp.exp,
            "log": sp.log,
            "ln": sp.log,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "cot": sp.cot,
            "sec": sp.sec,
            "csc": sp.csc,
            "arcsin": sp.asin,
            "arccos": sp.acos,
            "arctan": sp.atan,
            "arccot": sp.acot,
            "arcsec": sp.asec,
            "arccsc": sp.acsc,
            "sinh": sp.sinh,
            "cosh": sp.cosh,
            "tanh": sp.tanh,
            "coth": sp.acoth if hasattr(sp, "acoth") else None,
            "sech": sp.sech if hasattr(sp, "sech") else None,
            "csch": sp.csch if hasattr(sp, "csch") else None,
            "arsinh": sp.asinh,
            "arcosh": sp.acosh,
            "artanh": sp.atanh,
            "arcoth": sp.acoth if hasattr(sp, "acoth") else None,
            "arsech": sp.asech if hasattr(sp, "asech") else None,
            "arcsch": sp.acsch if hasattr(sp, "acsch") else None,
            "erf": sp.erf,
            "erfc": sp.erfc,
            "gamma": sp.gamma,
            "lgamma": sp.loggamma,
            "digamma": sp.digamma,
            "polygamma": sp.polygamma,
            "besselj": sp.besselj,
            "bessely": sp.bessely,
            "besseli": sp.besseli,
            "besselk": sp.besselk,
            "airyai": sp.airyai,
            "airybi": sp.airybi,
            "Heaviside": sp.Heaviside,
            "DiracDelta": sp.DiracDelta,
            "re": sp.re,
            "im": sp.im,
            "conjugate": sp.conjugate,
        }

        # Nettoyage des entrées None ajoutées conditionnellement
        locals_dict = {k: v for k, v in locals_dict.items() if v is not None}

        if type_input in ("Suites de fonctions", "Séries de fonctions", "Série entière", "Développement en série entière"):
            x = sp.symbols("x", real=True)
            locals_dict["x"] = x
            intervalle = _parse_interval_payload(data)

            if intervalle == S.EmptySet:
                st.latex(_latex_rouge("Impossible : l’intervalle choisi est vide."))
                return jsonify({"messages": out})

            if not user_input1:
                st.latex(_latex_rouge("Veuillez entrer une expression."))
                return jsonify({"messages": out})

            expr = sp.simplify(sp.sympify(user_input1, locals=locals_dict))

            if type_input == "Série entière":
                # app.js utilise user_input2 pour l'indice minimal
                n_min_raw = (data.get("n_min") or "").strip() or user_input2 or "0"
                try:
                    n_min = int(n_min_raw)
                except Exception:
                    st.latex(_latex_rouge("L’indice de départ doit être un entier."))
                    return jsonify({"messages": out})

                etude_serie_entiere_convergence(expr, n, x, intervalle, n_min=n_min, N_graphe=20)
                return jsonify({"messages": out})

            if type_input == "Développement en série entière":
                ordre = 10
                try:
                    if user_input2:
                        ordre = int(user_input2)
                except Exception:
                    ordre = 10

                f, ser, R = developpement_en_serie_entiere(expr, x, ordre=ordre)
                if f is not None:
                    if R is not None:
                        if R == sp.oo:
                            st.latex(r"\text{Intervalle de convergence (réel) : }\mathbb{R}.")
                        else:
                            st.latex(r"\text{Intervalle de convergence (réel) : }(-" + sp.latex(R) + r"," + sp.latex(R) + r").")
                    st.latex(r"\text{Représentation : la fonction et ses polynômes de Taylor.}")
                    _trace_fonction_et_tronques(f, x, intervalle, ordre_max=min(10, max(1, ordre)))
                return jsonify({"messages": out})

            if type_input == "Suites de fonctions":
                etude_suite_de_fonctions(expr, n, x, intervalle)
                return jsonify({"messages": out})

            if type_input == "Séries de fonctions":
                n_min_raw = (data.get("n_min") or "").strip() or user_input2
                if not n_min_raw:
                    st.latex(_latex_rouge("Veuillez renseigner l’indice de départ de la série."))
                    return jsonify({"messages": out})
                try:
                    n_min = int(n_min_raw)
                except Exception:
                    st.latex(_latex_rouge("L’indice de départ doit être un entier."))
                    return jsonify({"messages": out})
                etude_serie_de_fonctions(expr, n, x, intervalle, n_min)
                return jsonify({"messages": out})

        # Cas Suite/Série numériques
        if not user_input1:
            st.latex(_latex_rouge("Veuillez entrer une expression correcte."))
            return jsonify({"messages": out})

        u = sp.sympify(user_input1, locals=locals_dict)
        u_n = sp.simplify(u)

        if type_input == "Série":
            if not user_input2:
                st.latex(_latex_rouge("Veuillez renseigner l'indice de départ."))
                return jsonify({"messages": out})

            n_min = int(user_input2)
            kk = sp.symbols("k", integer=True)
            S_n = sp.Sum(u_n.subs(n, kk), (kk, n_min, n))

            st.latex(r"\text{Vous avez entré la série de somme partielle :}")
            st.latex(sp.latex(S_n))
            st.latex(r"\text{Son terme général est : } u_n=" + sp.latex(u_n))

            try:
                critere(u_n, n, n_min)  # 
            except Exception:
                pass

            try:
                graphe(S_n, n, "blue") 
                graphe(u_n, n, "red")   
            except Exception:
                pass

            return jsonify({"messages": out})

        # Suite numérique
        st.latex(r"\text{Vous avez entré la suite définie par }u_n=" + sp.latex(u_n))
        lim = sp.limit_seq(u_n, n)
        st.latex(r"\lim_{n\to\infty} " + sp.latex(u_n) + "=" + sp.latex(lim))

        if getattr(lim, "is_real", False):
            st.latex(r"\text{La suite admet une limite réelle : convergence.}")
        else:
            st.latex(r"\text{La suite ne possède pas de limite réelle : divergence.}")

        try:
            graphe(u_n, n, "blue")  # noqa: F821
        except Exception:
            pass

        return jsonify({"messages": out})

    except Exception as e:
        st.latex(_latex_rouge("Entrée invalide. Veuillez entrer une expression correcte."))
        st.latex(r"\text{Détail technique : " + str(e).replace("_", r"\_") + r"}")
        return jsonify({"messages": out})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)