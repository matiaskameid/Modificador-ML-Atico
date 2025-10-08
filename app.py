# app.py
import json
import math
import os
import unicodedata
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

<<<<<<< HEAD
# ——— añadir cerca de los imports ———
import subprocess, os
from datetime import datetime

import subprocess, os
from datetime import datetime

def git_sync(paths, message):
    g = st.secrets.get("git")
    if not g:
        st.error("No hay configuración [git] en secrets.toml")
        return False

    repo_url = g["repo_url"]
    branch   = g.get("branch", "main")
    token    = g.get("token")
    user     = g.get("user_name", "ML Bot")  # default
    email    = g.get("user_email", "ml-bot@example.com")  # default

    # URL con token para push (no imprime token)
    push_url = repo_url
    if token and repo_url.startswith("https://"):
        push_url = repo_url.replace("https://", f"https://{token}@")

    def run(cmd):
        r = subprocess.run(cmd, capture_output=True, text=True, shell=False)
        if r.returncode != 0:
            raise RuntimeError(r.stderr or r.stdout)

    try:
        # 1) Marcar dir como seguro (Git en contenedores a veces lo exige)
        try:
            run(["git","config","--global","--add","safe.directory", os.getcwd()])
        except Exception:
            pass

        # 2) Identidad para commits (por si no hay global)
        run(["git","config","user.name", user])
        run(["git","config","user.email", email])

        # 3) Asegurar rama de trabajo (evitar detached HEAD)
        try:
            run(["git","rev-parse","--verify", branch])
            run(["git","checkout", branch])
        except Exception:
            run(["git","checkout","-b", branch])

        # 4) Asegurar remoto “origin” (no es obligatorio, pero ayuda con pull)
        try:
            run(["git","remote","get-url","origin"])
        except Exception:
            # si no existe, lo creamos sin token (pull por origin puede fallar y lo ignoramos)
            run(["git","remote","add","origin", repo_url])

        # 5) Añadir archivos (forzar por si estuvieron ignorados)
        run(["git","add","-f", *paths])

        # 6) Commit (si no hay cambios, tratamos como OK y aún intentamos push)
        committed = True
        try:
            run(["git","commit","-m", message])
        except Exception:
            committed = False  # nothing to commit

        # 7) Pull rebase ‘best-effort’
        try:
            run(["git","pull","--rebase","origin", branch])
        except Exception:
            pass  # ok si repo inicial o sin permisos de lectura por origin

        # 8) Push usando la URL con token (independiente de “origin”)
        run(["git","push", push_url, f"HEAD:{branch}"])
        return True
    except Exception as e:
        st.error(f"Sync a GitHub falló: {e}")
        return False




=======
>>>>>>> 254480d (feat: cacheos y mejoras de fluidez en app.py)
from ml_client import MercadoLibreClient

st.set_page_config(page_title="ML · Márgenes y Precios", page_icon="🛒", layout="wide")

# =========================
# Config & Cliente
# =========================
conf = st.secrets["ml"]
app_conf = st.secrets.get("app", {})

CLIENT_ID = conf["client_id"]
CLIENT_SECRET = conf["client_secret"]
REFRESH_TOKEN = conf["refresh_token"]
SITE = conf.get("site", "MLC")
SELLER_ID = conf.get("seller_id")  # opcional

RPM_CEILING = int(app_conf.get("rpm_ceiling", 1200))
BULK_SIZE = int(app_conf.get("bulk_size", 20))            # ¡Debe ser 20!
BULK_WORKERS = int(app_conf.get("bulk_workers", 64))      # paralelismo
SNAPSHOT_MAX_AGE_DAYS = int(app_conf.get("snapshot_max_age_days", 7))
SNAPSHOT_PATH = app_conf.get("snapshot_path", "sku_index.json")
FEES_RULES_PATH = app_conf.get("fees_rules_path", "fees_rules.json")

# IVA Chile
VAT_RATE = 0.19
IVA_MULT = 1.0 + VAT_RATE

ml = MercadoLibreClient(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    refresh_token=REFRESH_TOKEN,
    site=SITE,
    rpm_ceiling=RPM_CEILING,
    bulk_size=BULK_SIZE,
)

# =========================
# Utilidades
# =========================
def normalize_sku(raw: Any) -> str:
    s = unicodedata.normalize("NFKC", str(raw))
    s = s.replace("\u200b", "").replace("\u00A0", " ")
    s = s.strip()
    s = s.replace("—", "-").replace("–", "-")
    return s.upper()

def strip_accents_lower(text: Any) -> str:
    s = str(text) if text is not None else ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()

def ceil_to_10(x: float) -> int:
    return int(math.ceil(float(x) / 10.0) * 10.0)

def now_utc_iso() -> str:
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

def fmt_money(v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    return f"${int(v):,}"

def fmt_pct(v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    return f"{float(v):.1f}%"

# =========================
<<<<<<< HEAD
=======
# Cache helpers (pesado)
# =========================
@st.cache_data(show_spinner=False)
def read_excel_cached(file_bytes: bytes) -> pd.DataFrame:
    # Lee Excel desde bytes para que cambie al subir un archivo distinto
    from io import BytesIO
    return pd.read_excel(BytesIO(file_bytes), dtype={"CODIGO SKU": str})

@st.cache_data(show_spinner=False)
def fetch_items_info_cached(ids_tuple: Tuple[str, ...], workers: int) -> Dict[str, Dict[str, Any]]:
    """Consulta /items?ids=... en paralelo y cachea por conjunto ordenado de IDs + #workers."""
    if not ids_tuple:
        return {}
    bulk = ml.get_items_bulk_parallel(list(ids_tuple), workers=workers)
    out: Dict[str, Dict[str, Any]] = {}
    for entry in bulk:
        body = entry.get("body") or {}
        iid = body.get("id")
        if iid:
            out[iid] = {"title": body.get("title"), "price": body.get("price")}
    return out

@st.cache_data(show_spinner=False)
def build_no_ml_excel_bytes(rows: List[Tuple[str, str]]) -> bytes:
    """Genera el Excel de 'no encontrados' y lo cachea por contenido (lista de filas)."""
    import io
    buf = io.BytesIO()
    df = pd.DataFrame(rows, columns=["SKU", "PRODUCTO"])
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        df.to_excel(w, index=False, sheet_name="No_en_ML")
    buf.seek(0)
    return buf.getvalue()

# =========================
>>>>>>> 254480d (feat: cacheos y mejoras de fluidez en app.py)
# Snapshot SKU→ID
# =========================
def load_snapshot_from_disk(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw_map: Dict[str, str] = data.get("map", {})
        normalized_map: Dict[str, str] = {}
        conflicts: List[Tuple[str, str, str]] = []
        for k, v in raw_map.items():
            nk = normalize_sku(k)
            if nk in normalized_map and normalized_map[nk] != v:
                conflicts.append((nk, normalized_map[nk], v))
            normalized_map[nk] = v
        data["map"] = normalized_map
        if conflicts:
            data.setdefault("meta", {})["conflicts"] = conflicts
        return data
    except Exception as e:
        st.warning(f"No se pudo leer el snapshot: {e}")
        return None

def save_snapshot_to_disk(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def snapshot_is_fresh(generated_at_iso: str, max_age_days: int) -> bool:
    try:
        dt = datetime.fromisoformat(generated_at_iso.replace("Z", "+00:00"))
        return datetime.now(timezone.utc) - dt <= timedelta(days=max_age_days)
    except Exception:
        return False

def rebuild_snapshot_from_ml(progress_cb=None) -> Dict[str, Any]:
    sid = SELLER_ID
    if not sid:
        me = ml.get_me()
        sid = str(me.get("id"))
    raw_index = ml.build_sku_index(seller_id=sid, progress=progress_cb)
    normalized_map: Dict[str, str] = {}
    conflicts: List[Tuple[str, str, str]] = []
    for k, v in raw_index.items():
        nk = normalize_sku(k)
        if nk in normalized_map and normalized_map[nk] != v:
            conflicts.append((nk, normalized_map[nk], v))
        normalized_map[nk] = v
    payload = {
        "schema_version": 1,
        "site": SITE,
        "seller_id": sid,
        "generated_at": now_utc_iso(),
        "map": normalized_map,
        "meta": {"total_items": len(normalized_map), "conflicts": conflicts},
    }
    return payload

# =========================
# Fees por tramos (persistencia + cálculo)
# =========================
DEFAULT_FEE_RULES = {
    "schema_version": 1,
    "updated_at": None,
    "rules": [
        {"min": 0, "max": 9900, "percent": 17.0, "fixed": 588},     # fijo NETO
        {"min": 9900, "max": 19990, "percent": 17.0, "fixed": 840}, # fijo NETO
        {"min": 19990, "max": None, "percent": 17.0, "fixed": 2227} # 2650 BRUTO / 1.19
    ],
}

def load_fees_rules(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rules = []
            for r in data.get("rules", []):
                rules.append({
                    "min": int(r["min"]) if r["min"] is not None else 0,
                    "max": int(r["max"]) if r.get("max") is not None else None,
                    "percent": float(r["percent"]),
                    "fixed": int(r["fixed"]),
                })
            data["rules"] = sorted(rules, key=lambda x: (x["min"], 10**9 if x["max"] is None else x["max"]))
            return data
        except Exception as e:
            st.warning(f"No se pudo leer fees_rules.json, usando defaults. Detalle: {e}")
    return DEFAULT_FEE_RULES.copy()

def save_fees_rules(path: str, payload: Dict[str, Any]) -> None:
    payload = dict(payload)
    payload["updated_at"] = now_utc_iso()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def validate_fee_rules(rules: List[Dict[str, Any]]) -> Tuple[bool, str]:
    if not rules:
        return False, "Debe existir al menos un tramo."
    rules_sorted = sorted(rules, key=lambda r: (r["min"], 10**9 if r["max"] is None else r["max"]))
    for i, r in enumerate(rules_sorted):
        mn = r["min"]; mx = r["max"]
        if mn < 0:
            return False, f"El tramo #{i+1} tiene min < 0."
        if mx is not None and mx <= mn:
            return False, f"El tramo #{i+1} tiene max <= min."
        if r["percent"] >= 100 or r["percent"] < 0:
            return False, f"El tramo #{i+1} tiene % inválido."
        if r["fixed"] < 0:
            return False, f"El tramo #{i+1} tiene fijo negativo."
        if i > 0:
            prev = rules_sorted[i-1]
            prev_max = prev["max"]
            if prev_max is None or prev_max > mn:
                return False, f"Solapamiento entre tramos #{i} y #{i+1}."
    if rules_sorted[-1]["max"] is not None:
        return False, "El último tramo debe ser abierto (max vacío)."
    return True, "OK"

def find_rule_for_gross_price(pvp_bruto: float, rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    p = float(pvp_bruto)
    for r in rules:
        mn = r["min"]; mx = r["max"]
        if (p >= mn) and (mx is None or p < mx):
            return r
    return rules[-1]

def fee_neto_desde_pvp(pvp_bruto: float, rules: List[Dict[str, Any]]) -> int:
    r = find_rule_for_gross_price(pvp_bruto, rules)
    pn = float(pvp_bruto) / IVA_MULT
    perc = float(r["percent"]) / 100.0
    fijo_neto = int(r["fixed"])
    return int(round(fijo_neto + perc * pn))

def margen_pct_sobre_neto(pvp_bruto: float, costo_neto: float, rules: List[Dict[str, Any]]) -> float:
    c = float(costo_neto)
    if c <= 0:
        return float("inf")
    pn = float(pvp_bruto) / IVA_MULT
    fee = fee_neto_desde_pvp(pvp_bruto, rules)
    ingreso_neto = pn - fee
    return (ingreso_neto - c) / pn * 100.0

def solve_pvp_para_margen_neto(costo_neto: float, target_pct: float, rules: List[Dict[str, Any]]) -> int:
    c = float(costo_neto)
    targ = float(target_pct) / 100.0
    candidate_pvp = None
    for r in rules:
        perc = float(r["percent"]) / 100.0
        fijo = int(r["fixed"])
        denom = (1.0 - perc) - targ
        if denom <= 0:
            continue
        pnet_req = (c + fijo) / denom
        pvp_req = ceil_to_10(pnet_req * IVA_MULT)
        mn, mx = r["min"], r["max"]
        if pvp_req < mn:
            pvp_req = ceil_to_10(mn)
        if mx is None or pvp_req < mx:
            candidate_pvp = pvp_req
            break
    if candidate_pvp is None:
        last = rules[-1]
        perc = float(last["percent"]) / 100.0
        fijo = int(last["fixed"])
        denom = (1.0 - perc) - targ
        if denom <= 0:
            pnet_req = c + fijo + 1.0
        else:
            pnet_req = (c + fijo) / denom
        candidate_pvp = ceil_to_10(pnet_req * IVA_MULT)
    pvp = int(candidate_pvp)
    for _ in range(2000):
        if margen_pct_sobre_neto(pvp, c, rules) + 1e-9 >= float(target_pct):
            return pvp
        pvp += 10
    return pvp

def option_A_mismo_tramo(pvp_actual: Optional[float], rules: List[Dict[str, Any]]) -> Optional[int]:
    if pvp_actual is None or (isinstance(pvp_actual, float) and math.isnan(pvp_actual)):
        return None
    r = find_rule_for_gross_price(pvp_actual, rules)
    if r["max"] is None:
        return None
    mx = int(r["max"])
    return int(((mx - 1) // 10) * 10)

# =========================
# Estado inicial
# =========================
if "snapshot" not in st.session_state:
    snap = load_snapshot_from_disk(SNAPSHOT_PATH)
    if snap:
        st.session_state.snapshot = snap
    else:
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        def cb(msg: str, pct: float) -> None:
            status_placeholder.info(msg)
            progress_bar.progress(min(max(pct, 0.0), 1.0))
        try:
            status_placeholder.info("No hay snapshot. Construyendo índice desde ML…")
            snap = rebuild_snapshot_from_ml(progress_cb=cb)
            save_snapshot_to_disk(SNAPSHOT_PATH, snap)
            st.session_state.snapshot = snap
            progress_bar.progress(1.0)
            status_placeholder.success(
                f"Snapshot creado ({snap['meta']['total_items']} SKUs). Guardado en {SNAPSHOT_PATH}."
            )
        except Exception as e:
            progress_bar.empty()
            status_placeholder.error("Error al construir snapshot.")
            st.exception(e)
            st.stop()

if "fees_rules" not in st.session_state:
    st.session_state.fees_rules = load_fees_rules(FEES_RULES_PATH)

# selección global persistente
if "mass_selected_ids" not in st.session_state:
    st.session_state.mass_selected_ids = set()
if "sel_version" not in st.session_state:
    st.session_state.sel_version = 0
if "confirm_update" not in st.session_state:
    st.session_state.confirm_update = None
if "bulk_plan" not in st.session_state:
    st.session_state.bulk_plan = None

# overrides locales de precios unitarios (para refresco inmediato en buscador)
if "unitary_override_prices" not in st.session_state:
    st.session_state.unitary_override_prices = {}

# ÉXITO unitario (payload) para mostrar abajo en el buscador
if "unitary_success_payload" not in st.session_state:
    st.session_state.unitary_success_payload = None

# Resets pendientes de inputs C (Opción C) — se procesan ANTES de crear widgets
if "pending_resets" not in st.session_state:
    st.session_state.pending_resets = set()

SNAPSHOT = st.session_state["snapshot"]
SKU_MAP: Dict[str, str] = SNAPSHOT.get("map", {})
SNAP_META = SNAPSHOT.get("meta", {}) or {}
FEE_DOC = st.session_state["fees_rules"]
FEE_RULES = FEE_DOC["rules"]

# =========================
# UI (tabs)
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["🔎 Buscar (SKU/Título)", "📊 Excel · Márgenes", "🗂️ Snapshot", "💸 Fees / Tramos"])

# ---------- TAB 4: Fees ----------
with tab4:
    st.subheader("Reglas de fees por tramos (aplican a TODO el catálogo)")
    st.caption("• Rango en PVP (BRUTO) tipo [min, max).  • % se aplica sobre PRECIO NETO (PVP/1,19).  • Fijo se ingresa en NETO (CLP).  • Último tramo abierto.")
    rules_df = pd.DataFrame(FEE_RULES)
    rules_df_display = st.data_editor(
        rules_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "min": st.column_config.NumberColumn("Min PVP bruto (incl)", step=10),
            "max": st.column_config.NumberColumn("Max PVP bruto (excl, vacío=abierto)", step=10),
            "percent": st.column_config.NumberColumn("% ML (sobre neto)", step=0.1),
            "fixed": st.column_config.NumberColumn("Fijo (NETO, CLP)", step=10),
        },
        hide_index=True,
        key="fees_editor",
    )

    def _coerce_max(v):
        if v is None:
            return None
        if isinstance(v, str) and v.strip().lower() in ("none", "", "nan"):
            return None
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass
        return int(v)

    def _coerce_rules(df_display: pd.DataFrame) -> list[dict]:
        raw = df_display.to_dict(orient="records")
        out = []
        for r in raw:
            out.append({
                "min": int(r["min"]) if r.get("min") is not None else 0,
                "max": _coerce_max(r.get("max")),
                "percent": float(r["percent"]),
                "fixed": int(r["fixed"]),
            })
        return out

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Validar reglas"):
            rules_list = _coerce_rules(rules_df_display)
            ok, msg = validate_fee_rules(rules_list)
            st.success("Reglas válidas." if ok else msg)
    with c2:
        if st.button("Guardar reglas", type="primary"):
            rules_list = _coerce_rules(rules_df_display)
            ok, msg = validate_fee_rules(rules_list)
            if not ok:
                st.error(msg)
            else:
                doc = {"schema_version": 1, "updated_at": None, "rules": rules_list}
                save_fees_rules(FEES_RULES_PATH, doc)
                st.session_state.fees_rules = load_fees_rules(FEES_RULES_PATH)
<<<<<<< HEAD
                ok = git_sync([FEES_RULES_PATH], "chore(fees): update")
                st.success("Fees guardados y subidos a GitHub.") if ok else st.error("No se pudo subir a GitHub.")

=======
                st.success(f"Guardado en {FEES_RULES_PATH}.")
>>>>>>> 254480d (feat: cacheos y mejoras de fluidez en app.py)
    with c3:
        rules_for_download = _coerce_rules(rules_df_display)
        st.download_button(
            "Descargar reglas",
            data=json.dumps({"schema_version": 1, "rules": rules_for_download}, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="fees_rules.json",
            mime="application/json",
            use_container_width=True,
        )
    st.caption(f"Última actualización: {FEE_DOC.get('updated_at') or '—'}")

# ---------- TAB 3: Snapshot ----------
with tab3:
    st.subheader("Snapshot SKU → Item ID")
    cols = st.columns(4)
    with cols[0]:
        st.metric("Total SKUs", f"{SNAP_META.get('total_items', len(SKU_MAP)):,}")
    with cols[1]:
        gen_at = SNAPSHOT.get("generated_at", "—")
        st.metric("Generado", gen_at.replace("T", " ").split("+")[0])
    with cols[2]:
        fresh = snapshot_is_fresh(SNAPSHOT.get("generated_at", ""), SNAPSHOT_MAX_AGE_DAYS)
        st.metric("Freshness", "OK" if fresh else "Antiguo")
    with cols[3]:
        st.caption(f"bulk={BULK_SIZE} · workers={BULK_WORKERS} · rpm≈{RPM_CEILING}")

    st.caption(f"Archivo: {SNAPSHOT_PATH} · Máx. antigüedad recomendada: {SNAPSHOT_MAX_AGE_DAYS} días · Site: {SNAPSHOT.get('site','?')} · Seller: {SNAPSHOT.get('seller_id','?')}")
    st.download_button("Descargar snapshot actual", data=json.dumps(SNAPSHOT, ensure_ascii=False, indent=2).encode("utf-8"), file_name="sku_index.json", mime="application/json", use_container_width=True)

    up = st.file_uploader("Cargar snapshot (JSON)", type=["json"])
    if up:
        try:
            loaded = json.load(up)
            raw_map: Dict[str, str] = loaded.get("map", {})
            normalized_map: Dict[str, str] = {}
            conflicts: List[Tuple[str, str, str]] = []
            for k, v in raw_map.items():
                nk = normalize_sku(k)
                if nk in normalized_map and normalized_map[nk] != v:
                    conflicts.append((nk, normalized_map[nk], v))
                normalized_map[nk] = v
            loaded["map"] = normalized_map
            loaded.setdefault("meta", {})["conflicts"] = conflicts
            st.session_state.snapshot = loaded
            st.success(f"Snapshot cargado ({len(normalized_map)} SKUs).")
            save_snapshot_to_disk(SNAPSHOT_PATH, loaded)
        except Exception as e:
            st.error(f"No se pudo cargar el JSON: {e}")

    st.divider()
    status_placeholder = st.empty()
    progress_bar = st.progress(0)

    if st.button("Reconstruir desde ML y guardar snapshot", type="primary", use_container_width=True):
        def cb2(msg: str, pct: float) -> None:
            status_placeholder.info(msg)
            progress_bar.progress(min(max(pct, 0.0), 1.0))
        try:
            snap2 = rebuild_snapshot_from_ml(progress_cb=cb2)
            save_snapshot_to_disk(SNAPSHOT_PATH, snap2)
            st.session_state.snapshot = snap2
<<<<<<< HEAD
            ok = git_sync([SNAPSHOT_PATH], "chore(snapshot): update")
            st.success("Snapshot guardado y subido a GitHub.") if ok else st.error("No se pudo subir a GitHub.")
=======
            status_placeholder.success(f"Snapshot reconstruido ({snap2['meta']['total_items']} SKUs). Guardado en {SNAPSHOT_PATH}.")
>>>>>>> 254480d (feat: cacheos y mejoras de fluidez en app.py)
            progress_bar.progress(1.0)
        except Exception as e:
            progress_bar.empty()
            status_placeholder.error("Error al reconstruir snapshot.")
            st.exception(e)

# ---------- TAB 2: Excel · Márgenes (masivo)
with tab2:
    st.subheader("Analizar Excel y sugerir precio objetivo (igual a tu Excel)")

    pct = st.slider("Margen objetivo (%) sobre PRECIO NETO", min_value=1, max_value=80, value=30, step=1, key="pct_mass")
    target_pct = float(pct)

    up_xlsx = st.file_uploader(
        "Sube Excel con columnas: CODIGO SKU, PRODUCTO, COSTO PROMEDIO ACTUAL, NUEVO COSTO PROMEDIO",
        type=["xlsx", "xls"],
        key="uploader_mass",
    )

    show_only_fail = st.checkbox("Mostrar solo 'No cumple (margen actual sobre neto)'", value=True, key="show_fail_mass")
    page_size = st.selectbox("Filas por página", options=[50, 100, 200], index=1, key="pagesize_mass")
    sku_filter = st.text_input("Filtrar por SKU (contiene)", value="", key="skufilter_mass")

    if up_xlsx is not None:
<<<<<<< HEAD
        try:
            df_raw = pd.read_excel(up_xlsx, dtype={"CODIGO SKU": str})
=======
        # ---------- lectura Excel (cacheada por bytes) ----------
        try:
            file_bytes = up_xlsx.getvalue()
            df_raw = read_excel_cached(file_bytes)
>>>>>>> 254480d (feat: cacheos y mejoras de fluidez en app.py)
        except Exception as e:
            st.error("No se pudo leer el Excel. Verifica el formato y nombres de columnas.")
            st.exception(e)
            st.stop()

        required_cols = {"CODIGO SKU", "PRODUCTO", "COSTO PROMEDIO ACTUAL", "NUEVO COSTO PROMEDIO"}
        missing = required_cols - set(map(str, df_raw.columns))
        if missing:
            st.error(f"Faltan columnas requeridas: {', '.join(missing)}")
            st.stop()

        df = df_raw.copy()
        df["__SKU_NORM__"] = df["CODIGO SKU"].apply(normalize_sku)
        df["title_norm"] = df["PRODUCTO"].apply(strip_accents_lower)

        def costo_efectivo(row):
            try:
                nuevo = float(row["NUEVO COSTO PROMEDIO"])
            except Exception:
                nuevo = 0.0
            try:
                actual = float(row["COSTO PROMEDIO ACTUAL"])
            except Exception:
                actual = 0.0
            return nuevo if nuevo > 0 else actual

        df["costo"] = df.apply(costo_efectivo, axis=1)
        df["item_id"] = df["__SKU_NORM__"].map(SKU_MAP)

        st.session_state["excel_global"] = df[["CODIGO SKU", "PRODUCTO", "__SKU_NORM__", "title_norm", "costo", "item_id"]].copy()

        work = df[df["item_id"].notna()].copy()
        item_ids = work["item_id"].tolist()
<<<<<<< HEAD

        st.caption(f"Consultando precios en ML: {len(set(item_ids)):,} ítems · lotes de {BULK_SIZE} · {BULK_WORKERS} workers")

        def fetch_items_info(iids: List[str]) -> Dict[str, Dict[str, Any]]:
            info: Dict[str, Dict[str, Any]] = {}
            if not iids:
                return info
            seen = set(); uniq = []
            for iid in iids:
                if iid not in seen:
                    seen.add(iid); uniq.append(iid)
            bulk = ml.get_items_bulk_parallel(uniq, workers=BULK_WORKERS)
            for entry in bulk:
                body = entry.get("body") or {}
                iid = body.get("id")
                if iid:
                    info[iid] = {"title": body.get("title"), "price": body.get("price")}
            return info

        info_map = fetch_items_info(item_ids)
=======
        uniq_ids = tuple(sorted(set(item_ids)))

        st.caption(f"Consultando precios en ML (cacheado por conjunto de IDs): {len(uniq_ids):,} ítems · lotes de {BULK_SIZE} · {BULK_WORKERS} workers")

        # ---------- consulta ML (cacheada por tupla de IDs) ----------
        info_map = fetch_items_info_cached(uniq_ids, workers=BULK_WORKERS)

>>>>>>> 254480d (feat: cacheos y mejoras de fluidez en app.py)
        work["price_actual"] = work["item_id"].map(lambda x: info_map.get(x, {}).get("price"))
        work["titulo"] = work["item_id"].map(lambda x: info_map.get(x, {}).get("title", ""))

        def compute_row(row):
            c = float(row["costo"])
            pa = row["price_actual"]
            pa_val = float(pa) if pd.notna(pa) else None
            if pa_val is None or c <= 0:
                m_curr = None
                cumple_actual = True
            else:
                m_curr = margen_pct_sobre_neto(pa_val, c, FEE_RULES)
                cumple_actual = m_curr >= (target_pct - 1e-9)
            pB = solve_pvp_para_margen_neto(costo_neto=c, target_pct=target_pct, rules=FEE_RULES)
            mB = margen_pct_sobre_neto(pB, c, FEE_RULES)
            pA = option_A_mismo_tramo(pa_val, FEE_RULES)
            mA = None if pA is None else margen_pct_sobre_neto(pA, c, FEE_RULES)
            return pd.Series({
                "margen_actual_pct": m_curr, "cumple_actual": cumple_actual,
                "precio_A_mismo_tramo": pA, "margen_A_pct": mA,
                "precio_B_objetivo": pB, "margen_B_pct": mB,
            })

        calc = work.apply(compute_row, axis=1)
        work = pd.concat([work, calc], axis=1)

        total_excel = len(df)
        matched = len(work)
        pct_matched = (matched / total_excel * 100.0) if total_excel else 0.0
        not_found = int(df["item_id"].isna().sum())
        pct_not_found = (not_found / total_excel * 100.0) if total_excel else 0.0
        no_cumple_actual = int((~work["cumple_actual"]).sum())
        pct_no_cumple = (no_cumple_actual / matched * 100.0) if matched else 0.0

        kp1, kp2, kp3, kp4 = st.columns(4)
        kp1.metric("Total en Excel", f"{total_excel:,}")
        kp2.metric("Encontrados en ML", f"{matched:,}", f"{pct_matched:.1f}%")
        kp3.metric("No están en ML", f"{not_found:,}", f"{pct_not_found:.1f}%")
        kp4.metric("No cumplen (del total de encontrados)", f"{no_cumple_actual:,}", f"{pct_no_cumple:.1f}%")

        show_df = work if not show_only_fail else work[~work["cumple_actual"]].copy()
        if sku_filter.strip():
            q = normalize_sku(sku_filter).replace(" ", "")
            show_df = show_df[show_df["__SKU_NORM__"].str.replace(" ", "").str.contains(q, na=False)]

        def brecha_B(row):
            try:
                return int(row["precio_B_objetivo"]) - int(row["price_actual"])
            except Exception:
                return 0
        show_df["brecha_B"] = show_df.apply(brecha_B, axis=1)
        show_df = show_df.sort_values(by=["brecha_B"], ascending=False, na_position="last")

        cols_view = [
            "CODIGO SKU","PRODUCTO","costo","price_actual","margen_actual_pct",
            "precio_A_mismo_tramo","margen_A_pct","precio_B_objetivo","margen_B_pct","item_id",
        ]
        show_df = show_df[cols_view].copy()
        show_df.rename(columns={
            "CODIGO SKU":"SKU","PRODUCTO":"Título (Excel)","costo":"Costo (NETO)",
            "price_actual":"PVP actual (BRUTO)","margen_actual_pct":"Margen actual (%)",
            "precio_A_mismo_tramo":"Opción A (mismo tramo)","margen_A_pct":"Margen A (%)",
            "precio_B_objetivo":"Opción B (objetivo)","margen_B_pct":"Margen B (%)",
        }, inplace=True)

        total_rows = len(show_df); page_size = int(page_size)
        total_pages = max(1, math.ceil(total_rows / page_size))
        page = st.number_input("Página", min_value=1, max_value=total_pages, value=1, step=1, key="page_mass")
        start = (page - 1) * page_size; end = start + page_size
        page_df = show_df.iloc[start:end].copy()

        st.caption(f"Mostrando {len(page_df)} de {total_rows} filas · Página {page}/{total_pages}")

        # ===== Tabla principal con selección persistente =====
        sel_ids = st.session_state.mass_selected_ids
        precheck = page_df["item_id"].apply(lambda iid: iid in sel_ids).tolist()

        view_df = page_df[[
            "SKU","Título (Excel)","Costo (NETO)","PVP actual (BRUTO)",
            "Margen actual (%)","Opción A (mismo tramo)","Margen A (%)",
            "Opción B (objetivo)","Margen B (%)"
        ]].copy()
        view_df.insert(0, "✓", precheck)

        editor_key = f"mass_view_editor_{page}_{st.session_state.sel_version}"
        view_ed = st.data_editor(
            view_df,
            hide_index=True,
            use_container_width=True,
            key=editor_key,
            column_config={
                "✓": st.column_config.CheckboxColumn("✓", help="Seleccionar para masivo (persiste entre páginas)"),
                "Costo (NETO)": st.column_config.NumberColumn("Costo (NETO)", format="$%d", step=10),
                "PVP actual (BRUTO)": st.column_config.NumberColumn("PVP actual (BRUTO)", format="$%d", step=10),
                "Margen actual (%)": st.column_config.NumberColumn("Margen actual (%)", format="%.1f"),
                "Opción A (mismo tramo)": st.column_config.NumberColumn("Opción A (mismo tramo)", format="$%d", step=10),
                "Margen A (%)": st.column_config.NumberColumn("Margen A (%)", format="%.1f"),
                "Opción B (objetivo)": st.column_config.NumberColumn("Opción B (objetivo)", format="$%d", step=10),
                "Margen B (%)": st.column_config.NumberColumn("Margen B (%)", format="%.1f"),
            },
        )

        # Actualizar selección global con lo editado en esta página
        page_ids = set(page_df["item_id"].tolist())
        st.session_state.mass_selected_ids -= page_ids
        selected_idx = [i for i, checked in zip(page_df.index.tolist(), view_ed["✓"].tolist()) if checked]
        selected_ids_page = set(page_df.loc[selected_idx, "item_id"].tolist())
        st.session_state.mass_selected_ids |= selected_ids_page

        # Limpiar selección global
        c_clear, _ = st.columns([0.2, 0.8])
        with c_clear:
            if st.button("Limpiar selección", key=f"clear_sel_{page}"):
                st.session_state.mass_selected_ids = set()
                st.session_state.sel_version += 1
                st.rerun()

        st.caption(f"Seleccionados acumulados: {len(st.session_state.mass_selected_ids)}  ·  (se mantienen al cambiar de página)")

        # ===== Masivo: TODOS de la página (B) =====
        st.divider()
        page_cnt = len(page_df)
        if st.button(f"Actualizar TODOS los de esta página a Opción B — {page_cnt} ítems", type="secondary", use_container_width=True, key=f"btn_allB_{page}"):
            plan = []
            for _, r in page_df.iterrows():
                try:
                    new_price = int(r["Opción B (objetivo)"])
                except Exception:
                    continue
                old_price = 0 if pd.isna(r["PVP actual (BRUTO)"]) else int(r["PVP actual (BRUTO)"])
                plan.append({"modo": "B", "sku": r["SKU"], "item_id": r["item_id"], "old_price": old_price, "new_price": new_price})
            st.session_state.bulk_plan = {"note": f"Página {page} → Opción B", "updates": plan, "skipped": []}

        # ===== Panel de SELECCIONADOS (checkbox para quitar) =====
        st.divider()
        st.write("### Seleccionados (todas las páginas dentro del filtro actual)")

        # Solo los seleccionados que estén dentro del filtro actual:
        selected_all = [iid for iid in st.session_state.mass_selected_ids if iid in set(show_df["item_id"])]
        if selected_all:
            sel_df = show_df.set_index("item_id").loc[selected_all].reset_index()

            out = pd.DataFrame({
                "item_id": sel_df["item_id"],
                "SKU": sel_df["SKU"],
                "Título (Excel)": sel_df["Título (Excel)"],
                "PVP actual (BRUTO)": sel_df["PVP actual (BRUTO)"].apply(lambda v: None if pd.isna(v) else int(v)),
                "Opción A ($)": sel_df["Opción A (mismo tramo)"].apply(lambda v: None if pd.isna(v) else int(v)),
                "Margen A (%)": sel_df["Margen A (%)"],
                "Opción B ($)": sel_df["Opción B (objetivo)"].astype(int),
                "Margen B (%)": sel_df["Margen B (%)"].astype(float),
            })

            # Vista editable con checkbox para mantener/quitar (marcados por defecto)
            base_ids = out["item_id"].tolist()
            edit_view = out[["SKU","Título (Excel)","PVP actual (BRUTO)","Opción A ($)","Margen A (%)","Opción B ($)","Margen B (%)"]].copy()
            edit_view.insert(0, "✓", True)

            sel_editor_key = f"selected_editor_{st.session_state.sel_version}"
            edited = st.data_editor(
                edit_view,
                hide_index=True,
                use_container_width=True,
                key=sel_editor_key,
                column_config={
                    "✓": st.column_config.CheckboxColumn("✓", help="Destilda para quitar de la selección"),
                    "PVP actual (BRUTO)": st.column_config.NumberColumn("PVP actual (BRUTO)", format="$%d"),
                    "Opción A ($)": st.column_config.NumberColumn("Opción A ($)", format="$%d"),
                    "Margen A (%)": st.column_config.NumberColumn("Margen A (%)", format="%.1f"),
                    "Opción B ($)": st.column_config.NumberColumn("Opción B ($)", format="$%d"),
                    "Margen B (%)": st.column_config.NumberColumn("Margen B (%)", format="%.1f"),
                },
            )

<<<<<<< HEAD
            if st.button("Cambiar seleccionados", type="secondary", use_container_width=True):
=======
            if st.button("Aplicar cambios a la selección", type="secondary", use_container_width=True):
>>>>>>> 254480d (feat: cacheos y mejoras de fluidez en app.py)
                keep_mask = edited["✓"].astype(bool).tolist()
                keep_ids = {base_ids[i] for i, keep in enumerate(keep_mask) if keep}
                to_remove = set(base_ids) - keep_ids
                if to_remove:
                    st.session_state.mass_selected_ids -= to_remove
                    st.session_state.sel_version += 1
                st.rerun()
        else:
            st.caption("No hay seleccionados.")

        # ===== Aplicar a SELECCIONADOS (A o B) + Confirmación =====
        st.write("")
        csel1, csel2 = st.columns([0.6, 0.4])
        with csel1:
            st.caption(f"Seleccionados totales (todas las páginas): {len(st.session_state.mass_selected_ids)}")
        with csel2:
            mode_sel = st.selectbox("Opción para SELECCIONADOS", ["B (objetivo)", "A (mismo tramo)"], index=0, key="sel_mode")

        if st.button("Aplicar a SELECCIONADOS", type="primary", use_container_width=True, key="btn_apply_selected"):
            plan, skipped = [], []
            use_A = mode_sel.startswith("A")
            show_all_map = show_df.set_index("item_id")
            for iid in list(st.session_state.mass_selected_ids):
                if iid not in show_all_map.index:
                    continue
                r = show_all_map.loc[iid]
                if use_A:
                    if pd.isna(r["Opción A (mismo tramo)"]):
                        skipped.append({"sku": r["SKU"], "item_id": iid, "motivo": "Opción A no disponible (tramo abierto)"})
                        continue
                    new_price = int(r["Opción A (mismo tramo)"])
                    modo = "A"
                else:
                    new_price = int(r["Opción B (objetivo)"])
                    modo = "B"
                old_price = 0 if pd.isna(r["PVP actual (BRUTO)"]) else int(r["PVP actual (BRUTO)"])
                plan.append({"modo": modo, "sku": r["SKU"], "item_id": iid, "old_price": old_price, "new_price": new_price})
            st.session_state.bulk_plan = {"note": f"Seleccionados → Opción {'A' if use_A else 'B'}", "updates": plan, "skipped": skipped}

        # ===== Ejecución del plan MASIVO =====
        bp = st.session_state.get("bulk_plan")
        if bp:
            st.info(f"{bp['note']} · Cambios a aplicar: {len(bp['updates'])}  · Omitidos: {len(bp['skipped'])}")
            if bp["skipped"]:
                with st.expander("Omitidos (no aplicables)", expanded=False):
                    st.dataframe(pd.DataFrame(bp["skipped"]), use_container_width=True, hide_index=True)

            if bp["updates"]:
                df_plan = pd.DataFrame(bp["updates"])[["sku","item_id","old_price","new_price","modo"]]
                df_plan.rename(columns={"sku":"SKU","item_id":"ItemID","old_price":"Precio actual","new_price":"Nuevo precio","modo":"Opción"}, inplace=True)
                st.dataframe(df_plan, use_container_width=True, hide_index=True)

                col_ok, col_cancel = st.columns(2)
                with col_ok:
                    if st.button("Confirmar actualización MASIVA", type="primary", use_container_width=True):
                        ok, fail, errors = 0, 0, []
                        succeeded_ids = []
                        for upd in bp["updates"]:
                            try:
                                ml.update_item_price(upd["item_id"], upd["new_price"])
                                ok += 1
                                succeeded_ids.append(upd["item_id"])
                            except Exception as e:
                                fail += 1
<<<<<<< HEAD
                                (errors := errors or []).append({**upd, "error": str(e)})
=======
                                errors.append({**upd, "error": str(e)})
>>>>>>> 254480d (feat: cacheos y mejoras de fluidez en app.py)
                        st.success(f"Listo: {ok} OK, {fail} fallos.")
                        if errors:
                            st.error("Algunos ítems fallaron:")
                            st.dataframe(pd.DataFrame(errors)[["sku","item_id","old_price","new_price","modo","error"]], use_container_width=True, hide_index=True)
                        # Limpiar SOLO los que salieron bien
                        if succeeded_ids:
                            st.session_state.mass_selected_ids -= set(succeeded_ids)
                            st.session_state.sel_version += 1
                        st.session_state.bulk_plan = None
                with col_cancel:
                    if st.button("Cancelar plan", use_container_width=True):
                        st.session_state.bulk_plan = None

<<<<<<< HEAD
=======
        # ---------- No encontrados (tabla + Excel cacheado) ----------
>>>>>>> 254480d (feat: cacheos y mejoras de fluidez en app.py)
        st.divider()
        with st.expander(f"No están en ML (según snapshot) — {int(df['item_id'].isna().sum())}", expanded=False):
            no_ml = df[df["item_id"].isna()][["CODIGO SKU", "PRODUCTO"]].rename(columns={"CODIGO SKU":"SKU"})
            st.dataframe(no_ml, use_container_width=True, hide_index=True)

<<<<<<< HEAD
=======
            # Generar Excel cacheado por contenido (lista de tuplas)
            rows = list(no_ml[["SKU", "PRODUCTO"]].itertuples(index=False, name=None))
            xlsx_bytes = build_no_ml_excel_bytes(rows)
            st.download_button(
                label="⬇️ Descargar como Excel los No Encontrados",
                data=xlsx_bytes,
                file_name="no_encontrados_ml.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

>>>>>>> 254480d (feat: cacheos y mejoras de fluidez en app.py)
# ---------- TAB 1: Buscar (SKU/Título) ----------
with tab1:
    st.subheader("Buscar por SKU o Título (usa el Excel cargado)")

    pct_search = st.slider("Margen objetivo (%) sobre PRECIO NETO", min_value=1, max_value=80, value=30, step=1, key="pct_search")
    target_pct_search = float(pct_search)
    mode = st.radio("Buscar por", options=["SKU", "Título"], horizontal=True)
    query = st.text_input("Ingresa tu búsqueda", placeholder="Ej: LIBRO12345  •  o parte del título sin acentos")

    if "excel_global" not in st.session_state:
        st.warning("Primero sube el Excel en la pestaña 📊 Excel · Márgenes para habilitar el buscador.")
    else:
        base_df = st.session_state["excel_global"]
        if query.strip():
            if mode == "SKU":
                q_norm = normalize_sku(query)
                candidates = base_df[base_df["__SKU_NORM__"] == q_norm].copy()
            else:
                q_norm = strip_accents_lower(query)
                candidates = base_df[base_df["title_norm"].str.contains(q_norm, na=False)].copy()

            if candidates.empty:
                st.info("Sin coincidencias en el Excel.")
            else:
                not_in_ml = candidates[candidates["item_id"].isna()].copy()
                in_ml = candidates[candidates["item_id"].notna()].copy()

                if len(not_in_ml):
                    with st.expander(f"Coincidencias que no están en ML (según snapshot): {len(not_in_ml)}", expanded=False):
                        st.dataframe(
                            not_in_ml[["CODIGO SKU","PRODUCTO"]].rename(columns={"CODIGO SKU":"SKU","PRODUCTO":"Título"}),
                            use_container_width=True, hide_index=True
                        )

                if in_ml.empty:
                    st.stop()

                ids = in_ml["item_id"].tolist()
<<<<<<< HEAD
                st.caption(f"Consultando {len(set(ids))} ítems en ML…")

                def fetch_items_info(iids: List[str]) -> Dict[str, Dict[str, Any]]:
                    info: Dict[str, Dict[str, Any]] = {}
                    if not iids:
                        return info
                    seen=set(); uniq=[]
                    for iid in iids:
                        if iid not in seen:
                            seen.add(iid); uniq.append(iid)
                    bulk = ml.get_items_bulk_parallel(uniq, workers=BULK_WORKERS)
                    for entry in bulk:
                        body = entry.get("body") or {}
                        iid = body.get("id")
                        if iid:
                            info[iid] = {"title": body.get("title"), "price": body.get("price")}
                    return info

                info_map = fetch_items_info(ids)
=======
                uniq_ids_s = tuple(sorted(set(ids)))
                st.caption(f"Consultando {len(uniq_ids_s)} ítems en ML… (cacheado por IDs)")

                # ---------- consulta ML (cacheada) ----------
                info_map = fetch_items_info_cached(uniq_ids_s, workers=BULK_WORKERS)

>>>>>>> 254480d (feat: cacheos y mejoras de fluidez en app.py)
                in_ml["price_actual"] = in_ml["item_id"].map(lambda x: info_map.get(x, {}).get("price"))
                in_ml["titulo_ml"] = in_ml["item_id"].map(lambda x: info_map.get(x, {}).get("title", ""))

                # overrides locales tras éxito unitario (refresco inmediato)
                overrides = st.session_state.get("unitary_override_prices", {})
                if overrides:
                    in_ml["price_actual"] = in_ml.apply(
                        lambda r: overrides.get(r["item_id"], r["price_actual"]), axis=1
                    )

                def compute_row_s(row):
                    c = float(row["costo"])
                    pa = row["price_actual"]
                    pa_val = float(pa) if pd.notna(pa) else None
                    if pa_val is None:
                        m_curr = None
                    else:
                        m_curr = margen_pct_sobre_neto(pa_val, c, FEE_RULES)
                    pB = solve_pvp_para_margen_neto(costo_neto=c, target_pct=target_pct_search, rules=FEE_RULES)
                    mB = margen_pct_sobre_neto(pB, c, FEE_RULES)
                    pA = option_A_mismo_tramo(pa_val, FEE_RULES)
                    mA = None if pA is None else margen_pct_sobre_neto(pA, c, FEE_RULES)
                    return pd.Series({
                        "margen_actual_pct": m_curr,
                        "precio_A_mismo_tramo": pA,
                        "margen_A_pct": mA,
                        "precio_B_objetivo": pB,
                        "margen_B_pct": mB,
                    })

                calc_s = in_ml.apply(compute_row_s, axis=1)
                in_ml = pd.concat([in_ml, calc_s], axis=1)

                cols_view = [
                    "CODIGO SKU","PRODUCTO","costo","price_actual","margen_actual_pct",
                    "precio_A_mismo_tramo","margen_A_pct","precio_B_objetivo","margen_B_pct","item_id","titulo_ml"
                ]
                show = in_ml[cols_view].copy()
                show.rename(columns={
                    "CODIGO SKU":"SKU","PRODUCTO":"Título (Excel)","costo":"Costo (NETO)",
                    "price_actual":"PVP actual (BRUTO)","margen_actual_pct":"Margen actual (%)",
                    "precio_A_mismo_tramo":"Opción A (mismo tramo)","margen_A_pct":"Margen A (%)",
                    "precio_B_objetivo":"Opción B (objetivo)","margen_B_pct":"Margen B (%)",
                    "titulo_ml":"Título (ML)",
                }, inplace=True)

                page_size_s = st.selectbox("Filas por página", options=[10, 20, 50], index=1, key="pagesize_search")
                total_rows_s = len(show); total_pages_s = max(1, math.ceil(total_rows_s / page_size_s))
                page_s = st.number_input("Página", min_value=1, max_value=total_pages_s, value=1, step=1, key="page_search")
                start_s = (page_s - 1) * page_size_s; end_s = start_s + page_size_s
                page_df = show.iloc[start_s:end_s].copy()

                st.caption(f"Mostrando {len(page_df)} de {total_rows_s} filas · Página {page_s}/{total_pages_s}")

                st.dataframe(
                    page_df.style.format({
                        "Costo (NETO)":"${:,.0f}",
                        "PVP actual (BRUTO)":lambda x: "-" if pd.isna(x) else f"${int(x):,}",
                        "Margen actual (%)":lambda x: "-" if x is None or pd.isna(x) else f"{x:.1f}%",
                        "Opción A (mismo tramo)":lambda x: "-" if pd.isna(x) else f"${int(x):,}",
                        "Margen A (%)":lambda x: "-" if x is None or pd.isna(x) else f"{x:.1f}%",
                        "Opción B (objetivo)":"${:,.0f}",
                        "Margen B (%)":"{:.1f}%".format,
                    }),
                    use_container_width=True, hide_index=True,
                )

                st.divider()
                st.write("Acciones (A/B/C) por fila")

                if "confirm_update" not in st.session_state:
                    st.session_state.confirm_update = None

                # Placeholder para mostrar el éxito ABAJO (cerca de las acciones)
                success_placeholder = st.empty()

                for _, row in page_df.iterrows():
                    sku = row["SKU"]; item_id = row["item_id"]; pa = row["PVP actual (BRUTO)"]
                    pA = None if pd.isna(row["Opción A (mismo tramo)"]) else int(row["Opción A (mismo tramo)"])
                    pB = int(row["Opción B (objetivo)"])
                    base_row = in_ml[in_ml["item_id"] == item_id].iloc[0]
                    costo_net = float(base_row["costo"])
                    mA = base_row["margen_A_pct"]
                    mB = base_row["margen_B_pct"]
                    m_curr = base_row["margen_actual_pct"]

                    c1, c2, c3, c4 = st.columns([0.35, 0.22, 0.22, 0.21])
                    with c1:
                        st.write(f"{sku} — {base_row['titulo_ml']}")
                        st.write(f"Actualmente: {fmt_money(pa)} ({fmt_pct(m_curr)})")
                    with c2:
                        labelA = "Aplicar Opción A (n/a)" if pA is None else f"Aplicar Opción A ({fmt_money(pA)} — Margen {fmt_pct(mA)})"
                        if st.button(labelA, key=f"apA_s_{item_id}_{page_s}", disabled=(pA is None)):
                            st.session_state.confirm_update = {
                                "item_id": item_id, "sku": sku,
                                "old_price": 0 if pd.isna(pa) else int(pa),
                                "new_price": pA,
                                "nota": f"Opción A · Margen≈{fmt_pct(mA)}",
                                "context": "search",
                            }
                    with c3:
                        labelB = f"Aplicar Opción B ({fmt_money(pB)} — Margen {fmt_pct(mB)})"
                        if st.button(labelB, key=f"apB_s_{item_id}_{page_s}"):
                            st.session_state.confirm_update = {
                                "item_id": item_id, "sku": sku,
                                "old_price": 0 if pd.isna(pa) else int(pa),
                                "new_price": pB,
                                "nota": f"Opción B · Margen≈{fmt_pct(mB)}",
                                "context": "search",
                            }
                    with c4:
                        key_c = f"priceC_{item_id}"

                        # --- Procesar RESET PENDIENTE del input C ANTES de dibujar el widget ---
                        if key_c in st.session_state.pending_resets:
                            if key_c in st.session_state:
                                del st.session_state[key_c]
                            st.session_state.pending_resets.remove(key_c)
                        # ----------------------------------------------------------------------

                        price_c = st.number_input("Precio C", min_value=0, step=10, format="%d", key=key_c)
                        if price_c > 0:
                            mC = margen_pct_sobre_neto(price_c, costo_net, FEE_RULES)
                            st.caption(f"Margen C: {mC:.1f}%")
                            if st.button(f"Aplicar Opción C ({fmt_money(price_c)})", key=f"apC_s_{item_id}_{page_s}"):
                                st.session_state.confirm_update = {
                                    "item_id": item_id, "sku": sku,
                                    "old_price": 0 if pd.isna(pa) else int(pa),
                                    "new_price": int(price_c),
                                    "nota": f"Opción C (personalizado) · Margen≈{mC:.1f}%",
                                    "context": "search",
                                    "reset_input_key": key_c,
                                }
                        else:
                            st.caption("Margen C: —")

                cu = st.session_state.confirm_update
                if cu and cu.get("context") == "search":
                    msg = f"{cu['nota']} — Vas a cambiar {cu['item_id']} ({cu['sku']}) de {fmt_money(cu['old_price'])} a {fmt_money(cu['new_price'])}. Confirma para aplicar."
<<<<<<< HEAD
                    st.code(msg, language=None)   # fondo gris, monoespaciado, no colapsa espacios
=======
                    st.code(msg, language=None)   # texto plano sin colapsar espacios
>>>>>>> 254480d (feat: cacheos y mejoras de fluidez en app.py)
                    d1, d2 = st.columns(2)
                    with d1:
                        if st.button("Confirmar cambio", type="primary"):
                            try:
                                ml.update_item_price(cu["item_id"], cu["new_price"])
                                st.session_state.unitary_success_payload = {"text": "Se ha modificado el precio de manera correcta."}
                                st.session_state.unitary_override_prices[cu["item_id"]] = cu["new_price"]
                                if cu.get("reset_input_key"):
                                    st.session_state.pending_resets.add(cu["reset_input_key"])
                                st.session_state.confirm_update = None
                                st.rerun()
                            except Exception as e:
                                st.error("Fallo al actualizar el precio.")
                                st.exception(e)
                    with d2:
                        if st.button("Cancelar"):
                            st.session_state.confirm_update = None

                payload = st.session_state.unitary_success_payload
                if payload:
                    success_placeholder.success(payload.get("text", "Operación realizada."))
                    try:
                        st.toast(payload.get("text", "Operación realizada."))
                    except Exception:
                        pass
                    st.session_state.unitary_success_payload = None
