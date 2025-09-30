from __future__ import annotations

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Callable

import requests
from requests.adapters import HTTPAdapter

ML_TOKEN_URL = "https://api.mercadolibre.com/oauth/token"
ML_API_BASE = "https://api.mercadolibre.com"


class MercadoLibreClient:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
        site: str = "MLC",
        rpm_ceiling: int = 1200,
        bulk_size: int = 20,   # Límite real del endpoint /items?ids=... (20)
        timeout: int = 20,
        max_retries: int = 5,
        initial_backoff: float = 0.8,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self.site = site
        self.bulk_size = max(1, min(20, bulk_size))  # ¡NO > 20!
        self.timeout = timeout
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff

        self._access_token: Optional[str] = None
        self._token_expiry_ts: float = 0.0

        # Rate limit (token bucket) protegido con lock para uso en paralelo
        self._rpm_ceiling = max(60, rpm_ceiling)
        self._last_tick = time.time()
        self._allowance = float(self._rpm_ceiling)
        self._rate_lock = threading.Lock()

        # Lock para refrescar token sin carreras
        self._token_lock = threading.Lock()

        # Session con pool de conexiones (keep-alive)
        self._session = requests.Session()
        adapter = HTTPAdapter(pool_connections=64, pool_maxsize=64)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    # -------- Token & Rate Limit ---------
    def _throttle(self) -> None:
        """Token bucket por minuto, protegido con lock para multihilo."""
        with self._rate_lock:
            now = time.time()
            time_passed = now - self._last_tick
            self._last_tick = now
            self._allowance += time_passed * (self._rpm_ceiling / 60.0)
            if self._allowance > self._rpm_ceiling:
                self._allowance = float(self._rpm_ceiling)
            if self._allowance < 1.0:
                sleep_s = (1.0 - self._allowance) * (60.0 / self._rpm_ceiling)
                if sleep_s > 0:
                    time.sleep(sleep_s)
                self._allowance = 0.0
            else:
                self._allowance -= 1.0

    def _ensure_token(self) -> None:
        # fast path
        if self._access_token and time.time() < self._token_expiry_ts - 60:
            return
        # evitar refrescos simultáneos
        with self._token_lock:
            if self._access_token and time.time() < self._token_expiry_ts - 60:
                return
            self._refresh_access_token()

    def _refresh_access_token(self) -> None:
        data = {
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": self.refresh_token,
        }
        resp = self._session.post(ML_TOKEN_URL, data=data, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"OAuth refresh failed: {resp.status_code} {resp.text}")
        payload = resp.json()
        self._access_token = payload["access_token"]
        ttl = int(payload.get("expires_in", 21600))  # ~6h por defecto
        self._token_expiry_ts = time.time() + ttl

    # ------------- HTTP wrapper ---------------
    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        self._ensure_token()
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"Bearer {self._access_token}"
        url = f"{ML_API_BASE}{path}"

        backoff = self.initial_backoff
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                resp = self._session.request(method, url, headers=headers, timeout=self.timeout, **kwargs)
            except requests.RequestException:
                if attempt == self.max_retries:
                    raise
                time.sleep(backoff)
                backoff *= 1.7
                continue

            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt == self.max_retries:
                    return resp
                time.sleep(backoff + (0.1 * attempt))  # backoff con jitter
                backoff *= 1.7
                continue

            return resp
        return resp  # pragma: no cover

    # ------------- API de conveniencia ---------------
    def get_me(self) -> Dict[str, Any]:
        r = self._request("GET", "/users/me")
        if r.status_code != 200:
            raise RuntimeError(f"/users/me failed: {r.status_code} {r.text}")
        return r.json()

    def list_item_ids_scan(self, seller_id: str) -> List[str]:
        """Intenta usar search_type=scan; si no funciona, cae a paginación por offset."""
        ids: List[str] = []
        # Preferimos scan
        params = {"search_type": "scan", "limit": 50}
        scroll_id: Optional[str] = None
        while True:
            if scroll_id:
                params["scroll_id"] = scroll_id
            r = self._request("GET", f"/users/{seller_id}/items/search", params=params)
            if r.status_code != 200:
                break  # fallback a offset
            data = r.json()
            batch = data.get("results", [])
            ids.extend(batch)
            scroll_id = data.get("scroll_id")
            if not batch or not scroll_id:
                return ids

        # Fallback a offset
        ids = []
        limit = 50
        offset = 0
        while True:
            r = self._request("GET", f"/users/{seller_id}/items/search", params={"limit": limit, "offset": offset})
            if r.status_code != 200:
                raise RuntimeError(f"search items failed: {r.status_code} {r.text}")
            data = r.json()
            batch = data.get("results", [])
            ids.extend(batch)
            if len(batch) < limit:
                break
            offset += limit
        return ids

    # ---- helpers para /items?ids=... ----
    def _get_items_chunk(self, chunk: List[str]) -> List[Dict[str, Any]]:
        ids_param = ",".join(chunk)
        r = self._request("GET", "/items", params={"ids": ids_param})
        if r.status_code != 200:
            raise RuntimeError(f"/items bulk failed: {r.status_code} {r.text}")
        return r.json()

    def get_items_bulk(self, item_ids: List[str]) -> List[Dict[str, Any]]:
        """Serial (compat)"""
        out: List[Dict[str, Any]] = []
        if not item_ids:
            return out
        for i in range(0, len(item_ids), self.bulk_size):
            chunk = item_ids[i : i + self.bulk_size]
            out.extend(self._get_items_chunk(chunk))
        return out

    def get_items_bulk_parallel(self, item_ids: List[str], workers: int = 10) -> List[Dict[str, Any]]:
        """Paraleliza los bulk GET /items?ids=... respetando rate-limit global."""
        out: List[Dict[str, Any]] = []
        if not item_ids:
            return out

        # Crear chunks (de 20)
        chunks: List[List[str]] = []
        for i in range(0, len(item_ids), self.bulk_size):
            chunks.append(item_ids[i : i + self.bulk_size])

        # Ejecutar en paralelo
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futures = {ex.submit(self._get_items_chunk, ch): idx for idx, ch in enumerate(chunks)}
            for fut in as_completed(futures):
                out.extend(fut.result())
        return out

    def get_item(self, item_id: str) -> Dict[str, Any]:
        r = self._request("GET", f"/items/{item_id}")
        if r.status_code != 200:
            raise RuntimeError(f"/items/{item_id} failed: {r.status_code} {r.text}")
        return r.json()

    def update_item_price(self, item_id: str, new_price: float) -> Dict[str, Any]:
        payload = {"price": int(round(new_price))}  # CLP sin decimales
        r = self._request("PUT", f"/items/{item_id}", json=payload)
        if r.status_code not in (200, 202):
            raise RuntimeError(f"PUT /items price failed: {r.status_code} {r.text}")
        return r.json() if r.text else {"status": r.status_code}

    # --------- Índice SKU→ItemID ---------
    def build_sku_index(
        self,
        seller_id: str,
        progress: Optional[Callable[[str, float], None]] = None,
    ) -> Dict[str, str]:
        """Construye {SELLER_SKU: item_id}. Usa scan + bulk y extrae de attributes."""
        def notify(msg: str, pct: float) -> None:
            if progress:
                try:
                    progress(msg, pct)
                except Exception:
                    pass

        notify("Listando publicaciones del vendedor…", 0.05)
        all_ids = self.list_item_ids_scan(seller_id)
        total = max(1, len(all_ids))

        sku_index: Dict[str, str] = {}
        processed = 0
        notify("Descargando detalles en bloque…", 0.1)

        for i in range(0, len(all_ids), self.bulk_size):
            chunk = all_ids[i : i + self.bulk_size]
            bulk = self._get_items_chunk(chunk)
            for entry in bulk:
                body = entry.get("body") or {}
                if not body:
                    continue
                item_id = body.get("id")
                attributes = body.get("attributes", [])
                seller_custom_field = body.get("seller_custom_field")
                sku_value: Optional[str] = None
                # Buscar atributo SELLER_SKU
                for attr in attributes or []:
                    if (attr.get("id") or "").upper() == "SELLER_SKU":
                        sku_value = (
                            attr.get("value_name")
                            or attr.get("value_id")
                            or attr.get("value")
                        )
                        break
                if not sku_value:
                    sku_value = seller_custom_field
                if sku_value and item_id:
                    sku_index[str(sku_value).strip()] = item_id

            processed += len(chunk)
            notify("Procesando…", 0.1 + 0.85 * (processed / total))

        notify("Índice SKU→ID listo", 1.0)
        return sku_index
