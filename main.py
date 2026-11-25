import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai

app = FastAPI(title="Telemetry & AI Race Engineer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = os.getenv("DATABASE_URL")

def get_db():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

def init_db():
    if not DATABASE_URL:
        return
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS laps (
            id SERIAL PRIMARY KEY,
            usuario TEXT NOT NULL,
            plataforma TEXT NOT NULL,
            juego TEXT NOT NULL,
            circuito TEXT NOT NULL,
            vehiculo TEXT NOT NULL,
            tiempo_total_ms INTEGER NOT NULL,
            valida BOOLEAN DEFAULT TRUE,
            telemetria JSONB NOT NULL
        );
        ALTER TABLE laps ADD COLUMN IF NOT EXISTS valida BOOLEAN DEFAULT TRUE;
    """)
    conn.commit()
    cur.close()
    conn.close()

@app.on_event("startup")
def startup_event():
    init_db()

class TelemetryPoint(BaseModel):
    pos: float
    time_ms: int
    speed: float
    throttle: float
    brake: float
    lean_angle: float = 0.0
    x: float
    z: float

class LapPayload(BaseModel):
    usuario: str
    plataforma: str
    juego: str
    circuito: str
    vehiculo: str
    tiempo_total_ms: int
    valida: bool = True
    telemetria: List[TelemetryPoint]

def interpolate_lap(lap: dict, grid: np.ndarray):
    pts = sorted(lap["telemetria"], key=lambda p: p["pos"])
    pos = np.array([p["pos"] for p in pts])
    return {
        "time_s": np.interp(grid, pos, [p["time_ms"] / 1000.0 for p in pts]),
        "speed": np.interp(grid, pos, [p["speed"] for p in pts]),
        "throttle": np.interp(grid, pos, [p["throttle"] for p in pts]),
        "brake": np.interp(grid, pos, [p["brake"] for p in pts]),
        "lean": np.interp(grid, pos, [p.get("lean_angle", 0.0) for p in pts]),
        "x": np.interp(grid, pos, [p["x"] for p in pts]),
        "z": np.interp(grid, pos, [p["z"] for p in pts]),
    }

def ask_race_engineer_multi(ref_lap: dict, target_lap: dict, delta: np.ndarray, delta_rate: np.ndarray, ref_data: dict, tgt_data: dict, grid: np.ndarray) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    worst_indices = np.argsort(delta_rate)[-3:][::-1]
    loss_zones = []
    for idx in worst_indices:
        if delta_rate[idx] > 0.0005:
            loss_zones.append({
                "pct_circuito": round(float(grid[idx]) * 100, 1),
                "vel_piloto_kmh": round(float(tgt_data["speed"][idx]), 1),
                "vel_referencia_kmh": round(float(ref_data["speed"][idx]), 1),
                "freno_piloto": round(float(tgt_data["brake"][idx]), 2),
                "freno_referencia": round(float(ref_data["brake"][idx]), 2),
                "delta_acumulado_s": round(float(delta[idx]), 3)
            })

    if not api_key:
        zona = loss_zones.__getitem__(0)["pct_circuito"] if loss_zones else 0
        vel_diff = round(loss_zones.__getitem__(0)["vel_referencia_kmh"] - loss_zones.__getitem__(0)["vel_piloto_kmh"], 1) if loss_zones else 0
        return (
            f"📻 [Radio Simulada] {target_lap['usuario']} cede {delta[-1]:+.3f}s frente a {ref_lap['usuario']}.\n"
            f"• Punto crítico al {zona}% del trazado: diferencia de {vel_diff} km/h en paso por curva.\n"
            f"• Consejo: retrasa ligeramente la frenada y abre la trazada antes de girar para ganar velocidad de salida."
        )

    client = genai.Client(api_key=api_key)
    prompt = f"""
    Eres un ingeniero de pista de motorsport. Compara la vuelta de {target_lap['usuario']} contra la referencia de {ref_lap['usuario']} en {target_lap['juego']} ({target_lap['circuito']} - {target_lap['vehiculo']}).
    Diferencia total: {delta[-1]:+.3f} segundos.
    Zonas donde más tiempo pierde {target_lap['usuario']}: {loss_zones}.
    Dame 3 consejos directos y fáciles de entender para un principiante (máximo 120 palabras, estilo radio de boxes).
    """
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return response.text
    except Exception as e:
        return f"Radio no disponible: {str(e)}"

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

@app.post("/api/laps")
def upload_lap(payload: LapPayload):
    lap = payload.model_dump()
    juego = lap["juego"].replace("\x00", "").strip()
    circuito = lap["circuito"].replace("\x00", "").strip()
    vehiculo = lap["vehiculo"].replace("\x00", "").strip()
    
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO laps (usuario, plataforma, juego, circuito, vehiculo, tiempo_total_ms, valida, telemetria)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
        """,
        (lap["usuario"], lap["plataforma"], juego, circuito, vehiculo, lap["tiempo_total_ms"], lap["valida"], json.dumps(lap["telemetria"]))
    )
    new_id = cur.fetchone()["id"]
    conn.commit()
    cur.close()
    conn.close()
    return {"status": "ok", "lap_id": new_id}

@app.get("/api/filters")
def get_filters():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT juego, circuito FROM laps ORDER BY juego, circuito;")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    juegos = sorted(list({r["juego"] for r in rows}))
    circuitos_por_juego = {j: [] for j in juegos}
    for r in rows:
        circuitos_por_juego[r["juego"]].append(r["circuito"])
    return {"juegos": juegos, "circuitos_por_juego": circuitos_por_juego}

@app.get("/api/leaderboard")
def get_leaderboard(juego: Optional[str] = None, circuito: Optional[str] = None):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, usuario, plataforma, juego, circuito, vehiculo, tiempo_total_ms, COALESCE(valida, TRUE) as valida
        FROM laps
        WHERE LOWER(juego) = LOWER(%s) AND LOWER(circuito) = LOWER(%s)
        ORDER BY tiempo_total_ms ASC;
        """,
        (juego or "", circuito or "")
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

@app.get("/api/compare")
def compare_laps(ids: List[int] = Query(...)):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM laps WHERE id = ANY(%s);", (ids,))
    selected = cur.fetchall()
    cur.close()
    conn.close()

    if not selected:
        raise HTTPException(status_code=404, detail="No se encontraron vueltas")

    selected.sort(key=lambda l: ids.index(l["id"]))
    ref_lap = selected.__getitem__(0)

    grid = np.linspace(0.0, 1.0, 500)
    ref_interp = interpolate_lap(ref_lap, grid)

    palette = ["#22c55e", "#3b82f6", "#f59e0b", "#ec4899", "#a855f7"]
    series = []
    engineer_advice = ""

    for i, lap in enumerate(selected):
        interp = interpolate_lap(lap, grid)
        delta = interp["time_s"] - ref_interp["time_s"]
        delta_rate = np.gradient(delta)

        if i == 1:
            engineer_advice = ask_race_engineer_multi(ref_lap, lap, delta, delta_rate, ref_interp, interp, grid)

        series.append({
            "id": lap["id"],
            "usuario": lap["usuario"],
            "plataforma": lap["plataforma"],
            "vehiculo": lap["vehiculo"],
            "tiempo_total_ms": lap["tiempo_total_ms"],
            "is_reference": i == 0,
            "color": palette.__getitem__(i % len(palette)),
            "delta": np.round(delta, 3).tolist(),
            "delta_rate": np.round(delta_rate, 5).tolist(),
            "speed": np.round(interp["speed"], 1).tolist(),
            "throttle": np.round(interp["throttle"], 2).tolist(),
            "brake": np.round(interp["brake"], 2).tolist(),
            "lean": np.round(interp["lean"], 1).tolist(),
            "x": np.round(interp["x"], 2).tolist(),
            "z": np.round(interp["z"], 2).tolist(),
        })

    if len(selected) == 1:
        engineer_advice = f"Mostrando vuelta base de {ref_lap['usuario']}. Marca otra vuelta en la tabla para comparar trazadas y activar al Ingeniero IA."

    return {
        "grid": np.round(grid * 100, 2).tolist(),
        "series": series,
        "ai_engineer": engineer_advice
    }
