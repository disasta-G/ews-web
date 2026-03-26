# CLAUDE.md — EWS-Web: Erdwärmesonden-Simulationsprogramm

## Projektziel

Modernes, KI-gestütztes Web-Programm zur Berechnung, Auslegung und Optimierung von
Erdwärmesonden-Anlagen (EWS). Ersatz für das alte Desktop-Programm EWS 5.5 (Huber
Energietechnik AG). Physikalisch identische Rechenmodelle, moderne Python/React-Architektur,
drastisch reduzierter Benutzeraufwand durch KI-Automatisierung.

**Primärer Markt:** Kanton Graubünden (Schweiz), Erweiterung auf weitere Kantone vorgesehen.

---

## Entwicklungsphasen (in dieser Reihenfolge)

### Phase 1 — Kern-Rechenengine (Priorität: HOCH)
Physikalisch korrektes Python-Rechenmodul. Validierung gegen `Ausgabe.ews` (Referenzprojekt).

### Phase 2 — FastAPI-Backend + REST-API
Simulation als Service. Alle Inputs/Outputs über JSON.

### Phase 3 — React-Frontend (Web-UI)
Eingabe, Visualisierung, Karten. Browser-basiert, kein Desktop-Client.

### Phase 4 — KI-Automatisierung (Standort, Wärmebedarf, Kataster)
Adresse → automatisches Standortprofil. Wärmepumpen-PDF → extrahierte Kennwerte.

### Phase 5 — Sondenoptimierung (Positionierung auf Parzelle)
Automatische Platzierung innerhalb definierter Bereiche auf einer Parzelle.

### Phase 6 — Formulare GR (Bohrgesuch + Dimensionierungsnachweis)
Automatisches Ausfüllen der kantonalen ANU-Formulare F-405-01d und F-405-02d.

### Phase 7 — Lernmodus (ML-Surrogate + Selbstoptimierung)
Autonomes Lernen durch Massensimulationen. Nur manuell aktivierbar.

---

## Phase 1: Kern-Rechenengine

### 1.1 Physikalisches Modell

Alle Gleichungen stammen aus: EWS 5.5 Manual (Huber Energietechnik AG, August 2022),
Anhang A, Seiten 77–90.

#### Wärmeleitungsgleichung (Gl. 6.6)
```
∂T_Earth/∂t = a * (∂²T_Earth/∂r² + (1/r) * ∂T_Earth/∂r)
```
Temperaturleitfähigkeit: `a = λ_Earth / (ρ_Earth * cp_Earth)`

#### Numerisches Schema: Crank-Nicholson im Simulationsgebiet
- Eindimensional in radialer Richtung, schichtweise (bis 10 Erdschichten)
- Innere Randbedingung: mittlere Solentemperatur der jeweiligen Schicht
- Äussere Randbedingung: g-Funktionen (Carslaw & Jaeger ODER Eskilson, wählbar)
- Zeitschritte: Sole (klein) → Simulationsgebiet (mittel) → g-Funktion (wöchentlich)

#### Rechengitter (Gl. 6.1–6.5)
```python
# Gitterfaktor f definiert radiales Gitter
r[0] = Di/2                          # Innenradius Sonde
r[1] = Db/2                          # Bohrlochrand
r[j+1] = r[j] + (r[j] - r[j-1]) * f # exponentiell wachsend
# Massenschwerpunkt:
rz[j] = sqrt((r[j]**2 + r[j+1]**2) / 2)
```

#### g-Funktion nach Carslaw & Jaeger (Gl. 6.11)
```python
def g_carslaw_jaeger(r, t, a):
    """Dimensionslose Sprungantwort für unendliche Linienquelle"""
    gamma = 0.5772156649  # Euler-Konstante
    g = 0.5 * (np.log(4*a*t/r**2) - gamma)
    # Korrektur für kurze Zeiten (vollständige Reihenentwicklung Gl. 6.11)
    return g
```

#### g-Funktion nach Eskilson (Gl. 6.12–6.15)
```python
def t_s(H, a):
    """Zeitkonstante nach Eskilson (Gl. 6.12)"""
    return H**2 / (9 * a)

def g_eskilson_single(r1, H, t, a):
    """g-Funktion Einzelsonde (Gl. 6.14), gültig 5*r1²/a < t < t_s"""
    Es = t / t_s(H, a)
    return np.log(H / (2*r1)) + 0.5 * np.log(Es)

def g_eskilson_equilibrium(r1, H):
    """Gleichgewichtszustand t > t_s (Gl. 6.15)"""
    return np.log(H / (2*r1))
```

#### Superposition Sondenfeld (Gl. 6.16–6.17)
```python
def g_field(positions, r1, H, t, a):
    """g-Funktion Sondenfeld via Superposition (Gl. 6.17)"""
    n = len(positions)
    g_r1 = g_eskilson_single(r1, H, t, a)
    sum_corrections = 0
    for i, pos_i in enumerate(positions):
        for j, pos_j in enumerate(positions):
            if i != j:
                A_xy = np.linalg.norm(np.array(pos_i) - np.array(pos_j))
                sum_corrections += np.log(A_xy / r1)
    return g_r1 + sum_corrections / n
```

#### Superposition mit Nachbarsonden (Gl. 6.18)
Wie g_field, aber n = Projektsonden + Nachbarsonden. Mittelwert nur über Projektsonden.

#### Thermische Widerstände Ra, Rb — Doppel-U-Sonde (Gl. 6.27–6.38, Hellström)
```python
def calc_Ra_Rb_doubleU(r0, rs, r1, Bu, lambda_fill, lambda_earth, lambda_s, alpha_w):
    """
    r0:          Innenradius Sondenrohr [m]
    rs:          Aussenradius Sondenrohr [m]
    r1:          Bohrlochradius [m]
    Bu:          Rohrabstand shank spacing [m]
    lambda_fill: Wärmeleitfähigkeit Hinterfüllung [W/mK]
    lambda_earth: Wärmeleitfähigkeit Erdreich [W/mK]
    lambda_s:    Wärmeleitfähigkeit Sondenrohrwand [W/mK]
    alpha_w:     Wärmeübergangskoeffizient Fluid [W/m²K]
    """
    beta = (lambda_fill - lambda_earth) / (lambda_fill + lambda_earth)  # Gl. 6.30
    b = Bu / (2 * r1)   # Exzentrizität Gl. 6.27
    Rs = np.log(rs/r0) / (2 * np.pi * lambda_s)  # Gl. 6.36
    R_conv = 1 / (8 * np.pi * r0 * alpha_w)       # Gl. 6.34
    # Ra nach Hellström (vollständige Formel Gl. 6.35)
    # Rb nach Hellström (vollständige Formel Gl. 6.37–6.38)
    # Vollständige Formeln aus Manual S. 86–87 implementieren
```

#### Thermische Widerstände Ra, Rb — Koaxialsonde (Gl. 6.39–6.40)
Für Koaxialsonden (ri, ra = inneres Rohr; ro, rs = äusseres Rohr).

#### Analytische Erdwärmesondengleichung (Gl. 6.51–6.54)
```python
def T_source(T_m, Rg, Rb, Rm, Rf, q_dot):
    """Quellentemperatur (Gl. 6.53)"""
    return T_m - (Rg + Rb + Rm + Rf) * q_dot

def T_sink(T_m, Rg, Rb, Rm, Rf, q_dot):
    """Rücklauftemperatur (Gl. 6.54)"""
    return T_m - (Rg + Rb - Rm - Rf) * q_dot

def T_m_undisturbed(T_mo, T_grad, H):
    """Mittlere ungestörte Erdreichtemperatur in Tiefe H/2 (Gl. 6.24)"""
    return T_mo + T_grad * H / 2
```

#### Iterationsschleife pro Zeitschritt (Manual S. 78)
```
FOREACH Stunde t:
  1. Einlesen: TSink oder QSource aus Lastprofil
  2. IF Leistungsmodus (QSource gegeben):
       Iteriere TSink bis |TSink - TSink_alt| < Genauigkeit
       TSink_neu = TSource - QSource / (cp_Sole * m_dot)
  3. Berechne Sole (explizit, substep)
  4. Crank-Nicholson für Erdreich (Simulationsgebiet)
  5. g-Funktion Update (wöchentlich)
  6. Schreibe TSource, TSink, QSource, Druck auf Output
```

#### Druckabfall (Hydraulik)
```python
def pressure_drop(L, d_i, m_dot, rho, nu, n_pipes=4):
    """Druckabfall Doppel-U-Sonde (4 Rohre parallel)"""
    v = m_dot / (rho * n_pipes * np.pi * (d_i/2)**2)
    Re = v * d_i / nu
    f = 64 / Re if Re < 2300 else 0.316 * Re**(-0.25)  # Hagen-Poiseuille / Blasius
    return f * (L / d_i) * rho * v**2 / 2
```

### 1.2 Simulationsparameter — Referenzprojekt (Ausgabe.ews)

Projekt: **4 × 160.5 m, Via Travers, Zuoz**

```python
# Sonden
AnzahlSonden = 4
Sondenlaenge = 160.5          # m
Bohrdurchmesser = 0.135       # m
Sondendurchmesser = 0.040     # m (Innen)
Dicke_Sondenrohr = 0.0037     # m
KoaxialSonde = False

# Validierungswerte thermische Widerstände (aus Ausgabe.ews)
R1 = 4.63419950469117e-3      # K/W
Ra = 2.97515608201173e-1      # Km/W
Rb = 7.89644509518729e-2      # Km/W

# Sole
lambdaSole = 0.500            # W/mK
rhoSole = 1037.0              # kg/m³
cpSole = 3905.0               # J/kgK
nueSole = 3.5e-6              # m²/s
Massenstrom = 0.549           # kg/s (pro Sonde, Betrieb)
AuslegungsMassenstrom = 1.024 # kg/s (total)
Druckabfall_Validierung = 5233.0  # Pa

# Erdreich (homogen, alle 10 Schichten gleich)
lambdaErd = 2.70              # W/mK
rhoErd = 2600.0               # kg/m³
cpErd = 1000.0                # J/kgK

# Hinterfüllung
lambdaFill = 2.00             # W/mK
rhoFill = 1180.0              # kg/m³
cpFill = 3040.0               # J/kgK

# Temperaturen
JahresmittelTemp = 2.0        # °C Lufttemperatur
TGrad = 0.030                 # °C/m
BodenerWaermung = 3.71        # °C

# Simulation
LastYear = 50                 # Jahre
Zeitschritt = 60              # Min
numrows = 8760
DimRad = 5
DimAxi = 10
Gitterfaktor = 2.0
RechenRadius = 2.5            # m (Faktor)
Genauigkeit = 0.01            # °C

# g-Funktionen (Validierungswerte aus Ausgabe.ews)
g_values = {
    -4: 5.73871946983590,
    -2: 10.6886512892442,
     0: 14.2826512892442,
    +2: 15.9386512892442,
    +3: 16.1126512892442,
}
Sondenabstand = 10.0          # m

# Lastprofil (Laufzeiten h/d, Jan–Dez)
Laufzeit = [12, 11, 9, 7, 3, 2, 2, 2, 3, 7, 9, 11]
QSpitzeFeb = 1.0              # kW
DauerLastSpitze = 2           # Tage

# WP-Kennwerte (COP)
COP_minus5 = 2.5
COP_0 = 3.0
COP_5 = 3.5
COP_10 = 4.0
COP_15 = 4.5
```

### 1.3 EWS-Dateiformat (.ews)

```
Zeile 1:     Versionsnummer ("28.0 Version des Ausgabefiles, Programm EWS")
Zeilen 2–4:  Projektbezeichnung (Titel, Standort, Auftraggeber)
Zeile 5:     Leerzeile
Zeilen 6–~100: Parameter (Tab-getrennt: Wert [Tab] Einheit [Tab] Bezeichnung [Tab] Variablenname)
Ab Zeile ~101: 8760 Stundenwerte, 8 Spalten Tab-getrennt:
  Stunde | Massenstrom[kg/s] | TSource[°C] | QSource[kW] | TSink[°C] | TMittel[°C] | Druck[Pa] | Betrieb[TRUE/FALSE]
```

### 1.4 Validierungsziel

**Akzeptanzkriterium Phase 1:**
- Mittlere Abweichung TSource: < 0.05 °C
- Maximale Abweichung TSource: < 0.20 °C
- Druckabfall: < ±100 Pa gegen 5233 Pa Referenz
- Ra, Rb: < ±1 % gegen Referenzwerte

---

## Phase 2: FastAPI-Backend

### Projektstruktur
```
ews-web/
├── backend/
│   ├── engine/
│   │   ├── simulation.py         # Kern-Rechenengine
│   │   ├── g_functions.py        # g-Funktionen (Carslaw/Eskilson)
│   │   ├── thermal_resistance.py # Ra, Rb (Hellström)
│   │   ├── hydraulics.py         # Druckabfall
│   │   ├── optimizer.py          # Sondenfeldoptimierung (Phase 5)
│   │   └── surrogate.py          # ML-Surrogate (Phase 7)
│   ├── api/
│   │   ├── main.py               # FastAPI App
│   │   ├── routes/
│   │   │   ├── simulate.py       # POST /api/simulate
│   │   │   ├── optimize.py       # POST /api/optimize
│   │   │   ├── location.py       # GET  /api/location
│   │   │   └── forms.py          # POST /api/forms/gr
│   │   └── models.py             # Pydantic-Schemas
│   ├── services/
│   │   ├── geo_service.py        # GIS-APIs (Swisstopo, GeoGR)
│   │   ├── ai_service.py         # Claude API (WP-PDF, Lastprofil)
│   │   ├── form_service.py       # PDF-Formular-Füllung (PyMuPDF)
│   │   └── learning_service.py   # Lernmodus-Scheduler
│   └── tests/
│       ├── test_engine.py        # Validierung gegen Ausgabe.ews
│       └── test_api.py
├── frontend/                     # Phase 3
└── CLAUDE.md
```

### API-Endpunkte
```
POST /api/simulate
  Input:  SimulationInput (alle Parameter)
  Output: SimulationResult (8760 Stundenwerte + Zusammenfassung + SIA-Auswertung)

POST /api/optimize
  Input:  Parzellenpolygon, Energiebedarf, Constraints (max_bohrmeter, T_min, R-Klasse)
  Output: Optimale Sondenpositionen, Tiefen, Kostenschätzung

GET  /api/location?address=...
  Output: λ, ρ, cp, T_boden, T_gradient, max_bohrmeter, EWS-Zone (grün/gelb/rot), Nachbarsonden

POST /api/extract-heatpump
  Input:  PDF oder Bild (multipart)
  Output: Modell, Kälteleistung, COP-Kurve, Massenstrom, WP-Frostgrenze

POST /api/forms/gr/bohrgesuch
  Input:  Vollständige Projektdaten
  Output: Ausgefülltes PDF F-405-01d (bytes)

POST /api/forms/gr/dimensionierungsnachweis
  Input:  Simulationsergebnis + Projektdaten
  Output: Ausgefülltes PDF F-405-02d (bytes)

POST /api/learning/start   # Lernmodus starten (manuell)
POST /api/learning/stop
GET  /api/learning/status
```

---

## Phase 3: React-Frontend

### Technologie
```
React 18 + TypeScript | Vite | Tailwind CSS
Recharts (Temperatur/Leistungs-Plots)
MapLibre GL JS (Karte, Sondenfeld, Parzelle)
React Hook Form (Parametereingabe)
```

### Wärmebedarf-Eingabe (drei gleichwertige Wege)
```
Option A: Manuell
  - Heizlast [kW], Jahreswärmebedarf [kWh/a]
  - Lastprofil: monatliche Laufzeiten [h/d] ODER CSV-Upload (8760 Stundenwerte)
  - WP-Parameter manuell (COP, Kälteleistung, Frostgrenze)

Option B: Wärmepumpen-Datenblatt hochladen
  - PDF oder Foto der technischen Daten
  - Claude API extrahiert: Modell, Kälteleistung, COP-Kurve, Massenstrom
  - User ergänzt nur den Wärmebedarf des Gebäudes

Option C: KI-Schätzung aus Gebäudebeschreibung
  - Gebäudekategorie (SIA 380/1) + m² + Baujahr + Höhenlage
  - Claude schätzt EBF, Heizbedarf, Volllaststunden nach SIA 384/6 Anhang D
```

### Sondenfeld-UI
```
- MapLibre Karte mit Swisstopo WMTS Hintergrund
- Kataster automatisch laden (geogr.ch für GR, Parzellenpolygon via WFS)
- Erdwärmenutzungskarte GR als Overlay (grün/gelb/rot)
- Vorgeschlagene Max-Bohrmeter (aus Erdsondenkarte, als Hinweistext)
- Bebauungsbereiche einzeichnen (erlaubte Bohrzonen)
- Sonden manuell setzen ODER "Auto-Optimieren" Button
- Bestehende Bohrungen im 50m-Umkreis anzeigen (für Nachbarsonden)
```

### Ergebnis-Seite
```
Plots:
  - TSource + TSink über letztes Simulationsjahr (stündlich)
  - T_min/T_mittel/T_max über 50-Jahr-Periode (jährlich)
  - Entzugsleistung monatlich
  - Druckabfall

Zusammenfassung:
  - T_min (kritisch), Auftrittszeitpunkt
  - SIA 384/6: R1/R2/R3/R4 erfüllt/nicht erfüllt
  - Jahresentzugsenergie, Volllaststunden
  - Druckabfall, Strömungsregime (laminar/turbulent)

Downloads:
  - PDF-Bericht (2-seitig, SIA-konform)
  - CSV Stundenwerte
  - Bohrgesuch F-405-01d (GR) → 1 Klick
  - Dimensionierungsnachweis F-405-02d (GR, bei >4 Sonden) → 1 Klick
```

---

## Phase 4: KI-Automatisierung

### Standortprofil aus Adresse
```python
async def get_location_profile(address: str) -> LocationProfile:
    coords = await swisstopo_geocode(address)
    zone = await query_ews_zone_gr(coords)        # ANU GR Erdwärmenutzungskarte
    max_depth = await query_erdsondenkarte(coords) # kann-ich-bohren.ch
    nearby = await query_nearby_boreholes(coords, radius=50)
    geology = await swisstopo_geocover(coords)
    lambda_est = estimate_lambda_from_geology(geology)
    T_surface = await meteoswiss_annual_mean(coords)
    T_grad = lookup_geothermal_gradient(coords)
    parcel = await load_cadastral_gr(coords)       # geogr.ch WFS
    return LocationProfile(...)
```

### WP-Datenblatt → Kennwerte (Claude API)
```python
async def extract_heatpump_data(file: UploadFile) -> HeatPumpData:
    base64_data = base64.b64encode(await file.read()).decode()
    media_type = "application/pdf" if file.filename.endswith(".pdf") else "image/jpeg"
    
    response = await anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "document" if media_type == "application/pdf" else "image",
                    "source": {"type": "base64", "media_type": media_type, "data": base64_data}
                },
                {
                    "type": "text",
                    "text": """Extrahiere folgende Wärmepumpen-Kennwerte als JSON (nur JSON, kein Text):
{
  "modell": "Herstellerbezeichnung Modell",
  "heizleistung_kw": 0.0,
  "kaelteleistung_kw": 0.0,
  "cop_values": {"B-5W35": 0.0, "B0W35": 0.0, "B5W35": 0.0, "B10W35": 0.0},
  "nenn_massenstrom_kg_s": 0.0,
  "temperaturspreizung_k": 0.0,
  "wp_frostgrenze_c": -10.0
}"""
                }
            ]
        }]
    )
    return HeatPumpData(**json.loads(response.content[0].text))
```

### Kataster GR automatisch laden
```python
async def load_cadastral_gr(coords: LV95Coords) -> CadastralData:
    """Parzellenpolygon von geogr.ch via WFS"""
    wfs_url = (
        "https://geo.gr.ch/wfs/av?"
        "SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature"
        "&TYPENAMES=av:LiegenschaFL"
        f"&BBOX={bbox_from_point(coords, buffer_m=200)},EPSG:2056"
    )
    geojson = await fetch_wfs(wfs_url)
    parcel = find_parcel_containing_point(geojson, coords)
    return CadastralData(geometry=parcel, coords=coords)
```

---

## Phase 5: Sondenfeldoptimierung

```python
from scipy.optimize import differential_evolution

def optimize_borehole_field(
    allowed_zones: List[Polygon],   # erlaubte Bohrzonen auf Parzelle
    energy_demand: EnergyDemand,
    soil_params: SoilParams,
    constraints: OptConstraints,    # max_bohrmeter, T_min_sonde, min_abstand
) -> OptimizationResult:
    """
    Optimiert: n Sonden, Positionen (x,y), Tiefe H
    Minimiert: Gesamtbohrmeter (= Kosten)
    
    Constraints:
      - T_min_sonde >= T_min (SIA 384/6 R1: -1.5°C)
      - Alle Sonden innerhalb allowed_zones
      - Abstand zwischen Sonden >= min_abstand (default 6m)
      - Abstand zu Parzellengrenze >= 3m
      - Tiefe je Sonde <= max_bohrmeter (aus Erdsondenkarte)
    """
    def objective(x):
        positions, depths = decode_params(x)
        if not all_in_zones(positions, allowed_zones): return 1e9
        if min_distance(positions) < constraints.min_abstand: return 1e9
        result = simulate_simplified(positions, depths, energy_demand, soil_params)
        if result.T_min < constraints.T_min: return 1e9
        return sum(depths)
    
    result = differential_evolution(
        objective, bounds=get_bounds(allowed_zones, constraints),
        seed=42, maxiter=500, tol=0.01, workers=-1, popsize=15
    )
    return decode_result(result)
```

**Max-Bohrmeter:**
- Wird aus kantonaler Erdsondenkarte (`kann-ich-bohren.ch` / GR API) abgerufen
- Dem User als **Vorschlag** angezeigt (Hinweistext mit Quelle), nicht zwingend
- Kann im UI überschrieben werden (eigene Eingabe oder Bohrfirmenwissen)

---

## Phase 6: Kantonale Formulare Graubünden

### Rechtlicher Rahmen GR
- **Bewilligungsbehörde:** ANU (Amt für Natur und Umwelt), Abt. Gewässerschutz
- **Einreicheweg:** Gesuchsteller → Standortgemeinde → ANU
- **Kontakt ANU:** vincenzo.cataldi@anu.gr.ch, +41 81 257 29 75
- **≤ 4 Sonden:** Nur F-405-01d erforderlich
- **> 4 Sonden:** F-405-01d + **F-405-02d zwingend** (detaillierte Dimensionierung SIA 384/6)
- **Tiefe > 200m:** Individuelle Prüfung ANU erforderlich (vorher Kontakt aufnehmen)
- **Zonen:** Grün = Standard, Gelb = Spezialauflagen, Rot = verboten
- **Grundwasserschutzzonen S/SS/SA:** EWS verboten

### Formular F-405-01d: Bohrgesuch (Stand 27.11.2023)
**URL:** `https://www.gr.ch/DE/.../ANU_Dokumente/F-405-01d_gesuch_bewilligung_wp_erdwaermesonden.pdf`

Felder die automatisch aus Projektdaten befüllt werden:
```python
BOHRGESUCH_MAPPING = {
    # Standort
    "Gemeinde":               project.gemeinde,
    "Parzellennummer":        project.parzellennummer,
    "Koordinaten_E":          project.coords.E,    # LV95
    "Koordinaten_N":          project.coords.N,
    "Zone_EWK":               location.ews_zone,   # aus Erdwärmenutzungskarte
    
    # Anlage
    "Anzahl_Sonden":          project.n_sonden,
    "Tiefe_je_Sonde_m":       project.H,
    "Gesamttiefe_m":          project.n_sonden * project.H,
    "Sondentyp":              "Doppel-U" if not project.koaxial else "Koaxial",
    "Bohrdurchmesser_mm":     project.Db * 1000,
    
    # Wärmepumpe
    "WP_Hersteller":          project.wp_hersteller,
    "WP_Modell":              project.wp_modell,
    "WP_Heizleistung_kW":     project.wp_heizleistung,
    "WP_Kaelteleistung_kW":   project.wp_kaelteleistung,
    
    # Sole
    "Sole_Typ":               project.sole_typ,
    "Sole_Volumen_Liter":     project.sole_volumen,
    
    # Hinterfüllung
    "Hinterfuellung":         project.fill_material,
    
    # Nutzung
    "Anlagenutzung":          "Heizung und Warmwasser" if not project.kuehlung
                              else "Heizung, Warmwasser und Kühlung",
}
```

### Formular F-405-02d: Dimensionierungsnachweis (Stand 30.11.2022)
**URL:** `https://www.gr.ch/DE/.../ANU_Dokumente/F-405-02d_...pdf`
**Pflicht bei:** > 4 Erdwärmesonden

Zusätzliche Felder aus Simulationsergebnis:
```python
DIMNACHWEIS_ZUSATZ = {
    "Bodentemperatur_C":          location.T_jahresm,
    "Temperaturgradient_K_m":     location.T_grad,
    "Lambda_Erdreich":            soil.lambda_erd,
    "Waermekapazitaet_MJ_m3K":   soil.rho * soil.cp / 1e6,
    "Entzugsleistung_kW":         result.Q_auslegung,
    "Jahresentzugsenergie_kWh":   result.Q_jahres,
    "Volllaststunden_h":          result.volllaststunden,
    "Simulationsdauer_Jahre":     50,              # SIA-Pflicht: 50 Jahre
    "T_min_Sonde_C":              result.T_min,
    "T_min_Monat":                result.T_min_monat,
    "T_min_Jahr":                 result.T_min_jahr,
    "SIA_R1_erfuellt":            result.T_min >= -1.5,
    "Regenerationsrate_Pct":      result.regen_rate,
    "Sondenabstand_m":            project.sondenabstand,
    "Druckabfall_Pa":             result.druckabfall,
    "Stroemung_turbulent":        result.Re > 2300,
    "Berechnungssoftware":        "EWS-Web v1.0 (Methodik: EWS 5.5, Huber Energietechnik AG)",
}
```

### PDF-Ausfüllung
```python
import fitz  # PyMuPDF

async def fill_form_gr(form_type: str, data: dict) -> bytes:
    FORM_URLS = {
        "bohrgesuch": "https://www.gr.ch/DE/.../F-405-01d_gesuch_bewilligung_wp_erdwaermesonden.pdf",
        "dimensionierungsnachweis": "https://www.gr.ch/DE/.../F-405-02d_....pdf",
    }
    pdf_bytes = await fetch_and_cache_pdf(FORM_URLS[form_type])
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    for page in doc:
        for field in page.widgets():
            if field.field_name in data:
                field.field_value = str(data[field.field_name])
                field.update()
    
    return doc.tobytes()
```

**Wichtig:** Formular-PDFs beim ersten Aufruf cachen und Hash prüfen. GR aktualisiert
gelegentlich (zuletzt F-405-01d: 27.11.2023). Bei geändertem Hash: User warnen,
Feldmapping prüfen.

---

## Phase 7: Lernmodus

**Aktivierung:** Ausschliesslich manuell via `POST /api/learning/start`. Stoppt automatisch
bei eingehenden API-Anfragen (Priorität: User-Requests).

```python
class LearningService:
    """
    Massensimulationen → Training von ML-Surrogaten für g-Funktionen.
    Ziel: Optimierungsläufe 100× beschleunigen.
    """
    PARAMETER_RANGES = {
        "H":           (50, 400),    # Sondenlänge [m]
        "n":           (1, 20),      # Anzahl Sonden
        "lambda":      (1.0, 4.5),   # λ Erdreich [W/mK]
        "T_surface":   (2.0, 15.0),  # Bodenoberfläche [°C]
        "q_spezifisch":(10, 80),     # W/m Entzug
        "abstand":     (5, 20),      # Sondenabstand [m]
        "Rb":          (0.04, 0.20), # Bohrlochwidersand [Km/W]
        "Jahre":       [25, 50, 100],
    }
    
    async def run_batch(self, n: int = 1000):
        """Latin Hypercube Sampling über Parameterraum"""
        from scipy.stats.qmc import LatinHypercube
        sampler = LatinHypercube(d=7, seed=42)
        samples = scale_samples(sampler.random(n), self.PARAMETER_RANGES)
        for params in samples:
            result = await simulate_full(params)
            await save_training_record(params, result)
        await retrain_surrogate()
    
    async def retrain_surrogate(self):
        """XGBoost Surrogate für T_min nach 50 Jahren"""
        # Aktivierung nur wenn RMSE < 0.3°C auf Validierungsset
        # Physikalisches Modell bleibt immer als Fallback verfügbar
```

---

## SIA 384/6:2019 — Massgebende Anforderungen

```python
# Minimale Wärmeträgertemperatur nach 50 Jahren (Tabelle 2, prSIA 384/6)
SIA_T_MIN = {
    "R1": -1.5,   # Grundanforderung (keine erhöhten Anforderungen)
    "R2":  0.0,   # Erhöhte Anforderung
    "R3": +1.5,   # Stark erhöhte Anforderung
    "R4": None,   # Regenerationspflicht (Sonderfall)
}
# Alle °C = mittlere Wärmeträgertemperatur = (T_Vorlauf + T_Rücklauf) / 2

NACHBARSONDEN_RADIUS_M = 50      # Alle EWS im 50m-Umkreis berücksichtigen (3.1.1.5)
SIA_NACHWEIS_JAHRE = 50          # Simulationsdauer für SIA-Nachweis (3.4.4.2)
MIN_ABSTAND_GRENZE_M = 3.0       # Parzellengrenzen (SIA 384/6 + GR Vollzug)
MIN_ABSTAND_SONDEN_M = 6.0       # Default, standortabhängig
MAX_TIEFE_STANDARD_GR = 200      # m, darüber individuelle ANU-Prüfung
```

---

## Externe APIs und Datenquellen

| Zweck | URL | Format |
|---|---|---|
| Geocoding CH | `api3.geo.admin.ch/rest/services/api/SearchServer` | JSON |
| Geologie/GeoCover | `api3.geo.admin.ch/rest/services/NE_bund/MapServer` | JSON |
| Swisstopo WMTS | `wmts.geo.admin.ch/1.0.0/` | Tiles |
| Kataster GR | `geo.gr.ch/wfs/av` (Layer: `av:LiegenschaFL`) | WFS/GeoJSON |
| EWS-Nutzungskarte GR | `geo.gr.ch` (WMS) | WMS |
| Erdsondenkarte CH | `kann-ich-bohren.ch` / BFE | Web/JSON |
| Bohrprofile BFE | `map.geo.admin.ch` (Geothermie-Layer) | JSON |
| Formular F-405-01d | `gr.ch/DE/.../F-405-01d_....pdf` | PDF |
| Formular F-405-02d | `gr.ch/DE/.../F-405-02d_....pdf` | PDF |
| Claude API (KI) | `api.anthropic.com/v1/messages` | JSON, API-Key aus `ANTHROPIC_API_KEY` |

**Koordinatensystem:** LV95 (EPSG:2056) ist Standard in GR. Umrechnung mit `pyproj`.

---

## Python-Abhängigkeiten

```
# Phase 1–2
numpy scipy pandas fastapi uvicorn pydantic httpx pytest pyproj

# Phase 3 (Build-Tools)
node npm  # React/Vite im frontend/

# Phase 4–6
anthropic pymupdf (PyMuPDF) aiohttp shapely

# Phase 5
scipy  # bereits Phase 1

# Phase 7
scikit-learn xgboost joblib
```

---

## Tests

```python
# tests/test_engine.py

def test_reference_project():
    """Hauptvalidierung gegen Via Travers Zuoz (4×160.5m)"""
    params = parse_ews_file("tests/fixtures/Ausgabe.ews")
    result = simulate(params)
    reference = parse_ews_hourly("tests/fixtures/Ausgabe.ews")
    
    diff = np.abs(result.T_source_hourly - reference.T_source)
    assert diff.mean() < 0.05, f"Mean error {diff.mean():.4f}°C > 0.05°C"
    assert diff.max()  < 0.20, f"Max error  {diff.max():.4f}°C > 0.20°C"

def test_pressure_drop():
    """Validierung Druckabfall: Referenz 5233 Pa"""
    dp = calc_pressure_drop(H=160.5, d_i=0.040, m_dot=0.549, rho=1037, nu=3.5e-6)
    assert abs(dp - 5233) < 100

def test_thermal_resistances():
    """Ra, Rb aus Ausgabe.ews als Referenz"""
    Ra, Rb = calc_Ra_Rb_doubleU(...)
    assert abs(Ra - 0.297515608) / 0.297515608 < 0.01  # < 1%
    assert abs(Rb - 0.078964451) / 0.078964451 < 0.01

def test_sia_compliance():
    """SIA R1: T_min >= -1.5°C nach 50 Jahren"""
    result = simulate(params_50yr)
    assert result.T_min >= -1.5

def test_g_functions():
    """g-Werte gegen Tabellenwerte aus Ausgabe.ews"""
    g_ref = {-4: 5.7387, -2: 10.6887, 0: 14.2827, 2: 15.9387, 3: 16.1127}
    for lnts, g_val in g_ref.items():
        t = t_s(H=160.5, a=1.039e-6) * np.exp(lnts)
        g_calc = g_eskilson_single(r1=0.0675, H=160.5, t=t, a=1.039e-6)
        assert abs(g_calc - g_val) / g_val < 0.02  # < 2%
```

---

## Wichtige Hinweise für Claude Code

1. **Physik vor allem anderen.** Rechengenauigkeit > Performance > UX.
   Erst validieren (Phase 1 vollständig), dann nächste Phase beginnen.

2. **Ausgabe.ews ist die Wahrheit.** Jede Änderung an Gleichungen → sofort Validierung.

3. **Einheiten SI intern.** Konvertierung (mm→m, kWh→J, etc.) nur an Ein-/Ausgabe-Grenzen.

4. **Graubünden zuerst, dann generisch.** geo_service.py mit Strategy-Pattern aufbauen,
   damit weitere Kantone (ZH, BE, ...) später einfach ergänzt werden können.

5. **Formulare ändern sich.** Bei Start: HTTP HEAD auf Formular-PDFs, ETag/Last-Modified
   gegen gecachte Version prüfen. Bei Änderung: Warning loggen + Issue öffnen.

6. **Max-Bohrmeter = Vorschlag.** Aus Erdsondenkarte abgerufen, dem User prominent als
   "Empfehlung gemäss Erdsondenkarte" angezeigt. Nicht als harter Constraint erzwingen —
   der User (Fachplaner) entscheidet final.

7. **Lernmodus ist opt-in und unterbrechbar.** Niemals automatisch starten.
   Bei eingehender API-Anfrage sofort pausieren, danach weitermachen.

8. **Claude API = claude-sonnet-4-20250514.** Immer dieses Modell für Extraktion.
   API-Key aus Umgebungsvariable `ANTHROPIC_API_KEY` (nie hardcoden).

9. **LV95 (EPSG:2056) ist Heimkoordinatensystem.** Alle Parzellen- und Sondenkoordinaten
   intern in LV95. Anzeige in der Karte als WGS84 (MapLibre), Umrechnung mit pyproj.

10. **Wärmebedarf immer dreigleisig.** Manuell, PDF-Upload und KI-Schätzung sind
    gleichwertige Eingabewege. Keiner soll dem anderen vorgezogen werden in der UI.
