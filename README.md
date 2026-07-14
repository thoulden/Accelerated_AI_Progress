# Accelerated AI Progress simulator

**▶ Live site: https://thoulden.github.io/Accelerated_AI_Progress/**

Use the link above. It is a static site (GitHub Pages) with **no wake-up
delay**. The old Streamlit app at
https://accelerated-ai-progress.streamlit.app/ is deprecated — it now just
redirects here, but because Streamlit Community Cloud sleeps idle apps you
have to sit through its wake-up screen first, so please share/bookmark the
GitHub Pages link instead.

This repository contains two front-ends for the same simulation model.

## 1. Standalone static site (recommended) — `docs/`

A self-contained recreation of the tool as a plain HTML/CSS/JavaScript website
— no Python server, no wake-up. It reproduces the same layout, controls, math
and figures as the original Streamlit app, with a top tab bar to switch between
**Single Simulation**, **Multiple Simulations**, and the **Speed-Up
Calculator**.

- `docs/index.html` — page markup, text and parameter tables;
- `docs/css/styles.css` — styling;
- `docs/js/simulation.js` — a faithful JavaScript port of the simulation math
  (numerically identical to `single_sim.py` / `multiple_sims.py`, verified to
  IEEE-754 double precision across every scenario and option combination);
- `docs/js/app.js` — UI wiring and the Plotly figures (one-for-one with the
  original matplotlib charts);
- `docs/vendor/plotly.min.js` — vendored Plotly.js (basic bundle) for charting.

All simulation math runs client-side in the browser. To view it locally, serve
the `docs/` folder over HTTP (e.g. `python3 -m http.server -d docs`) and open
`http://localhost:8000/`. It is deployed via GitHub Pages (Settings → Pages →
source: `main` branch, `/docs` folder).

## 2. Streamlit app (deprecated / redirect)

- `app.py` — now redirects visitors to the static site above;
- `single_sim.py` — original single-simulation logic (still the reference for
  the JS port);
- `multiple_sims.py` — original multi-simulation logic;
- `requirements.txt` — Streamlit's package list.

The original interactive app can still be run locally with an older revision,
but the deployed Streamlit URL only serves the redirect.
