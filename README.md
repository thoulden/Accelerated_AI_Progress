# AI R&D App

This repository contains two independent front-ends for the same simulation model.

## 1. Streamlit app (original)

- `app.py` houses the main code to run the app;
- `multiple_sims.py` runs the code to simulate many of the models under distributions of parameter assumptions;
- `single_sim.py` runs the code to simulate a single simulation for given parameters;
- `requirements.txt` tells streamlit which packages to install.

Run with `streamlit run app.py`.

## 2. Standalone static site (no Streamlit)

The `docs/` folder is a self-contained recreation of the same tool as a plain
HTML/CSS/JavaScript website — no Python server required. It reproduces the exact
same layout, controls, math and figures as the Streamlit app, but replaces the
sidebar mode dropdown with a top tab bar to switch between **Single Simulation**
and **Multiple Simulations**.

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
`http://localhost:8000/`. It can also be deployed directly via GitHub Pages
(set the source to the `docs/` folder).

The Streamlit files above are untouched by this addition.
