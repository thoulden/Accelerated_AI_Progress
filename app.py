"""
The Accelerated AI Progress simulator has moved to a standalone (non-Streamlit)
website. This Streamlit entrypoint now just redirects visitors there.

The interactive simulator code still lives in this repo (single_sim.py,
multiple_sims.py) and has been reimplemented as a static site under docs/.
"""
import streamlit as st
import streamlit.components.v1 as components

# The simulator now lives here (the repo's docs/ folder, served via GitHub Pages).
# Update this if the site is hosted somewhere else.
NEW_URL = "https://thoulden.github.io/Accelerated_AI_Progress/"

st.set_page_config(page_title="AI Progress Sim has moved", page_icon="🤖")

# Streamlit runs custom HTML/JS inside a sandboxed iframe, so to send the visitor
# away we navigate the *top-level* page (window.top), with a window.parent
# fallback. If both are blocked, the manual link below still works.
components.html(
    f"""
    <script>
      (function () {{
        var url = "{NEW_URL}";
        try {{
          window.top.location.replace(url);
        }} catch (e) {{
          try {{ window.parent.location.replace(url); }} catch (e2) {{}}
        }}
      }})();
    </script>
    """,
    height=0,
)

st.title("This website has moved")
st.markdown(
    f"The Accelerated AI Progress simulator now lives at "
    f"**[{NEW_URL}]({NEW_URL})**.\n\n"
    f"If you are not redirected automatically, please click the link above."
)

st.stop()
