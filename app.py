import streamlit as st
from pipeline import run_research_pipeline

st.set_page_config(page_title="Multi-Agent Research System", layout="wide")

st.title("🧠 Multi-Agent Research System")
st.caption("Search → Read → Write → Critique")

topic = st.text_input("Enter a research topic")

if st.button("Run Research", type="primary"):
    if not topic.strip():
        st.warning("Please enter a topic.")
    else:
        with st.spinner("Agents are researching..."):
            state = run_research_pipeline(topic)

        tab1, tab2, tab3 = st.tabs(["Final Report", "Critic Feedback", "Raw Research"])

        with tab1:
            st.markdown(state["report"])
            st.download_button(
                "Download Report",
                state["report"],
                file_name="research_report.md",
            )

        with tab2:
            st.markdown(state["feedback"])

        with tab3:
            st.subheader("Search Results")
            st.write(state["search_results"])
            st.subheader("Scraped Content")
            st.write(state["scraped_content"])