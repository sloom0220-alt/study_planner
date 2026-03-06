import os
import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
import PyPDF2

load_dotenv()

st.title("📚 AI Study Planner")
st.write("Upload your study materials and get a personalized study plan!")

uploaded_files = st.file_uploader(
    "Upload PDF files", type="pdf", accept_multiple_files=True
)

days = st.number_input("How many days do you have to study?", min_value=1, max_value=30, value=7)

if st.button("Generate Study Plan"):
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY not found!")
        st.stop()

    llm = LLM(
        model="groq/llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0.5,
    )

    if not uploaded_files:
        st.warning("Please upload at least one PDF file.")
    else:
        with st.spinner("Analyzing your materials..."):
            all_text = ""
            for pdf in uploaded_files:
                reader = PyPDF2.PdfReader(pdf)
                for page in reader.pages:
                    all_text += page.extract_text() or ""

            analyzer = Agent(
                role="Study Material Analyzer",
                goal="Analyze study materials and identify key topics",
                backstory="You are an expert educator who analyzes study materials and identifies the most important topics and concepts.",
                llm=llm,
                allow_delegation=False,
                verbose=True
            )

            planner = Agent(
                role="Study Planner",
                goal="Create an organized and effective study schedule",
                backstory="You are an expert study coach who creates personalized study plans based on material complexity and available time.",
                llm=llm,
                allow_delegation=False,
                verbose=True
            )

            analyze_task = Task(
                description=f"Analyze the following study material and list the main topics and estimated time needed for each:\n\n{all_text[:3000]}",
                expected_output="A list of main topics with estimated study time for each topic.",
                agent=analyzer
            )

            plan_task = Task(
                description=f"Based on the analyzed topics, create a detailed {days}-day study plan. Distribute topics evenly across the days.",
                expected_output=f"A clear {days}-day study schedule showing what to study each day.",
                agent=planner
            )

            crew = Crew(
                agents=[analyzer, planner],
                tasks=[analyze_task, plan_task],
                verbose=True
            )

            result = crew.kickoff()

        st.success("✅ Study Plan Generated!")
        st.markdown(str(result))