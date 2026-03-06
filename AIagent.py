import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    print("Error: GOOGLE_API_KEY not found in environment variables.")
    exit(1)


def main():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=google_api_key,
        temperature=0.5,
    )

    customer_name = input("Enter the name of your customer: ")
    query = input("Enter your technical issue: ")

    support_agent = Agent(
        role="Technical Solutions Engineer",
        goal="Analyze customer technical issues deeply and provide clear step-by-step solutions",
        backstory=(
            f"You are a senior technical engineer at SDAIA. "
            f"You are assisting {customer_name}, an important client. "
            "You specialize in diagnosing issues, identifying root causes, and giving structured solutions. "
            "Always break down your answer into clear steps. Be precise, professional, and solution-oriented."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=True
    )

    support_task = Task(
        description="Answer this customer support question from {}: {}".format(
            customer_name, query),
        expected_output="A helpful support response in email format",
        agent=support_agent
    )

    support_crew = Crew(
        agents=[support_agent],
        tasks=[support_task],
        verbose=True,
        memory=True
    )

    response = support_crew.kickoff()
    print("\nFinal Support Response:\n", response)


if __name__ == "__main__":
    main()
