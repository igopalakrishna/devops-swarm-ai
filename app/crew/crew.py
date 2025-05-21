from typing import Any
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from vertexai import init as vertexai_init
from vertexai.generative_models import GenerativeModel

# Initialize Vertex AI once
vertexai_init(project="devops-swarm-ai-project", location="us-central1")


@CrewBase
class DevOpsCrew:
    """DevOps Review Crew"""

    def __init__(self, agents_config: dict[str, Any], tasks_config: dict[str, Any]):
        self.agents_config = agents_config
        self.tasks_config = tasks_config
        self.model_id = "gemini-1.5-flash-preview-0514"


    @agent
    def pr_analyzer_agent(self) -> Agent:
        model = GenerativeModel(self.model_id)

        def analyze_pr(title: str, mr_diff: str) -> str:
            prompt = f"Summarize the purpose of this GitLab Merge Request.\nTitle: {title}\nDiff:\n{mr_diff}"
            try:
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                return f"❌ Failed to analyze PR: {e}"

        return Agent(
            config=self.agents_config.get("pr_analyzer_agent"),
            tools=[analyze_pr],
            verbose=True,
            llm=self.llm
        )

    @agent
    def quality_check_agent(self) -> Agent:
        model = GenerativeModel(self.llm)

        def quality_check(code: str) -> str:
            prompt = (
                "You are a code reviewer.\n"
                "Review the following code for logic flaws, performance issues, and dangerous patterns.\n"
                f"Code:\n{code}"
            )
            try:
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                return f"❌ Review failed: {e}"

        return Agent(
            config=self.agents_config.get("quality_check_agent"),
            tools=[quality_check],
            verbose=True,
            llm=self.llm
        )

    @agent
    def ci_health_agent(self) -> Agent:
        model = GenerativeModel(self.llm)

        def ci_health_check(pipeline_logs: str, mr_id: str) -> str:
            prompt = (
                f"Given the following CI pipeline logs for MR ID {mr_id}, "
                "identify failed jobs, error messages, and suggest likely fixes.\n\n"
                f"Logs:\n{pipeline_logs}"
            )
            try:
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                return f"❌ CI health check failed: {e}"

        return Agent(
            config=self.agents_config.get("ci_health_agent"),
            tools=[ci_health_check],
            verbose=True,
            llm=self.llm
        )

    @agent
    def report_agent(self) -> Agent:

        def generate_ci_report(summary: str, quality_issues: str, ci_feedback: str) -> str:
            try:
                return (
                    "**Merge Request Review Summary**\n\n"
                    f"**PR Summary:**\n{summary}\n\n"
                    f"**Code Quality Feedback:**\n{quality_issues}\n\n"
                    f"**CI Health Report:**\n{ci_feedback}"
                )
            except Exception as e:
                return f"❌ Report generation failed: {e}"

        return Agent(
            config=self.agents_config.get("report_agent"),
            tools=[generate_ci_report],
            verbose=True,
            llm=self.llm
        )

    @task
    def analyze_pr_task(self) -> Task:
        return Task(
            config=self.tasks_config.get("analyze_pr_task"),
            agent=self.pr_analyzer_agent()
        )

    @task
    def quality_check_task(self) -> Task:
        return Task(
            config=self.tasks_config.get("quality_check_task"),
            agent=self.quality_check_agent(),
            context=[self.analyze_pr_task()]
        )

    @task
    def ci_health_check_task(self) -> Task:
        return Task(
            config=self.tasks_config.get("ci_health_check_task"),
            agent=self.ci_health_agent(),
            context=[self.analyze_pr_task()]
        )

    @task
    def generate_report_task(self) -> Task:
        return Task(
            config=self.tasks_config.get("generate_report_task"),
            agent=self.report_agent(),
            context=[
                self.analyze_pr_task(),
                self.quality_check_task(),
                self.ci_health_check_task()
            ]
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.pr_analyzer_agent(),
                self.quality_check_agent(),
                self.ci_health_agent(),
                self.report_agent()
            ],
            tasks=[
                self.analyze_pr_task(),
                self.quality_check_task(),
                self.ci_health_check_task(),
                self.generate_report_task()
            ],
            process=Process.sequential,
            verbose=True
        )
