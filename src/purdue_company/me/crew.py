import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List


@CrewBase
class MECrew:
    """Mechanical Engineering Academic Division — coursework, research, and project help."""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def _llm(self) -> LLM:
        return LLM(
            model=os.getenv("MODEL", "anthropic/claude-sonnet-4-6"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=8096,
        )

    @agent
    def subject_expert(self) -> Agent:
        return Agent(
            config=self.agents_config["subject_expert"],
            llm=self._llm(),
            tools=[SerperDevTool(), ScrapeWebsiteTool()],
            verbose=True,
            max_iter=10,
        )

    @agent
    def problem_solver(self) -> Agent:
        return Agent(
            config=self.agents_config["problem_solver"],
            llm=self._llm(),
            tools=[SerperDevTool(), ScrapeWebsiteTool()],
            verbose=True,
            max_iter=8,
        )

    @agent
    def academic_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["academic_writer"],
            llm=self._llm(),
            verbose=True,
            max_iter=5,
        )

    @task
    def research_topic_task(self) -> Task:
        return Task(config=self.tasks_config["research_topic_task"])

    @task
    def solve_or_explain_task(self) -> Task:
        return Task(config=self.tasks_config["solve_or_explain_task"])

    @task
    def write_me_report_task(self) -> Task:
        return Task(config=self.tasks_config["write_me_report_task"])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            manager_llm=self._llm(),
            verbose=True,
            output_log_file="output/me_crew.log",
        )
