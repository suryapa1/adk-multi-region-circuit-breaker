"""
ADK Agent using MultiRegionGemini with PyBreaker circuit breakers.

This is a simple agent that demonstrates how to use the MultiRegionGemini
model as a drop-in replacement for the standard Gemini model string.

The only difference from a normal ADK agent:
    model="gemini-2.5-flash"         →   model=MultiRegionGemini()

Everything else (tools, instructions, sub_agents) works exactly the same.
"""

from google.adk.agents import Agent

from .multi_region_gemini import MultiRegionGemini
from .tools import get_weather, get_time


root_agent = Agent(
    name="multi_region_agent",
    model=MultiRegionGemini(),
    instruction="""You are a helpful assistant with access to weather and time tools.

    When asked about weather, use the get_weather tool.
    When asked about time, use the get_time tool.
    Be concise and friendly in your responses.""",
    description="A resilient agent that automatically fails over across Gemini regions on throttling errors.",
    tools=[get_weather, get_time],
)
