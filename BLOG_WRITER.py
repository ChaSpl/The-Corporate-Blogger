from dotenv import load_dotenv
import os
import asyncio
import gc
import aiohttp

# Load environment variables
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
print("✅ API key loaded")

from google.adk.models.google_llm import Gemini
from google.adk.tools import google_search, FunctionTool
from google.genai import types
from google.adk.agents import Agent, SequentialAgent, LoopAgent
from google.adk.runners import InMemoryRunner

retry_config = types.HttpRetryOptions(
    attempts=3,
    exp_base=2,
    initial_delay=2,
    http_status_codes=[429, 500, 503],
)

# Global variable to store blog length
blog_length = 350


initial_research_agent = Agent(
    name="InitialResearchAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config,
    ),
    instruction="""Based on the user's prompt, gather novel, surprising, relevant and high-quality information 
    from your knowledge base and the web to assist in building the context to help write an inspiring
    yet accurate blog post. Focus on novelty, key facts, lateral connections into other areas
    and sources. Keep it concise yet diversity rich.""",
    output_key="research_info",
    tools=[google_search],
)

print("✅ initial_research_agent created")


initial_writer_agent = Agent(
    name="InitialWriterAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config,
    ),
    instruction=lambda context: f"""Based on the user's prompt and the {{research_info}}, write the 
    first draft of a blog with max {blog_length + 50} words.
    Output only the blog text and title, with no introduction or explanation.""",
    output_key="current_blog",
)

print("✅ initial_writer_agent created")


critic_agent = Agent(
    name="CriticAgent",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=retry_config,
    ),
    instruction="""You are a constructive but nerdish blog critic. 
    FACT CHECK the blog {current_blog} comparing to knowledge base and web-sources 
    and point out wrong or inaccurate facts. Then critique the blog: 
    - Evaluate the blog's logic structure, it's unique or novel information, and pacing.
    - Evaluate if the blog is well-written and complete. If you agree, you MUST respond with the exact phrase: "APPROVED"
    - Otherwise, provide 2-3 specific, actionable suggestions for improvement.""",
    output_key="critique",
    tools=[google_search],
)

print("✅ critic_agent created")


def exit_loop():
    return {
        "status": "approved",
        "message": "Story approved. Exiting refinement loop.",
    }

# Refiner Agent: additionally, any corporate text branding can be defined here

refiner_agent = Agent(
    name="RefinerAgent",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=retry_config,
    ),
    instruction=lambda context: f"""You are a blog refiner. You have a blog draft and critique.

    Blog Draft: {{current_blog}}
    Critique: {{critique}}

    Your task is to analyze the critique.
    - IF the critique is EXACTLY "APPROVED", you MUST call the `exit_loop` function.
    - OTHERWISE, rewrite the blog draft to fully incorporate the feedback from the critique.

    Requirements:
    - Unique examples, tension and surprising angles
    - Practical actionable content
    - Dissent or nuance
    - Game theory or tradeoffs where appropriate
    - A hint of British humour              
    - No repetition

    Max length is {blog_length} words.
    """,
    output_key="current_blog",
    tools=[FunctionTool(exit_loop)],
)


print("✅ refiner_agent created")


story_refinement_loop = LoopAgent(
    name="StoryRefinementLoop",
    sub_agents=[critic_agent, refiner_agent],
    max_iterations=2,
)


quality_agent = Agent(
    name="QualityAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config,
    ),
    instruction="""You determine the quality of a blog. You have a current blog {current_blog}.

    Judge based on:
    - Novel content compared to typical blogs
    - Factual accuracy
    - Writing style
    - Practical actionable insights
    - Depth and nuance
    - Overall value

    Provide a rating from 1 to 10. Provide the number only.

    Blog Draft: {current_blog}
    """,
    output_key="quality_rating",
    tools=[google_search],
)

print("✅ quality_agent created")


root_agent = SequentialAgent(
    name="BlogPipeline",
    sub_agents=[
        initial_research_agent,
        initial_writer_agent,
        story_refinement_loop,
        quality_agent,
    ],
)

print("✅ All agents and pipeline created")


async def main():
    user_prompt = input("Enter your blog topic: ")

    if not user_prompt.strip():
        print("❌ No prompt provided. Using default.")
        user_prompt = "Explain the benefits of IT standards."

    global blog_length
    try:
        length_input = input("Enter blog length in words (default 350): ").strip()
        if length_input:
            blog_length = int(length_input)
            if blog_length < 50:
                blog_length = 50
            elif blog_length > 2000:
                blog_length = 2000
    except ValueError:
        blog_length = 350

    print(f"🎬 Generating {blog_length} word blog for: '{user_prompt}'")
    print("=" * 60)

    runner = None

    try:
        runner = InMemoryRunner(agent=root_agent)

        response = await asyncio.wait_for(
            runner.run_debug(user_prompt),
            timeout=300,
        )

        print("\n" + "=" * 60)
        print("FINAL BLOG")
        print("=" * 60)

        final_blog = None
        quality_rating = None
        blog_versions = []

        for event in response:
            if hasattr(event, "actions") and event.actions and event.actions.state_delta:
                state = event.actions.state_delta

                if "current_blog" in state and state["current_blog"]:
                    content = state["current_blog"].strip()
                    if (
                        len(content) > 100
                        and not content.startswith("APPROVED")
                        and "approved" not in content.lower()
                    ):
                        blog_versions.append(("state_current", content))

                if "quality_rating" in state:
                    quality_rating = state["quality_rating"]

            if hasattr(event, "agent_name") and event.agent_name == "RefinerAgent":
                if hasattr(event, "content") and event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            text = part.text.strip()
                            if (
                                len(text) > 100
                                and not text.startswith("APPROVED")
                                and "exit_loop" not in text.lower()
                            ):
                                blog_versions.append(("refiner_agent", text))

        if blog_versions:
            for source, txt in blog_versions:
                if source == "state_current":
                    final_blog = txt
                    break

            if not final_blog:
                for source, txt in blog_versions:
                    if source == "refiner_agent":
                        final_blog = txt
                        break

            if not final_blog:
                final_blog = blog_versions[0][1]

        if final_blog:
            print(final_blog)
        else:
            print("Could not extract final blog")
            for i, (source, content) in enumerate(blog_versions):
                print(f"\nVersion {i+1} from {source}:")
                print(content[:200])

        if quality_rating:
            print("\n" + "=" * 60)
            print(f"✅ QUALITY RATING: {quality_rating}/10")

        print("\n" + "=" * 60)
        print("✅ Blog generation complete")

    except asyncio.TimeoutError:
        print("TIMEOUT: operation exceeded limit")

    except Exception as e:
        print(f"ERROR: {e}")

    finally:
        # First: close the main runner
        if runner is not None and hasattr(runner, "close"):
            await runner.close()

        # Second: force close any remaining aiohttp sessions created by tools
        
        

        to_close = []

        for obj in gc.get_objects():
            try:
                # Close open client sessions
                if isinstance(obj, aiohttp.ClientSession) and not obj.closed:
                    to_close.append(obj.close())

                # Close open connectors (may be coroutine depending on aiohttp version)
                elif isinstance(obj, aiohttp.TCPConnector) and not obj.closed:
                    maybe_coro = obj.close()
                    if asyncio.iscoroutine(maybe_coro):
                        to_close.append(maybe_coro)

            except Exception:
                pass

        if to_close:
            await asyncio.gather(*to_close, return_exceptions=True)




if __name__ == "__main__":
    asyncio.run(main())
