"""Requires yigraph[engine], YIGRAPH_HOME and configured external services."""

import argparse
import asyncio


async def analyze(dataset, question):
    from yigraph import AAGEngine, load_config

    engine = AAGEngine(load_config())
    try:
        print("Available datasets:", engine.list_datasets())
        engine.specific_dataset(dataset, "graph")
        print(await engine.run(question, mode="normal"))
    finally:
        await engine.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="AMLSim1K")
    parser.add_argument("--question", default="Find the ten most important accounts using PageRank.")
    args = parser.parse_args()
    asyncio.run(analyze(args.dataset, args.question))
