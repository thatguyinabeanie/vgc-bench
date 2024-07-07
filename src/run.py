import asyncio
from subprocess import DEVNULL, Popen, call


async def run_forever():
    while True:
        process = Popen(
            ["node", "pokemon-showdown", "start", "--no-security"],
            stdout=DEVNULL,
            stderr=DEVNULL,
            cwd="pokemon-showdown",
        )
        await asyncio.sleep(5)
        call(["python", "src/train.py"])
        process.terminate()
        process.wait()


if __name__ == "__main__":
    asyncio.run(run_forever())
