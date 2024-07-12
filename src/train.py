import asyncio
from subprocess import DEVNULL, Popen, call


async def train_forever():
    while True:
        process = Popen(
            ["node", "pokemon-showdown", "start", "--no-security"],
            stdout=DEVNULL,
            stderr=DEVNULL,
            cwd="pokemon-showdown",
        )
        await asyncio.sleep(5)
        call(["python", "src/train_step.py"])
        process.terminate()
        process.wait()


if __name__ == "__main__":
    asyncio.run(train_forever())
