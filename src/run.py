import argparse
from subprocess import run
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_teams", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()
    while True:
        with open("debug.log", "w") as debug_log:
            result = run(
                [
                    "python",
                    "src/train.py",
                    "--num_teams",
                    str(args.num_teams),
                    "--port",
                    str(args.port),
                    "--device",
                    str(args.device),
                ],
                stdout=debug_log,
                stderr=debug_log,
            )
        if result.returncode == 1:
            time.sleep(10)
