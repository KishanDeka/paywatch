import argparse
import asyncio
from aiokafka import AIOKafkaProducer
from .data import load_csv, row_event, split_chronologically


async def replay(args):
    if args.rate <= 0:
        raise ValueError("rate must be positive")
    df = load_csv(args.data)
    if args.split != "all":
        df = split_chronologically(df)[args.split]
    producer = AIOKafkaProducer(
        bootstrap_servers=args.bootstrap,
        enable_idempotence=True,
        acks="all",
        max_request_size=1048576,
    )
    await producer.start()
    try:
        for row in df.itertuples(index=False):
            event = row_event(row)
            await producer.send_and_wait(
                args.topic, event.model_dump_json().encode(), key=b"global", partition=0
            )
            await asyncio.sleep(1 / args.rate)
    finally:
        await producer.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/demo.csv")
    parser.add_argument("--bootstrap", default="localhost:9092")
    parser.add_argument("--topic", default="transactions-stream")
    parser.add_argument("--rate", type=float, default=100)
    parser.add_argument(
        "--split", choices=["all", "train", "calibration", "policy", "test"], default="test"
    )
    asyncio.run(replay(parser.parse_args()))


if __name__ == "__main__":
    main()
