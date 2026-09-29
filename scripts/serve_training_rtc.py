"""Serve a JAX Training RTC checkpoint using the official OpenPI websocket protocol."""

import dataclasses
import logging

import tyro

from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import config


@dataclasses.dataclass
class Args:
    checkpoint: str
    config: str = "pi05_piper_training_rtc_lora"
    port: int = 8000
    host: str = "0.0.0.0"
    num_steps: int = 10
    default_prompt: str | None = None


def main(args: Args):
    cfg = config.get_config(args.config)
    if getattr(cfg.model, "simulated_delay", None) is None:
        raise ValueError("Select an RTC config")
    policy = policy_config.create_trained_policy(
        cfg, args.checkpoint, sample_kwargs={"num_steps": args.num_steps}, default_prompt=args.default_prompt
    )
    logging.info("RTC metadata: %s", policy.metadata)
    WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
