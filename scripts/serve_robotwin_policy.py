"""Serve a RoboTwin fine-tuned checkpoint with an explicit action horizon H.

Run this in the openpi environment. RoboTwin connects through the lightweight
``openpi-client`` WebSocket package, so no SAPIEN/Open3D dependency is needed
on this server.
"""

import dataclasses
import logging
import socket

import tyro

from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as training_config


@dataclasses.dataclass
class Args:
    config: str
    checkpoint_dir: str
    action_horizon: int = 50
    port: int = 8000


def main(args: Args) -> None:
    config = training_config.get_config(args.config)
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, action_horizon=args.action_horizon))
    policy = policy_config.create_trained_policy(config, args.checkpoint_dir)
    logging.info("Serving %s with predicted action horizon H=%d on port %d", args.checkpoint_dir, args.action_horizon, args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
