#!/usr/bin/env python3
"""
Atropos Integration Launch Script for VeRL

This script demonstrates the complete $2500 bounty implementation:
- Starts policy and reference inference servers using VeRL's server management
- Connects to Atropos API for rollout data
- Performs GRPO training with token-level advantage support
- Updates policy weights on inference servers for online RL
- Provides the inference endpoints to Atropos environments

Usage:
    python examples/atropos_integration/run_atropos_grpo.py --config config.yaml

Example config.yaml:
    model_name: "microsoft/DialoGPT-small"
    policy_port: 8080
    reference_port: 8081
    atropos_api_url: "http://localhost:9000"
    batch_size: 4
    max_seq_len: 512
    learning_rate: 1e-5
    kl_coeff: 0.1
    num_epochs: 10
    update_frequency: 5
    device: "cuda"
"""

import argparse
import yaml
import logging
import os
import signal
import sys
from pathlib import Path

from verl.atropos_trainer import AtroposTrainer


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('atropos_grpo_training.log')
        ]
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set defaults for optional parameters
    config.setdefault('num_epochs', 1)
    config.setdefault('update_frequency', 1)
    config.setdefault('device', 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu')
    config.setdefault('log_level', 'INFO')
    
    return config


def validate_config(config: dict) -> None:
    """Validate required configuration parameters."""
    required_keys = [
        'model_name', 'policy_port', 'reference_port', 
        'atropos_api_url', 'batch_size', 'max_seq_len',
        'learning_rate', 'kl_coeff'
    ]
    
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")


def main():
    parser = argparse.ArgumentParser(description="Run Atropos GRPO Training")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load and validate configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        config['log_level'] = args.log_level
        validate_config(config)
        
        logger.info("Configuration loaded successfully:")
        for key, value in config.items():
            if 'password' not in key.lower() and 'token' not in key.lower():
                logger.info(f"  {key}: {value}")
        
        # Initialize trainer
        logger.info("Initializing Atropos trainer...")
        trainer = AtroposTrainer(config)
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            trainer.shutdown()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start training
        logger.info("Starting Atropos GRPO training...")
        logger.info("=" * 60)
        logger.info("ATROPOS INTEGRATION - $2500 BOUNTY IMPLEMENTATION")
        logger.info("=" * 60)
        logger.info(f"Policy server will be available at: http://localhost:{config['policy_port']}")
        logger.info(f"Reference server will be available at: http://localhost:{config['reference_port']}")
        logger.info(f"Atropos API endpoint: {config['atropos_api_url']}")
        logger.info("=" * 60)
        
        trainer.train()
        
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 