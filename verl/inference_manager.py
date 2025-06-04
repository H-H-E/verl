import torch
import requests
import logging
from typing import Dict, Optional, Any
from verl.atropos_inference import EmbeddedInferenceServer


class InferenceManager:
    """Manages inference servers for online RL training.
    
    Handles starting/stopping policy and reference model servers,
    updating weights, and providing endpoints to external systems.
    """
    
    def __init__(self):
        self._servers: Dict[str, EmbeddedInferenceServer] = {}
        self.logger = logging.getLogger(__name__)
        
    def start_policy_server(
        self, 
        model_name: str, 
        port: int, 
        device: Optional[str] = None
    ) -> str:
        """Start policy model inference server.
        
        Args:
            model_name: HuggingFace model name or path
            port: Port to serve on
            device: Device to load model on ("cpu", "cuda", etc.)
            
        Returns:
            Endpoint URL for the server
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        torch_device = torch.device(device)
        
        # Stop existing policy server if running
        if "policy" in self._servers:
            self.stop_policy_server()
            
        self.logger.info(f"Starting policy server: {model_name} on port {port}")
        server = EmbeddedInferenceServer(model_name, port, torch_device)
        server.start()
        
        self._servers["policy"] = server
        endpoint = f"http://localhost:{port}"
        
        self.logger.info(f"Policy server started at {endpoint}")
        return endpoint
        
    def stop_policy_server(self) -> None:
        """Stop the policy model inference server."""
        if "policy" in self._servers:
            self.logger.info("Stopping policy server")
            self._servers["policy"].stop()
            del self._servers["policy"]
            
    def update_policy_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Update weights on the running policy server.
        
        Args:
            state_dict: New model weights to load
        """
        if "policy" not in self._servers:
            raise RuntimeError("Policy server not running, cannot update weights")
            
        self.logger.info("Updating policy server weights")
        self._servers["policy"].load_weights(state_dict)
        
    def is_policy_server_ready(self, timeout: float = 15.0) -> bool:
        """Check if policy server is responding to health checks.
        
        Args:
            timeout: Request timeout in seconds
            
        Returns:
            True if server is ready, False otherwise
        """
        if "policy" not in self._servers:
            return False
            
        server = self._servers["policy"]
        health_url = f"http://localhost:{server.port}/health"
        
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(health_url, timeout=2.0)
                if response.status_code == 200:
                    self.logger.info(f"Policy server health check passed")
                    return True
            except Exception as e:
                self.logger.debug(f"Policy server health check failed: {e}")
                
            time.sleep(1.0)  # Wait 1 second before retrying
            
        self.logger.error(f"Policy server failed to become ready within {timeout} seconds")
        return False
            
    def get_policy_endpoint(self) -> Optional[str]:
        """Get the policy server endpoint URL.
        
        Returns:
            Endpoint URL or None if no server running
        """
        if "policy" not in self._servers:
            return None
            
        server = self._servers["policy"]
        return f"http://localhost:{server.port}"
        
    def start_reference_server(
        self, 
        model_name: str, 
        port: int, 
        device: Optional[str] = None
    ) -> str:
        """Start reference model inference server for KL divergence."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        torch_device = torch.device(device)
        
        if "reference" in self._servers:
            self.stop_reference_server()
            
        self.logger.info(f"Starting reference server: {model_name} on port {port}")
        server = EmbeddedInferenceServer(model_name, port, torch_device)
        server.start()
        
        self._servers["reference"] = server
        endpoint = f"http://localhost:{port}"
        
        self.logger.info(f"Reference server started at {endpoint}")
        return endpoint
        
    def stop_reference_server(self) -> None:
        """Stop the reference model inference server."""
        if "reference" in self._servers:
            self.logger.info("Stopping reference server")
            self._servers["reference"].stop()
            del self._servers["reference"]
            
    def get_reference_endpoint(self) -> Optional[str]:
        """Get the reference server endpoint URL."""
        if "reference" not in self._servers:
            return None
            
        server = self._servers["reference"]
        return f"http://localhost:{server.port}"
        
    def shutdown_all(self) -> None:
        """Stop all running inference servers."""
        self.logger.info("Shutting down all inference servers")
        self.stop_policy_server()
        self.stop_reference_server() 