"""
SentiKenya IPFS Service
========================
Handles pinning raw social media data to IPFS for immutability.
Provides content-addressed storage and Merkle tree verification.
"""

import hashlib
import json
import logging
import time
from typing import Optional, Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)


class MerkleTree:
    """
    Simple Merkle tree for batch verification of processed posts.

    Used for cross-node consensus — each node computes the same Merkle root
    for the same batch of posts, verifying processing integrity.
    """

    @staticmethod
    def hash_leaf(data: str) -> str:
        return hashlib.sha256(data.encode()).hexdigest()

    @staticmethod
    def hash_pair(left: str, right: str) -> str:
        combined = left + right
        return hashlib.sha256(combined.encode()).hexdigest()

    @classmethod
    def compute_root(cls, hashes: List[str]) -> str:
        """Compute Merkle root from list of leaf hashes."""
        if not hashes:
            return cls.hash_leaf("")

        if len(hashes) == 1:
            return hashes[0]

        # Pad to even number
        if len(hashes) % 2 != 0:
            hashes.append(hashes[-1])

        next_level = []
        for i in range(0, len(hashes), 2):
            next_level.append(cls.hash_pair(hashes[i], hashes[i + 1]))

        return cls.compute_root(next_level)


class IPFSService:
    """
    IPFS integration for immutable data storage.

    In production:
    - Connects to local IPFS node via HTTP API
    - Pins raw posts for tamper-proof audit trail
    - Generates content IDs (CIDs) for referencing

    In development:
    - Simulates IPFS by computing content hashes
    - Stores data locally with IPFS-like addressing
    """

    def __init__(
        self,
        api_url: str = "http://localhost:5001",
        gateway_url: str = "https://ipfs.io/ipfs",
        enabled: bool = False,
    ):
        self.api_url = api_url
        self.gateway_url = gateway_url
        self.enabled = enabled
        self._local_store: Dict[str, dict] = {}  # Dev-mode local store
        self._pin_count = 0

    async def pin_post(self, post_data: dict) -> Optional[str]:
        """
        Pin a raw post to IPFS.

        Returns the CID (Content Identifier) for the pinned data.
        """
        content = json.dumps(post_data, sort_keys=True, default=str)
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        if self.enabled:
            try:
                import httpx

                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.api_url}/api/v0/add",
                        files={"file": ("post.json", content.encode())},
                        timeout=10.0,
                    )
                    if response.status_code == 200:
                        result = response.json()
                        cid = result.get("Hash")
                        self._pin_count += 1
                        logger.debug(f"Pinned to IPFS: {cid}")
                        return cid
                    else:
                        logger.error(f"IPFS pin failed: {response.status_code}")
                        return None

            except Exception as e:
                logger.error(f"IPFS pin error: {e}")
                return None
        else:
            # Simulation mode — store locally with hash-based CID
            simulated_cid = f"Qm{content_hash[:44]}"
            self._local_store[simulated_cid] = post_data
            self._pin_count += 1
            return simulated_cid

    async def get_content(self, cid: str) -> Optional[dict]:
        """Retrieve content by CID."""
        if self.enabled:
            try:
                import httpx

                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.api_url}/api/v0/cat?arg={cid}",
                        timeout=10.0,
                    )
                    if response.status_code == 200:
                        return json.loads(response.text)
            except Exception as e:
                logger.error(f"IPFS get error: {e}")
                return None
        else:
            return self._local_store.get(cid)

    def compute_batch_merkle_root(self, content_hashes: List[str]) -> str:
        """
        Compute Merkle root for a batch of processed posts.

        Used for cross-node verification:
        - Node A processes batch -> computes Merkle root
        - Node B processes same batch -> computes Merkle root
        - Roots must match for consensus
        """
        return MerkleTree.compute_root(content_hashes)

    def get_stats(self) -> dict:
        """Return IPFS service statistics."""
        return {
            "enabled": self.enabled,
            "api_url": self.api_url,
            "gateway_url": self.gateway_url,
            "total_pins": self._pin_count,
            "local_store_size": len(self._local_store),
            "mode": "production" if self.enabled else "simulation",
        }
