"""Topology harness: declarative multi-core probe descriptions.

A topology is data (cores, buffers, links); the harness validates it against
the device model and, in later steps, plans DMA channels and emits aie.mlir.
Every hardware number comes from tools/aie-device-models.json.
"""
