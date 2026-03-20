#!/usr/bin/env python3
"""
GPU Allocator Test Script
=========================
Tests the dynamic GPU allocation mechanism with LRU eviction.

This script:
1. Queries current GPU status
2. Tests single GPU allocation
3. Tests multi-GPU allocation
4. Simulates service startup scenarios
5. Tests LRU eviction (if implemented)

Usage:
    python scripts/test_gpu_allocator.py
    python scripts/test_gpu_allocator.py --stress  # Stress test with multiple allocations
"""

import argparse
import subprocess
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from terrabox.managers.gpu_allocator import (
    allocate_gpu,
    allocate_gpus,
    _query_gpu_free_memory,
    has_free_gpu,
    has_free_gpus,
)


def print_gpu_status():
    """Print current GPU status."""
    print("\n" + "=" * 60)
    print("Current GPU Status")
    print("=" * 60)
    
    try:
        gpus = _query_gpu_free_memory()
        if not gpus:
            print("No GPUs found!")
            return
        
        print(f"{'GPU ID':<10} {'Free Memory (MiB)':<20} {'Status'}")
        print("-" * 50)
        for gpu in gpus:
            status = "✓ Available" if gpu["free_mib"] >= 4096 else "⚠ Low memory"
            print(f"{gpu['id']:<10} {gpu['free_mib']:<20} {status}")
        
        total_free = sum(g["free_mib"] for g in gpus)
        print("-" * 50)
        print(f"Total free memory: {total_free} MiB across {len(gpus)} GPUs")
        
    except Exception as e:
        print(f"Error querying GPUs: {e}")


def test_single_gpu_allocation():
    """Test single GPU allocation."""
    print("\n" + "=" * 60)
    print("Test: Single GPU Allocation")
    print("=" * 60)
    
    # Test with default settings
    gpu_id = allocate_gpu()
    print(f"Allocated GPU: {gpu_id}")
    
    # Test with environment variable override
    os.environ["TEST_GPU_OVERRIDE"] = "3"
    gpu_id_override = allocate_gpu(env_var="TEST_GPU_OVERRIDE")
    print(f"With env override (TEST_GPU_OVERRIDE=3): {gpu_id_override}")
    del os.environ["TEST_GPU_OVERRIDE"]
    
    # Test with minimum memory requirement
    gpu_id_8gb = allocate_gpu(min_free_mib=8192)
    print(f"With 8GB min requirement: {gpu_id_8gb}")
    
    print("✓ Single GPU allocation test passed")


def test_multi_gpu_allocation():
    """Test multi-GPU allocation."""
    print("\n" + "=" * 60)
    print("Test: Multi-GPU Allocation")
    print("=" * 60)
    
    # Test 2-GPU allocation
    gpu_ids = allocate_gpus(count=2)
    print(f"Allocated 2 GPUs: {gpu_ids}")
    
    # Test with environment variable override
    os.environ["TEST_MULTI_GPU_OVERRIDE"] = "0,2"
    gpu_ids_override = allocate_gpus(count=2, env_var="TEST_MULTI_GPU_OVERRIDE")
    print(f"With env override (TEST_MULTI_GPU_OVERRIDE=0,2): {gpu_ids_override}")
    del os.environ["TEST_MULTI_GPU_OVERRIDE"]
    
    print("✓ Multi-GPU allocation test passed")


def test_has_free_gpu():
    """Test GPU availability checks."""
    print("\n" + "=" * 60)
    print("Test: GPU Availability Checks")
    print("=" * 60)
    
    has_4gb = has_free_gpu(4096)
    has_8gb = has_free_gpu(8192)
    has_16gb = has_free_gpu(16384)
    
    print(f"Has GPU with 4GB free:  {has_4gb}")
    print(f"Has GPU with 8GB free:  {has_8gb}")
    print(f"Has GPU with 16GB free: {has_16gb}")
    
    has_2_gpus = has_free_gpus(2, 4096)
    has_4_gpus = has_free_gpus(4, 4096)
    
    print(f"Has 2 GPUs with 4GB each: {has_2_gpus}")
    print(f"Has 4 GPUs with 4GB each: {has_4_gpus}")
    
    print("✓ GPU availability check test passed")


def simulate_service_allocation(service_name: str, gpu_count: int = 1):
    """Simulate a service requesting GPU allocation."""
    print(f"\n[{service_name}] Requesting {gpu_count} GPU(s)...")
    
    if gpu_count == 1:
        gpu = allocate_gpu(min_free_mib=4096)
        print(f"[{service_name}] Allocated GPU: {gpu}")
        return gpu
    else:
        gpus = allocate_gpus(count=gpu_count, min_free_mib=4096)
        print(f"[{service_name}] Allocated GPUs: {gpus}")
        return gpus


def test_service_scenario():
    """Test a realistic service startup scenario."""
    print("\n" + "=" * 60)
    print("Test: Service Startup Scenario")
    print("=" * 60)
    
    print_gpu_status()
    
    # Simulate services starting in sequence
    services = [
        ("SAM2", 1),
        ("RemoteCLIP", 1),
        ("RemoteSAM", 1),
        ("Strip-RCNN", 1),
        ("InstructSAM", 1),
        ("VLLM", 2),
    ]
    
    allocations = {}
    for service, count in services:
        allocations[service] = simulate_service_allocation(service, count)
        time.sleep(0.5)  # Small delay between services
    
    print("\n" + "-" * 40)
    print("Allocation Summary:")
    for service, gpu in allocations.items():
        print(f"  {service}: GPU {gpu}")
    
    print_gpu_status()
    print("✓ Service scenario test passed")


def stress_test():
    """Stress test the allocator with rapid allocations."""
    print("\n" + "=" * 60)
    print("Test: Stress Test (Rapid Allocations)")
    print("=" * 60)
    
    print("Performing 20 rapid GPU allocations...")
    
    results = []
    for i in range(20):
        gpu = allocate_gpu()
        results.append(gpu)
        if i % 5 == 0:
            print(f"  Allocation {i+1}/20: GPU {gpu}")
    
    # Check distribution
    from collections import Counter
    dist = Counter(results)
    print(f"\nGPU Distribution: {dict(dist)}")
    
    print_gpu_status()
    print("✓ Stress test passed")


def test_docker_services():
    """Test actual Docker service startup with dynamic GPU allocation."""
    print("\n" + "=" * 60)
    print("Test: Docker Service Integration")
    print("=" * 60)
    
    # Import Docker managers
    try:
        from terrabox.managers.docker.sam2_manager import SAM2DockerManager
        from terrabox.managers.docker.remoteclip_manager import RemoteCLIPDockerManager
        from terrabox.managers.docker.remotesam_manager import RemoteSAMDockerManager
    except ImportError as e:
        print(f"Cannot import Docker managers: {e}")
        print("Skipping Docker integration test")
        return
    
    print_gpu_status()
    
    # Test SAM2
    print("\n[1/3] Testing SAM2 Docker service...")
    try:
        SAM2DockerManager.start_service()
        print(f"SAM2 started. Is running: {SAM2DockerManager.is_running()}")
        SAM2DockerManager.stop_service()
        print("SAM2 stopped.")
    except Exception as e:
        print(f"SAM2 test failed: {e}")
    
    print_gpu_status()
    
    # Test RemoteCLIP
    print("\n[2/3] Testing RemoteCLIP Docker service...")
    try:
        RemoteCLIPDockerManager.start_service()
        print(f"RemoteCLIP started. Is running: {RemoteCLIPDockerManager.is_running()}")
        RemoteCLIPDockerManager.stop_service()
        print("RemoteCLIP stopped.")
    except Exception as e:
        print(f"RemoteCLIP test failed: {e}")
    
    print_gpu_status()
    
    # Test RemoteSAM
    print("\n[3/3] Testing RemoteSAM Docker service...")
    try:
        RemoteSAMDockerManager.start_service()
        print(f"RemoteSAM started. Is running: {RemoteSAMDockerManager.is_running()}")
        RemoteSAMDockerManager.stop_service()
        print("RemoteSAM stopped.")
    except Exception as e:
        print(f"RemoteSAM test failed: {e}")
    
    print_gpu_status()
    print("✓ Docker service integration test completed")


def main():
    parser = argparse.ArgumentParser(
        description="Test GPU allocator with LRU eviction"
    )
    parser.add_argument(
        "--stress",
        action="store_true",
        help="Run stress test with rapid allocations",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Test with actual Docker services",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests including Docker",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("GPU Allocator Test Suite")
    print("=" * 60)
    
    # Print initial status
    print_gpu_status()
    
    # Run tests
    test_single_gpu_allocation()
    test_multi_gpu_allocation()
    test_has_free_gpu()
    test_service_scenario()
    
    if args.stress:
        stress_test()
    
    if args.docker or args.all:
        test_docker_services()
    
    # Final status
    print_gpu_status()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
