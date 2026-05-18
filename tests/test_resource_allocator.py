import json
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest
from unittest import mock

from terrabox.managers import gpu_allocator
from terrabox.managers import resource_allocator as ra


class ResourceAllocatorTests(unittest.TestCase):
    def test_allocate_port_uses_base_when_free(self):
        port = ra.allocate_port(49152, host="127.0.0.1")
        self.assertEqual(port, 49152)

    def test_allocate_port_increments_when_busy(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        busy_port = sock.getsockname()[1]
        try:
            self.assertEqual(ra.allocate_port(busy_port, host="127.0.0.1", max_tries=3), busy_port + 1)
        finally:
            sock.close()

    def test_allocate_gpu_strict_raises_without_enough_memory(self):
        with mock.patch.object(gpu_allocator, "_query_gpu_free_memory", return_value=[{"id": "0", "free_mib": 1000}]):
            with self.assertRaisesRegex(RuntimeError, "No GPU has"):
                gpu_allocator.allocate_gpu(min_free_mib=4096, strict=True)

    def test_allocate_gpu_env_override_still_pins(self):
        with mock.patch.dict("os.environ", {"TEST_GPU_PIN": "2"}):
            self.assertEqual(
                gpu_allocator.allocate_gpu(min_free_mib=999999, env_var="TEST_GPU_PIN", strict=True),
                "2",
            )

    def test_acquire_docker_lease_allocates_resources(self):
        with (
            mock.patch.object(ra, "docker_image_exists", return_value=True),
            mock.patch.object(ra, "allocate_port", return_value=9123),
            mock.patch.object(ra, "allocate_gpu", return_value="3"),
        ):
            lease = ra.acquire_docker_lease(
                service="test",
                image="terrabox/test:latest",
                container_base="terrabox-test",
                host="127.0.0.1",
                base_port=9000,
                internal_port=8000,
                gpu_count=1,
                min_free_mib=4096,
                fallback_gpu_devices="0",
                gpu_env_var="TEST_GPU",
            )

        self.assertEqual(lease.container_name, "terrabox-test-9123")
        self.assertEqual(lease.port, 9123)
        self.assertEqual(lease.internal_port, 8000)
        self.assertEqual(lease.gpu_devices, "3")
        self.assertEqual(lease.api_url, "http://127.0.0.1:9123")

    def test_acquire_docker_lease_evicts_lru_then_retries_gpu(self):
        from terrabox.managers.base_manager import ServiceRegistry

        with (
            mock.patch.object(ra, "docker_image_exists", return_value=True),
            mock.patch.object(ra, "allocate_port", return_value=9124),
            mock.patch.object(ra, "allocate_gpu", side_effect=[RuntimeError("no gpu"), "2"]),
            mock.patch.object(ServiceRegistry, "evict_lru_for_gpu") as evict,
        ):
            lease = ra.acquire_docker_lease(
                service="test",
                image="terrabox/test:latest",
                container_base="terrabox-test",
                host="127.0.0.1",
                base_port=9000,
                internal_port=8000,
                gpu_count=1,
                min_free_mib=4096,
                fallback_gpu_devices="0",
                gpu_env_var="TEST_GPU",
            )

        evict.assert_called_once()
        self.assertEqual(lease.gpu_devices, "2")

    def test_cleanup_managed_containers_dry_run(self):
        def fake_run(cmd, capture_output=True, text=True):
            self.assertIn("--filter", cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="terrabox-sam2-9002\n", stderr="")

        with mock.patch.object(ra.subprocess, "run", fake_run):
            self.assertEqual(
                ra.cleanup_managed_containers(run_id="run-1", dry_run=True),
                ["terrabox-sam2-9002"],
            )

    def test_remove_container_if_exists_removes_stale_name(self):
        commands = []

        def fake_run(cmd, capture_output=True, text=True):
            commands.append(cmd)
            if cmd[:2] == ["docker", "inspect"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="{}\n", stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with mock.patch.object(ra.subprocess, "run", fake_run):
            removed = ra.remove_container_if_exists("terrabox-vllm-9000", service="vlm")

        self.assertTrue(removed)
        self.assertIn(["docker", "rm", "-f", "terrabox-vllm-9000"], commands)

    def test_find_reusable_managed_lease_matches_existing_container(self):
        def fake_run(cmd, capture_output=True, text=True):
            if cmd[:2] == ["docker", "ps"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="terrabox-agent-llm-9110\n", stderr="")
            if cmd[:2] == ["docker", "inspect"]:
                payload = [{
                    "Name": "/terrabox-agent-llm-9110",
                    "State": {"Running": True},
                    "Config": {
                        "Image": "terrabox/agent-llm:latest",
                        "Cmd": ["--model", "/model", "--max-model-len", "24576"],
                        "Labels": {
                            "terrabox.port": "9110",
                            "terrabox.gpu_devices": "2",
                        },
                    },
                    "Mounts": [{"Source": "/models/qwen", "Destination": "/model"}],
                }]
                return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")
            raise AssertionError(f"unexpected command: {cmd}")

        with mock.patch.object(ra.subprocess, "run", fake_run):
            lease = ra.find_reusable_managed_lease(
                service="agent-llm",
                image="terrabox/agent-llm:latest",
                container_base="terrabox-agent-llm",
                host="127.0.0.1",
                internal_port=8000,
                required_model_path="/models/qwen",
                required_cmd_args={"--max-model-len": "24576"},
                health_check=lambda port: port == 9110,
            )

        self.assertIsNotNone(lease)
        assert lease is not None
        self.assertTrue(lease.reused)
        self.assertEqual(lease.container_name, "terrabox-agent-llm-9110")
        self.assertEqual(lease.port, 9110)
        self.assertEqual(lease.gpu_devices, "2")

    def test_find_reusable_managed_lease_can_adopt_legacy_named_container(self):
        def fake_run(cmd, capture_output=True, text=True):
            if cmd[:2] == ["docker", "ps"] and "label=terrabox.managed=true" in cmd:
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            if cmd[:2] == ["docker", "ps"] and "name=terrabox-vllm" in cmd:
                return subprocess.CompletedProcess(cmd, 0, stdout="terrabox-vllm\n", stderr="")
            if cmd[:2] == ["docker", "inspect"]:
                payload = [{
                    "Name": "/terrabox-vllm",
                    "State": {"Running": True},
                    "Config": {
                        "Image": "terrabox/vllm:latest",
                        "Cmd": ["--model", "/model", "--tensor-parallel-size", "1"],
                        "Labels": {},
                        "Env": ["CUDA_VISIBLE_DEVICES=0"],
                    },
                    "NetworkSettings": {
                        "Ports": {
                            "8000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "9000"}],
                        },
                    },
                    "Mounts": [{"Source": "/models/vlm", "Destination": "/model"}],
                }]
                return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")
            raise AssertionError(f"unexpected command: {cmd}")

        with mock.patch.object(ra.subprocess, "run", fake_run):
            lease = ra.find_reusable_managed_lease(
                service="vlm",
                image="terrabox/vllm:latest",
                container_base="terrabox-vllm",
                host="127.0.0.1",
                internal_port=8000,
                required_model_path="/models/vlm",
                required_cmd_args={"--tensor-parallel-size": "1"},
                health_check=lambda port: port == 9000,
            )

        self.assertIsNotNone(lease)
        assert lease is not None
        self.assertTrue(lease.reused)
        self.assertEqual(lease.container_name, "terrabox-vllm")
        self.assertEqual(lease.port, 9000)
        self.assertEqual(lease.gpu_devices, "0")

    def test_record_service_event_updates_manifest(self):
        lease = ra.ResourceLease(
            service="sam2",
            container_name="terrabox-sam2-9123",
            image="terrabox/sam2:latest",
            host="127.0.0.1",
            port=9123,
            internal_port=9002,
            gpu_devices="2",
            created_at="2026-05-06T00:00:00",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = f"{tmpdir}/run_manifest.json"
            with mock.patch.dict("os.environ", {"TERRABOX_RUN_MANIFEST_PATH": manifest}):
                ra.record_service_event({"event": "start_requested", "service": "sam2", "lease": lease.__dict__})

            data = json.loads(Path(manifest).read_text(encoding="utf-8"))
            self.assertEqual(data["services"]["terrabox-sam2-9123"]["gpu_devices"], "2")
            self.assertEqual(data["services"]["terrabox-sam2-9123"]["last_event"], "start_requested")


if __name__ == "__main__":
    unittest.main()
