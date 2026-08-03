import subprocess
import unittest
from unittest import mock

from terrabox.managers.resource_allocator import ResourceLease


class RunRecorder:
    def __init__(self):
        self.commands = []

    def __call__(self, cmd, capture_output=True, text=True, timeout=None):
        self.commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="cid\n", stderr="")


def _lease(service: str, port: int, internal_port: int = 9000) -> ResourceLease:
    return ResourceLease(
        service=service,
        container_name=f"terrabox-{service}-{port}",
        image=f"terrabox/{service}:latest",
        host="127.0.0.1",
        port=port,
        internal_port=internal_port,
        gpu_devices="3",
        created_at="2026-05-06T00:00:00",
    )


class DockerResourceManagerTests(unittest.TestCase):
    def test_sam2_docker_manager_uses_resource_lease(self):
        from terrabox.managers.docker import sam2_manager as mod

        recorder = RunRecorder()
        lease = _lease("sam2", 9012, 9002)
        with (
            mock.patch.object(mod.SAM2DockerManager, "_container_is_running", classmethod(lambda cls: False)),
            mock.patch.object(mod, "acquire_docker_lease", return_value=lease),
            mock.patch.object(mod, "remove_container_if_exists", return_value=False),
            mock.patch.object(mod.subprocess, "run", recorder),
        ):
            mod.SAM2DockerManager._start_docker()

        run_cmd = next(cmd for cmd in recorder.commands if cmd[:3] == ["docker", "run", "-d"])
        self.assertIn("--label", run_cmd)
        self.assertIn("terrabox.service=sam2", run_cmd)
        self.assertIn("9012:9002", run_cmd)
        self.assertEqual(mod.SAM2DockerManager.API_URL, "http://127.0.0.1:9012")

    def test_sam2_docker_manager_rebuilds_running_but_unhealthy_container(self):
        from terrabox.managers.docker import sam2_manager as mod

        recorder = RunRecorder()
        lease = _lease("sam2", 9018, 9002)
        stopped = {"called": False}

        def fake_stop(cls):
            stopped["called"] = True

        with (
            mock.patch.object(mod.SAM2DockerManager, "_container_is_running", classmethod(lambda cls: True)),
            mock.patch.object(mod.SAM2DockerManager, "stop_service", classmethod(fake_stop)),
            mock.patch.object(mod, "acquire_docker_lease", return_value=lease),
            mock.patch.object(mod, "remove_container_if_exists", return_value=False),
            mock.patch.object(mod.subprocess, "run", recorder),
        ):
            mod.SAM2DockerManager._start_docker()

        self.assertTrue(stopped["called"])
        run_cmd = next(cmd for cmd in recorder.commands if cmd[:3] == ["docker", "run", "-d"])
        self.assertIn("9018:9002", run_cmd)

    def test_instructsam_docker_manager_uses_resource_lease(self):
        from terrabox.managers.docker import instructsam_manager as mod

        recorder = RunRecorder()
        lease = _lease("instructsam", 9016, 9006)
        with (
            mock.patch.dict("os.environ", {"INSTRUCTSAM_PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
            mock.patch.object(mod.InstructSAMDockerManager, "_container_is_running", classmethod(lambda cls: False)),
            mock.patch.object(mod, "acquire_docker_lease", return_value=lease),
            mock.patch.object(mod, "remove_container_if_exists", return_value=False),
            mock.patch.object(mod.subprocess, "run", recorder),
        ):
            mod.InstructSAMDockerManager._start_docker()

        run_cmd = next(cmd for cmd in recorder.commands if cmd[:3] == ["docker", "run", "-d"])
        self.assertIn("terrabox.service=instructsam", run_cmd)
        self.assertIn("9016:9006", run_cmd)
        self.assertIn("DEVICE=cuda:0", run_cmd)
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", run_cmd)
        self.assertTrue(any(str(arg).startswith("VLLM_API_URL=") for arg in run_cmd))
        self.assertEqual(mod.InstructSAMDockerManager.API_URL, "http://127.0.0.1:9016")

    def test_tool_manager_env_port_overrides_yaml_defaults(self):
        from terrabox.managers.docker import changeos_manager as changeos_mod
        from terrabox.managers.docker import instructsam_manager as instructsam_mod

        with mock.patch.dict(
            "os.environ",
            {
                "INSTRUCTSAM_PORT": "9016",
                "INSTRUCTSAM_GPU_DEVICES": "2",
                "CHANGEOS_PORT": "9017",
                "CHANGEOS_GPU_DEVICES": "2",
            },
        ):
            instructsam_mod.InstructSAMDockerManager.API_URL = "http://127.0.0.1:9006"
            instructsam_mod.InstructSAMDockerManager.GPU_DEVICES = "0"
            changeos_mod.ChangeOSDockerManager.API_URL = "http://127.0.0.1:9007"
            changeos_mod.ChangeOSDockerManager.GPU_DEVICES = "0"

            instructsam_mod.InstructSAMDockerManager._apply_env_overrides()
            changeos_mod.ChangeOSDockerManager._apply_env_overrides()

        self.assertEqual(instructsam_mod.InstructSAMDockerManager.API_URL, "http://127.0.0.1:9016")
        self.assertEqual(instructsam_mod.InstructSAMDockerManager.GPU_DEVICES, "2")
        self.assertEqual(changeos_mod.ChangeOSDockerManager.API_URL, "http://127.0.0.1:9017")
        self.assertEqual(changeos_mod.ChangeOSDockerManager.GPU_DEVICES, "2")

    def test_remoteclip_docker_manager_uses_resource_lease(self):
        from terrabox.managers.docker import remoteclip_manager as mod

        recorder = RunRecorder()
        lease = _lease("remoteclip", 9013, 9003)
        with (
            mock.patch.object(mod.RemoteCLIPDockerManager, "_container_is_running", classmethod(lambda cls: False)),
            mock.patch.object(mod, "acquire_docker_lease", return_value=lease),
            mock.patch.object(mod, "remove_container_if_exists", return_value=False),
            mock.patch.object(mod.subprocess, "run", recorder),
        ):
            mod.RemoteCLIPDockerManager._start_docker()

        run_cmd = next(cmd for cmd in recorder.commands if cmd[:3] == ["docker", "run", "-d"])
        self.assertIn("terrabox.service=remoteclip", run_cmd)
        self.assertIn("9013:9003", run_cmd)
        self.assertEqual(mod.RemoteCLIPDockerManager.API_URL, "http://127.0.0.1:9013")

    def test_remotesam_docker_manager_uses_resource_lease(self):
        from terrabox.managers.docker import remotesam_manager as mod

        recorder = RunRecorder()
        lease = _lease("remotesam", 9014, 9004)
        with (
            mock.patch.object(mod.RemoteSAMDockerManager, "_container_is_running", classmethod(lambda cls: False)),
            mock.patch.object(mod, "acquire_docker_lease", return_value=lease),
            mock.patch.object(mod, "remove_container_if_exists", return_value=False),
            mock.patch.object(mod.subprocess, "run", recorder),
        ):
            mod.RemoteSAMDockerManager._start_docker()

        run_cmd = next(cmd for cmd in recorder.commands if cmd[:3] == ["docker", "run", "-d"])
        self.assertIn("terrabox.service=remotesam", run_cmd)
        self.assertIn("9014:9004", run_cmd)
        self.assertEqual(mod.RemoteSAMDockerManager.API_URL, "http://127.0.0.1:9014")

    def test_strip_rcnn_docker_manager_uses_dynamic_host_port_but_internal_arg(self):
        from terrabox.managers.docker import strip_rcnn_manager as mod

        recorder = RunRecorder()
        lease = _lease("strip-rcnn", 9015, 9005)
        with (
            mock.patch.object(mod.StripRCNNDockerManager, "_container_is_running", classmethod(lambda cls: False)),
            mock.patch.object(mod, "acquire_docker_lease", return_value=lease),
            mock.patch.object(mod, "remove_container_if_exists", return_value=False),
            mock.patch.object(mod.subprocess, "run", recorder),
        ):
            mod.StripRCNNDockerManager._start_docker()

        run_cmd = next(cmd for cmd in recorder.commands if cmd[:3] == ["docker", "run", "-d"])
        self.assertIn("9015:9005", run_cmd)
        self.assertEqual(run_cmd[run_cmd.index("--port") + 1], "9005")
        self.assertEqual(mod.StripRCNNDockerManager.API_URL, "http://127.0.0.1:9015")

    def test_vllm_docker_manager_uses_resource_lease(self):
        from terrabox.managers.docker import vllm_manager as mod

        recorder = RunRecorder()
        lease = _lease("vlm", 9010, 8000)
        with (
            mock.patch.object(mod.VLLMDockerManager, "_container_is_running", classmethod(lambda cls: False)),
            mock.patch.object(mod, "acquire_docker_lease", return_value=lease),
            mock.patch.object(mod, "remove_container_if_exists", return_value=False),
            mock.patch.object(mod.subprocess, "run", recorder),
        ):
            mod.VLLMDockerManager._start_docker()

        run_cmd = next(cmd for cmd in recorder.commands if cmd[:3] == ["docker", "run", "-d"])
        self.assertIn("terrabox.service=vlm", run_cmd)
        self.assertIn("--gpus", run_cmd)
        self.assertIn("device=3", run_cmd)
        self.assertIn("CUDA_VISIBLE_DEVICES=0", run_cmd)
        self.assertIn("9010:8000", run_cmd)
        self.assertEqual(mod.VLLMDockerManager.API_BASE, "http://127.0.0.1:9010/v1")

    def test_vllm_docker_manager_adopts_reusable_container(self):
        from terrabox.managers.docker import vllm_manager as mod

        lease = _lease("vlm", 9017, 8000)
        lease = ResourceLease(**{**lease.__dict__, "reused": True})

        with (
            mock.patch.object(mod.VLLMDockerManager, "is_running", classmethod(lambda cls: False)),
            mock.patch.object(mod, "find_reusable_managed_lease", return_value=lease),
            mock.patch.object(mod.VLLMDockerManager, "_start_docker", side_effect=AssertionError("should not start")),
        ):
            mod.VLLMDockerManager.start_service()

        self.assertEqual(mod.VLLMDockerManager.CONTAINER_NAME, lease.container_name)
        self.assertEqual(mod.VLLMDockerManager.API_BASE, "http://127.0.0.1:9017/v1")

    def test_agent_llm_docker_manager_returns_dynamic_api_base(self):
        from terrabox.managers.docker import agent_llm_manager as mod

        recorder = RunRecorder()
        lease = _lease("agent-llm", 9110, 8000)
        checks = {"n": 0}

        def fake_is_running(cls):
            checks["n"] += 1
            return checks["n"] >= 2

        def fake_container_is_running(cls):
            return any(cmd[:3] == ["docker", "run", "-d"] for cmd in recorder.commands)

        with (
            mock.patch.object(mod.AgentLLMDockerManager, "is_running", classmethod(fake_is_running)),
            mock.patch.object(mod.AgentLLMDockerManager, "_container_is_running", classmethod(fake_container_is_running)),
            mock.patch.object(mod, "find_reusable_managed_lease", return_value=None),
            mock.patch.object(mod, "acquire_docker_lease", return_value=lease),
            mock.patch.object(mod, "remove_container_if_exists", return_value=False),
            mock.patch.object(mod.subprocess, "run", recorder),
        ):
            mod.AgentLLMDockerManager.start_service(config=None)

        self.assertEqual(mod.AgentLLMDockerManager._api_base(), "http://127.0.0.1:9110/v1")
        run_cmd = next(cmd for cmd in recorder.commands if cmd[:3] == ["docker", "run", "-d"])
        self.assertIn("9110:8000", run_cmd)

    def test_agent_llm_docker_manager_adopts_reusable_container(self):
        from terrabox.managers.docker import agent_llm_manager as mod

        lease = _lease("agent-llm", 9112, 8000)
        lease = ResourceLease(**{**lease.__dict__, "reused": True})

        with (
            mock.patch.object(mod.AgentLLMDockerManager, "is_running", classmethod(lambda cls: False)),
            mock.patch.object(mod.AgentLLMDockerManager, "_container_is_running", classmethod(lambda cls: False)),
            mock.patch.object(mod, "find_reusable_managed_lease", return_value=lease),
            mock.patch.object(mod, "acquire_docker_lease", side_effect=AssertionError("should not allocate")),
        ):
            mod.AgentLLMDockerManager.start_service(config=None)

        self.assertEqual(mod.AgentLLMDockerManager.CONTAINER_NAME, lease.container_name)
        self.assertEqual(mod.AgentLLMDockerManager._api_base(), "http://127.0.0.1:9112/v1")


if __name__ == "__main__":
    unittest.main()
