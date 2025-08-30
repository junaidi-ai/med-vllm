import json
from ._helpers import run_cli


def test_list_profiles_table():
    code, out, err, _ = run_cli(["list-profiles"])  # table output
    assert code == 0, f"non-zero exit: {code}\nSTDERR:\n{err}"
    # Should include a table header and at least our known profile names
    assert "Deployment Profiles" in out
    assert "edge_cpu_int8" in out
    assert "cloud_gpu_tp4_fp16" in out


def test_list_profiles_json():
    code, out, err, _ = run_cli(["list-profiles", "--json"])  # json output
    assert code == 0, f"non-zero exit: {code}\nSTDERR:\n{err}"
    # Output should be valid JSON with profiles array and include known names
    data = json.loads(out)
    assert isinstance(data, dict) and "profiles" in data
    names = {p.get("name") for p in data["profiles"]}
    assert {"edge_cpu_int8", "cloud_gpu_tp4_fp16"}.issubset(names)
