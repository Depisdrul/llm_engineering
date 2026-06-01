#!/usr/bin/env python3
"""Kill all Gradio processes running on ports 7860-7900."""

import platform
import subprocess
import sys

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    import os
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    else:
        # Fallback for older Python versions
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


def get_processes_on_ports(start_port: int = 7860, end_port: int = 7900) -> dict[int, int]:
    """Get PIDs using ports in the range. Returns {port: pid}."""
    system = platform.system()
    port_to_pid = {}

    if system == "Windows":
        try:
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                text=True,
                check=True,
            )
            for line in result.stdout.splitlines():
                if "LISTENING" in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        addr = parts[1]
                        pid = parts[4]
                        if ":" in addr:
                            try:
                                port = int(addr.split(":")[-1])
                                if start_port <= port <= end_port:
                                    port_to_pid[port] = int(pid)
                            except (ValueError, IndexError):
                                continue
        except Exception as e:
            print(f"❌ Error getting processes: {e}")
            return {}

    else:  # Unix-like (macOS, Linux)
        try:
            result = subprocess.run(
                ["lsof", "-i", f":{start_port}-{end_port}", "-sTCP:LISTEN", "-t"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    try:
                        pid = int(line.strip())
                        # Get port for this PID
                        port_result = subprocess.run(
                            ["lsof", "-a", "-p", str(pid), "-i", "-sTCP:LISTEN"],
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        for port_line in port_result.stdout.splitlines()[1:]:
                            if ":" in port_line:
                                try:
                                    port = int(port_line.split(":")[1].split()[0])
                                    if start_port <= port <= end_port:
                                        port_to_pid[port] = pid
                                except (ValueError, IndexError):
                                    continue
                    except ValueError:
                        continue
        except FileNotFoundError:
            print("❌ 'lsof' command not found (Unix/macOS)")
            return {}
        except Exception as e:
            print(f"❌ Error getting processes: {e}")
            return {}

    return port_to_pid


def kill_process(pid: int) -> bool:
    """Kill a process by PID. Returns True if successful."""
    system = platform.system()

    try:
        if system == "Windows":
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], check=True, capture_output=True)
        else:
            subprocess.run(["kill", "-9", str(pid)], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False
    except Exception:
        return False


def main():
    """Kill all Gradio processes."""
    print("🔍 Searching for Gradio processes on ports 7860-7900...")

    port_to_pid = get_processes_on_ports()

    if not port_to_pid:
        print("✅ No Gradio processes found")
        return

    ports = sorted(port_to_pid.keys())
    pids = set(port_to_pid.values())

    print(f"📍 Found Gradio on ports: {', '.join(map(str, ports))}")
    print(f"🎯 Killing {len(pids)} process(es)...")

    success_count = 0
    for pid in pids:
        if kill_process(pid):
            success_count += 1

    if success_count == len(pids):
        print(f"✅ Successfully killed {success_count} process(es)!")
    else:
        print(f"⚠️  Killed {success_count}/{len(pids)} process(es)")


if __name__ == "__main__":
    main()
