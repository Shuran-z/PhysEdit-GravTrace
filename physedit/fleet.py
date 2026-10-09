"""Commands on the fleet, issued from the hub: the one machine that reaches every host (ssh, or locally for "hub")."""
import os
import re
import shlex
import subprocess
import tempfile

TAG = "physedit-job:"                  # a detached job's bash process carries TAG + its name, so `ps` finds it
JOB = re.compile(re.escape(TAG) + r"([\w.-]+/[\w.-]+/[\w.-]+)@([\d,]*)")


class Host:
    def __init__(self, name, ssh=None, site=None, gpus=(), busy_mib=1500, vars=None):
        self.name, self.site, self.gpus, self.busy_mib, self.vars = name, site, list(gpus), busy_mib, vars or {}
        self.argv = shlex.split(ssh) if ssh else []          # no ssh: the hub itself

    def __repr__(self):
        return self.name

    def command(self, remote):
        """argv that runs the shell command `remote` on this host"""
        return self.argv + [remote] if self.argv else ["bash", "-c", remote]

    def sh(self, script, timeout=600, check=True):
        """Run a bash script on the host; return its stdout. (stderr goes to a file, not a pipe: an NSCC ProxyCommand
        that outlives its ssh would otherwise hold the pipe open until the timeout.)"""
        with tempfile.TemporaryFile() as err:
            p = subprocess.run(self.command("bash -s"), input=script, stdout=subprocess.PIPE, stderr=err, text=True,
                               timeout=timeout)
            if check and p.returncode:
                err.seek(0)
                raise RuntimeError(f"{self.name}: exit {p.returncode}: {err.read().decode(errors='replace').strip()[-800:]}")
        return p.stdout

    def write(self, path, text):
        q = shlex.quote(path)
        text = text if text.endswith("\n") else text + "\n"
        self.sh(f"mkdir -p $(dirname {q}) && cat > {q}.tmp <<'PHYSEDIT_EOF'\n{text}PHYSEDIT_EOF\nmv {q}.tmp {q}\n")

    def spawn(self, job, gpus, cmd, log, events):
        """Start `cmd` detached (it outlives the ssh connection). `events` gets a start line (time, host, GPUs) and an
        end line (time, exit code); the output goes to `log`."""
        wrapped = (f"echo \"start $(date +%s) $(hostname) gpu={gpus}\" >> {events}; ({cmd}) >> {log} 2>&1; rc=$?; "
                   f"echo \"end $(date +%s) rc=$rc\" >> {events}")
        self.sh(f"mkdir -p $(dirname {log}) $(dirname {events}); cd /tmp; setsid nohup bash -c {shlex.quote(wrapped)} "
                f"{TAG}{job}@{gpus} > /dev/null 2>&1 < /dev/null &\n")

    def probe(self):
        """({allowed GPU: MiB in use}, {running job: its GPUs}), or None if the host cannot be reached."""
        try:
            out = self.sh("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null; echo ---; "
                          f"ps -eo args | grep -o '{TAG}[^ ]*' | sort -u\n", timeout=90)
        except (RuntimeError, subprocess.TimeoutExpired):
            return None
        gpu, jobs = out.split("---\n")
        used = dict(map(int, l.split(",")) for l in gpu.splitlines() if l.strip())
        return {g: used[g] for g in self.gpus if g in used}, dict(m.groups() for m in map(JOB.fullmatch, jobs.split()) if m)


def copy(src, src_dir, dst, dst_dir, files, streams=6):
    """Copy `files` (relative to src_dir) to dst_dir as parallel tar streams through the hub (one NSCC stream is
    throttled to ~160 KB/s; six reach ~3 MB/s). Symlinks are copied as the files they point to. The directories are
    made first: parallel extracts creating the same directory on NFS fail on cached negative lookups."""
    dirs = sorted({os.path.dirname(f) for f in files} - {""})
    dst.sh("mkdir -p %s && cd %s && mkdir -p %s\n" % (shlex.quote(dst_dir), shlex.quote(dst_dir),
                                                        " ".join(map(shlex.quote, dirs)) or "."))
    pipes = []
    for i in range(min(streams, len(files))):
        names = tempfile.TemporaryFile()
        names.write("".join(f + "\n" for f in files[i::streams]).encode()); names.seek(0)
        a = subprocess.Popen(src.command(f"cd {src_dir} && tar -chf - -T -"), stdin=names, stdout=subprocess.PIPE)
        b = subprocess.Popen(dst.command(f"mkdir -p {dst_dir} && cd {dst_dir} && tar -xf -"), stdin=a.stdout)
        a.stdout.close()
        pipes.append((a, b))
    if [p for a, b in pipes for p in (a.wait(), b.wait()) if p]:
        raise RuntimeError(f"copy {src}:{src_dir} -> {dst}:{dst_dir} failed")


def listing(host, d, find="-type f"):
    """{path relative to d: size} for the files under d that match the find expression (symlinks followed)."""
    out = host.sh("cd %s 2>/dev/null && find -L . %s -printf '%%P %%s\\n' || true\n" % (shlex.quote(d), find))
    return dict(l.rsplit(" ", 1) for l in out.splitlines() if l.strip())


def mirror(src, src_dir, dst, dst_dir, find="-type f", streams=6):
    """Bring dst_dir up to date with src_dir: copy what is missing or differs in size. If the copy fails, files left
    incomplete are removed, so the next attempt sends them again. Returns the number of files copied."""
    have = listing(dst, dst_dir, find)
    want = listing(src, src_dir, find)
    new = sorted(f for f, size in want.items() if have.get(f) != size)
    if new:
        try:
            copy(src, src_dir, dst, dst_dir, new, streams)
        except RuntimeError:
            now = listing(dst, dst_dir, find)
            bad = [f for f in new if f in now and now[f] != want[f]]
            if bad:
                dst.sh("cd %s && rm -f %s\n" % (shlex.quote(dst_dir), " ".join(map(shlex.quote, bad))))
            raise
    return len(new)
