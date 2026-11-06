from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import io
import json
import os
import signal
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from unittest.mock import patch

from _david.code.metacentrum import job, submit
from _david.code.metacentrum.submit import (
    MEM_GB_PER_CORE,
    MODEL_STEM,
    RESERVED_FLAGS,
    JobSpec,
    build_parser,
    parse_qstat_full,
    qsub_command,
    spec_payload,
    split_cli,
    storage_root,
)

STORAGE = "/storage/brno2/brno2/kolardavidcz/DrugEx"
REVISION = "ece2f23b79d49ff9b72ef7daac827e945ada166f"

QSTAT_FULL = """
Job Id: 25249720.pbs-m1.metacentrum.cz
    Job_Name = egcg_k4_c20_gold_gmean
    Job_Owner = kolardavidcz@pbs-m1.metacentrum.cz
    resources_used.cpupercent = 1500
    resources_used.walltime = 03:11:54
    job_state = F
    queue = p2e_1d
    exec_host = elmo3-6.hw.elixir-czech.cz/0-15
    Resource_List.walltime = 24:00:00
    Resource_List.select = 1:ncpus=16:mem=32gb
    comment = Job run at Tue Nov 03 at 18:00
Job Id: 25249783.pbs-m1.metacentrum.cz
    Job_Name = egcg_raw_5k
    job_state = Q
    queue = p2e_1d
    Resource_List.walltime = 24:00:00
"""


class TestJobSpec(unittest.TestCase):
    """A spec is the only place pipeline flags and PBS resources may diverge."""

    def test_defaults_are_cpu_only_and_self_consistent(self) -> None:
        spec = JobSpec(name="egcg_default")
        self.assertGreaterEqual(spec.ncpus, 1)
        self.assertGreaterEqual(spec.mem_gb, 1)
        self.assertGreaterEqual(spec.scratch_gb, 1)
        self.assertIsNone(spec.queue)
        self.assertEqual(spec.ngpus, 0)
        self.assertIn(f"ncpus={spec.ncpus}", spec.resources)
        self.assertIn(f"mem={spec.mem_gb}gb", spec.resources)
        self.assertNotIn("ngpus", spec.resources)
        # A CPU job reaches the execution queue through the walltime router.
        self.assertEqual(
            spec.queue_label, f"{submit.routed_queue(spec.walltime)} (default router)"
        )

    def test_walltime_selects_the_routed_queue(self) -> None:
        """Frontends cannot submit to execution queues directly; walltime routes the job."""
        cases = {
            "00:30:00": "q_2h",
            "02:00:00": "q_2h",
            "04:00:00": "q_4h",
            "12:00:00": "q_1d",
            "24:00:00": "q_1d",
            "48:00:00": "q_2d",
            "168:00:00": "q_1w",
            "300:00:00": "q_2w",
        }
        for walltime, queue in cases.items():
            self.assertEqual(submit.routed_queue(walltime), queue, walltime)
        with self.assertRaises(ValueError):
            submit.routed_queue("900:00:00")
        with self.assertRaises(ValueError):
            JobSpec(name="bad", walltime="1 day")

    def test_explicit_queue_overrides_the_router(self) -> None:
        spec = JobSpec(name="gpu_run", ngpus=1, gpu_cap="sm_89", queue="gpu_long")
        self.assertEqual(spec.queue_label, "gpu_long")
        self.assertIn("ngpus=1", spec.resources)
        self.assertIn("gpu_cap=sm_89", spec.resources)

    def test_spec_score_can_be_omitted(self) -> None:
        self.assertNotIn("spec=", JobSpec(name="plain", spec=None).resources)

    def test_invalid_names_are_rejected(self) -> None:
        for name in ("", "1leading_digit", "_underscore", "has space", "x" * 65, "a/b"):
            with self.assertRaises(ValueError, msg=name):
                JobSpec(name=name)

    def test_invalid_resources_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            JobSpec(name="bad", ncpus=0)
        with self.assertRaises(ValueError):
            JobSpec(name="bad", ngpus=-1)

    def test_cores_cannot_be_raised_without_memory(self) -> None:
        """A 32-core job on 32 GB is an OOM kill after the queue wait, not a speed-up.

        The scoring pool keeps ~2 GB per worker and the 16-core arms already reached
        their 32 GB ceiling, so ``--ncpus`` and ``--mem`` move together.
        """
        with self.assertRaises(ValueError):
            JobSpec(name="starved", ncpus=32, mem_gb=32)
        spec = JobSpec(name="wide", ncpus=32, mem_gb=MEM_GB_PER_CORE * 32)
        self.assertIn("mem=64gb", spec.resources)
        self.assertIn(f"ncpus={JobSpec.ncpus}", JobSpec(name="egcg_default").resources)

    def test_template_owned_flags_are_rejected_in_every_spelling(self) -> None:
        for flag in RESERVED_FLAGS:
            for args in ((flag, "value"), (f"{flag}=value",)):
                with self.assertRaises(ValueError, msg=f"{flag} {args}"):
                    JobSpec(name="reserved", args=args)


class TestSubmissionPayload(unittest.TestCase):
    def test_spec_payload_appends_the_staged_checkpoint_last(self) -> None:
        spec = JobSpec(name="egcg", args=("--criteria", "gmean"))
        payload = spec_payload(spec, STORAGE, REVISION)
        self.assertEqual(payload["revision"], REVISION)
        self.assertEqual(payload["source"], f"{STORAGE}/releases/{REVISION}")
        args = payload["args"]
        self.assertEqual(args[:2], ["--criteria", "gmean"])
        self.assertEqual(
            args[2:],
            [
                "--agent-path",
                f"{STORAGE}/models/{MODEL_STEM}.pkg",
                "--vocab-path",
                f"{STORAGE}/models/{MODEL_STEM}.vocab",
            ],
        )

    def test_qsub_command_names_spec_release_and_template(self) -> None:
        spec = JobSpec(name="egcg", args=("--criteria", "gmean"))
        argv = qsub_command(spec, STORAGE, REVISION)
        self.assertEqual(argv[:2], ["ssh", "meta"])
        command = argv[2]
        self.assertIn("-N egcg", command)
        # No -q by default: the frontend may only reach the router.
        self.assertNotIn("-q ", command)
        # Every variable the launcher reads must be exported, but new ones may be added.
        for assignment in (
            f"RUN_SPEC={STORAGE}/jobs/egcg/spec.json",
            f"STORAGE_DIR={STORAGE}",
            f"RELEASE_DIR={STORAGE}/releases/{REVISION}",
        ):
            self.assertIn(assignment, command)
        self.assertIn(f"{STORAGE}/releases/{REVISION}/_david/code/metacentrum/", command)
        self.assertIn("pbs_launcher.sh", command)
        # The pipeline arguments stay in spec.json; qsub only carries PBS state.
        self.assertNotIn("gmean", command)

    def test_explicit_queue_reaches_the_qsub_line(self) -> None:
        command = qsub_command(JobSpec(name="gpu", queue="gpu_long"), STORAGE, REVISION)[2]
        self.assertIn("-q gpu_long", command)

    def test_split_cli_keeps_pipeline_flags_out_of_argparse(self) -> None:
        cli, pipeline = split_cli(["submit", "--name", "x", "--", "--criteria", "gmean"])
        self.assertEqual(cli, ["submit", "--name", "x"])
        self.assertEqual(pipeline, ["--criteria", "gmean"])
        self.assertEqual(split_cli(["status"]), (["status"], []))

    def test_a_quoted_blob_after_double_dash_is_refused(self) -> None:
        """Job 25256773 failed because the whole tail arrived as one argv element."""
        with self.assertRaises(SystemExit) as ctx:
            submit._assert_unquoted_pipeline_tokens(
                ["--molecule CC(O)C --epochs 1 --samples 10"]
            )
        self.assertIn("separate arguments", str(ctx.exception))
        submit._assert_unquoted_pipeline_tokens(
            ["--molecule", "CC(O)C", "--epochs", "1", "--run-tag", "my run"]
        )

    def test_storage_root_uses_the_canonical_double_site_path(self) -> None:
        self.assertEqual(storage_root("kolardavidcz", "brno2"), STORAGE)
        self.assertEqual(
            storage_root("kolardavidcz", "praha1"),
            "/storage/praha1/home/kolardavidcz/DrugEx",
        )


class TestCliDefaults(unittest.TestCase):
    def test_there_is_no_preview_mode(self) -> None:
        """A submission is always a real submission; printing the qsub line was removed."""
        self.assertFalse(hasattr(build_parser().parse_args(["submit", "--name", "x"]), "dry_run"))
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["submit", "--name", "x", "--dry-run"])


class TestQstatParsing(unittest.TestCase):
    def test_list_jobs_resolves_ids_before_asking_for_long_format(self) -> None:
        """`qstat -u user -f` returns the short table on this PBS; ids must be expanded."""
        long_format = (
            "Job Id: 25256773.pbs-m1.metacentrum.cz\n"
            "    Job_Name = drugex_smoke\n"
            "    job_state = R\n"
            "    queue = p2e_1d\n"
        )
        calls: list[str] = []

        def fake_remote(host: str, command: str, stdin=None):
            calls.append(command)
            stdout = "25256773.pbs-m1.metacentrum.cz\n" if command.startswith("qselect") else long_format
            return SimpleNamespace(stdout=stdout)

        with patch.object(submit, "remote", side_effect=fake_remote):
            jobs = submit.list_jobs("meta", "kolardavidcz")
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].name, "drugex_smoke")
        self.assertEqual(jobs[0].state, "R")
        self.assertEqual(calls[0], "qselect -u kolardavidcz")
        self.assertIn("qstat -f 25256773.pbs-m1.metacentrum.cz", calls[1])
        self.assertNotIn("-u kolardavidcz -f", calls[1])

    def test_empty_queue_needs_no_second_call(self) -> None:
        with patch.object(submit, "remote", return_value=SimpleNamespace(stdout="")) as call:
            self.assertEqual(submit.list_jobs("meta", "kolardavidcz"), [])
        self.assertEqual(call.call_count, 1)

    def test_long_format_is_parsed_without_column_assumptions(self) -> None:
        jobs = parse_qstat_full(QSTAT_FULL)
        self.assertEqual([job.short_id for job in jobs], ["25249720", "25249783"])
        first = jobs[0]
        self.assertEqual(first.name, "egcg_k4_c20_gold_gmean")
        self.assertEqual(first.state, "F")
        self.assertEqual(first.queue, "p2e_1d")
        self.assertEqual(first.node, "elmo3-6.hw.elixir-czech.cz")
        self.assertEqual(first.elapsed, "03:11:54")
        self.assertEqual(first.walltime, "24:00:00")
        self.assertFalse(first.running)
        self.assertEqual(jobs[1].state, "Q")
        self.assertEqual(jobs[1].node, "")

    def test_empty_output_yields_no_jobs(self) -> None:
        self.assertEqual(parse_qstat_full(""), [])


class TestRemotePlumbing(unittest.TestCase):
    def test_binary_stdin_is_not_run_in_text_mode(self) -> None:
        """`git archive` is a byte stream; text=True would raise on the tarball."""
        completed = SimpleNamespace(
            args=["ssh"], returncode=0, stdout=b"staged", stderr=b""
        )
        with patch.object(submit.subprocess, "run", return_value=completed) as run:
            result = submit.remote("meta", "tar -xf -", stdin=b"tarball")
        self.assertFalse(run.call_args.kwargs["text"])
        self.assertEqual(result.stdout, "staged")

    def test_text_stdin_stays_text(self) -> None:
        completed = SimpleNamespace(
            args=["ssh"], returncode=0, stdout="ok", stderr=""
        )
        with patch.object(submit.subprocess, "run", return_value=completed) as run:
            result = submit.remote("meta", "cat > spec.json", stdin="{}")
        self.assertTrue(run.call_args.kwargs["text"])
        self.assertEqual(result.stdout, "ok")

    def test_kerberos_noise_becomes_an_actionable_error(self) -> None:
        completed = SimpleNamespace(
            args=["ssh"],
            returncode=255,
            stdout="",
            stderr="qstat: cannot connect to server pbs-m1.metacentrum.cz (errno=15010)",
        )
        with patch.object(submit.subprocess, "run", return_value=completed):
            with self.assertRaises(RuntimeError) as ctx:
                submit.remote("meta", "qstat")
        self.assertIn("kinit", str(ctx.exception))


class TestLogPath(unittest.TestCase):
    """A reused job name must not make `logs` show an earlier attempt's failure."""

    def test_live_job_is_resolved_by_its_id(self) -> None:
        live = submit.RemoteJob(job_id="25256877.pbs-m1.metacentrum.cz", name="egcg", state="R")
        with (
            patch.object(submit, "list_jobs", return_value=[live]),
            patch.object(
                submit, "remote", return_value=SimpleNamespace(stdout="/home/u/egcg.o25256877\n")
            ) as call,
        ):
            self.assertEqual(submit.log_path("meta", "egcg"), "/home/u/egcg.o25256877")
        self.assertIn("egcg.o25256877*", call.call_args.args[1])

    def test_live_job_without_a_log_yet_shows_nothing(self) -> None:
        live = submit.RemoteJob(job_id="25256877.pbs-m1.metacentrum.cz", name="egcg", state="R")
        with (
            patch.object(submit, "list_jobs", return_value=[live]),
            patch.object(submit, "remote", return_value=SimpleNamespace(stdout="")),
        ):
            self.assertEqual(submit.log_path("meta", "egcg"), "")

    def test_finished_job_falls_back_to_the_newest_file(self) -> None:
        with (
            patch.object(submit, "list_jobs", return_value=[]),
            patch.object(
                submit, "remote", return_value=SimpleNamespace(stdout="/home/u/egcg.o25256872\n")
            ),
        ):
            self.assertEqual(submit.log_path("meta", "egcg"), "/home/u/egcg.o25256872")


class TestPull(unittest.TestCase):
    def test_missing_remote_output_is_reported_and_no_empty_directory_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "_david" / "output"
            with patch.object(submit, "remote", return_value=SimpleNamespace(stdout="")):
                with self.assertRaises(RuntimeError) as ctx:
                    submit.pull_results("meta", STORAGE, dest)
            self.assertIn("Nothing to pull", str(ctx.exception))
            self.assertFalse(dest.exists())


class TestLauncher(unittest.TestCase):
    """The PBS entry point is a shim; job logic must not creep back into bash."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.text = Path(submit.__file__).with_name(submit.LAUNCHER_NAME).read_text()

    def test_launcher_is_a_strict_exec_shim(self) -> None:
        self.assertIn("set -euo pipefail", self.text)
        # A missing environment must abort before any expensive work starts.
        for variable in ("STORAGE_DIR", "RUN_SPEC", "RELEASE_DIR"):
            self.assertIn(f': "${{{variable}:?', self.text)
        self.assertIn('exec "$PYTHON"', self.text)
        self.assertIn("metacentrum/job.py", self.text)
        self.assertIn('"$STORAGE_DIR/.venv/bin/python"', self.text)

    def test_launcher_never_derives_the_release_from_its_own_path(self) -> None:
        # Regression pin: PBS spools the script to /var/spool/pbs/..., so
        # dirname(BASH_SOURCE) resolved to /var/spool/_david and job.py vanished
        # (jobs 25256872/25256873 died in one second on 2026-11-04).
        for marker in ("BASH_SOURCE", "dirname"):
            self.assertNotIn(marker, self.text)
        self.assertIn("${RELEASE_DIR:?", self.text)


class TestJobBody(unittest.TestCase):
    """job.py is the PBS body: stdlib-only, so its contract is unit-tested, not grepped."""

    def test_job_body_does_not_import_the_project(self) -> None:
        source = Path(job.__file__).read_text()
        for marker in ("from _david", "import _david", "import torch", "import rdkit"):
            self.assertNotIn(marker, source)

    def test_release_completeness_check_points_at_a_module_that_exists(self) -> None:
        """The check is a literal path, so a module move must move it too.

        The 2026-11-06 cleanup moved the pipeline into ``training/``; the stale path
        only surfaced on the next submitted job, which died after 37 s.
        """
        from _david.code.config import PROJECT_ROOT

        self.assertTrue((PROJECT_ROOT / job.PIPELINE_ENTRY).is_file(), job.PIPELINE_ENTRY)

    def test_spec_round_trips_from_submit_to_job(self) -> None:
        spec = JobSpec(name="egcg", args=("--criteria", "gmean"))
        payload = spec_payload(spec, STORAGE, REVISION)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "spec.json"
            path.write_text(json.dumps(payload))
            self.assertEqual(job.read_spec(path), payload)

    def test_spec_missing_or_malformed_fields_are_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "spec.json"
            path.write_text('{"name": "x"}')
            with self.assertRaises(job.JobError) as ctx:
                job.read_spec(path)
            self.assertIn("args, source, revision", str(ctx.exception))
            path.write_text('{"name": "x", "args": [1], "source": "s", "revision": "r"}')
            with self.assertRaises(job.JobError):
                job.read_spec(path)
            with self.assertRaises(job.JobError):
                job.read_spec(Path(tmp) / "nope.json")

    def test_missing_pbs_environment_is_one_actionable_line(self) -> None:
        with patch.dict(os.environ, {}, clear=True), redirect_stderr(io.StringIO()) as err:
            self.assertEqual(job.main([]), 1)
        self.assertIn("Missing PBS environment", err.getvalue())

    def test_run_id_matches_the_local_scheme_and_avoids_collisions(self) -> None:
        when = datetime(2026, 11, 4, 15, 30)
        with tempfile.TemporaryDirectory() as tmp:
            storage, scratch = Path(tmp) / "storage", Path(tmp) / "scratch"
            (storage / "_david" / "output").mkdir(parents=True)
            results, out = job.results_layout(storage, scratch, "egcg", when)
            self.assertEqual(results.name, "20261104_1530_egcg")
            self.assertEqual(out, scratch / "output" / results.name)
            results.mkdir()
            again, _ = job.results_layout(storage, scratch, "egcg", when)
            self.assertEqual(again.name, "20261104_1530_egcg_2")

    def test_core_count_comes_from_the_node_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            nodefile = Path(tmp) / "nodefile"
            nodefile.write_text("n1\nn1\nn1\nn1\n")
            self.assertEqual(job.job_cpus({"PBS_NODEFILE": str(nodefile), "NCPUS": "1"}), "4")
            self.assertEqual(job.job_cpus({"NCPUS": "1"}), "1")
            self.assertEqual(job.job_cpus({}), "?")

    def test_pipeline_command_owns_the_output_directory(self) -> None:
        spec = {"name": "x", "args": ["--criteria", "gmean"], "source": "/r", "revision": "abc"}
        command = job.pipeline_command(Path("/py"), Path("/out"), spec)
        self.assertEqual(command[:5], ["/py", "-u", "-m", "_david.code.training.pipeline", "--output-dir"])
        self.assertEqual(command[5], "/out")
        self.assertEqual(command[6:], ["--criteria", "gmean"])

    def test_worker_environment_pins_threads_and_the_scratch_release(self) -> None:
        env = job.job_environment(
            {"PYTHONPATH": "/elsewhere"}, Path("/scratch/DrugEx"), {"revision": "abc"}
        )
        self.assertEqual(env["MPLBACKEND"], "Agg")
        self.assertEqual(env["DRUGEX_MP_CONTEXT"], "forkserver")
        self.assertEqual(env["DRUGEX_SOURCE_REVISION"], "abc")
        self.assertEqual(env["PYTHONPATH"], "/scratch/DrugEx:/elsewhere")

    def test_prepare_run_copies_the_release_and_the_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            release = Path(tmp) / "release"
            (release / "_david" / "code").mkdir(parents=True)
            (release / "_david" / "code" / "pipeline.py").write_text("# code\n")
            spec_path = Path(tmp) / "spec.json"
            spec_path.write_text("{}\n")
            out_dir = Path(tmp) / "out" / "run"
            copy = Path(tmp) / "scratch" / "DrugEx"
            job.prepare_run(release, copy, spec_path, out_dir)
            self.assertTrue((copy / "_david" / "code" / "pipeline.py").is_file())
            self.assertEqual((out_dir / "submission.json").read_text(), "{}\n")

    def test_pipeline_output_is_mirrored_to_stdout_and_the_log(self) -> None:
        command = [
            sys.executable,
            "-c",
            "import sys; print('out'); print('err', file=sys.stderr); raise SystemExit(3)",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "execution.log"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                status = job.run_pipeline(command, Path(tmp), dict(os.environ), log_path)
            self.assertEqual(status, 3)
            self.assertIn("out", stdout.getvalue())
            self.assertIn("err", stdout.getvalue())
            log = log_path.read_text()
            self.assertIn("out", log)
            self.assertIn("err", log)

    def test_finalize_copies_results_then_cleans_scratch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scratch = Path(tmp) / "scratch"
            out_dir = scratch / "output" / "run"
            out_dir.mkdir(parents=True)
            (out_dir / "manifest.json").write_text("{}")
            (scratch / "DrugEx").mkdir()
            results = Path(tmp) / "storage" / "output" / "run"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(job.finalize(out_dir, results, scratch, 7), 7)
            self.assertEqual((results / "manifest.json").read_text(), "{}")
            self.assertFalse((scratch / "DrugEx").exists())

    def test_failed_transfer_keeps_scratch_and_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scratch = Path(tmp) / "scratch"
            out_dir = scratch / "output" / "run"
            out_dir.mkdir(parents=True)
            (out_dir / "manifest.json").write_text("{}")
            results = Path(tmp) / "storage" / "output" / "run"
            results.parent.mkdir(parents=True)
            results.write_text("a file where the directory should be")
            with redirect_stderr(io.StringIO()) as err:
                status = job.finalize(out_dir, results, scratch, 0)
            self.assertEqual(status, 1)
            self.assertIn("scratch kept", err.getvalue())
            self.assertTrue(out_dir.exists())

    def test_main_orchestrates_release_run_and_copy_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            storage, scratch, release = root / "storage", root / "scratch", root / "release"
            (storage / "_david" / "output").mkdir(parents=True)
            (storage / ".venv" / "bin").mkdir(parents=True)
            (storage / ".venv" / "bin" / "python").symlink_to(sys.executable)
            (release / job.PIPELINE_ENTRY).parent.mkdir(parents=True)
            (release / job.PIPELINE_ENTRY).write_text("# code\n")
            spec = root / "spec.json"
            spec.write_text(
                json.dumps({"name": "egcg", "args": [], "source": str(release), "revision": "abc"})
            )
            nodefile = root / "nodefile"
            nodefile.write_text("n1\nn1\n")
            env = {
                "PBS_JOBID": "1.pbs-m1",
                "SCRATCHDIR": str(scratch),
                "STORAGE_DIR": str(storage),
                "RUN_SPEC": str(spec),
                "PBS_NODEFILE": str(nodefile),
            }
            output = io.StringIO()
            with (
                patch.dict(os.environ, env, clear=True),
                patch.object(job, "run_pipeline", return_value=0) as run,
                redirect_stdout(output),
            ):
                self.assertEqual(job.main(["--release", str(release)]), 0)
            self.assertIn("cpus       : 2", output.getvalue())
            self.assertEqual(run.call_args.kwargs["cwd"], scratch / "DrugEx")
            commands = run.call_args.args[0]
            self.assertIn("--output-dir", commands)
            self.assertTrue(str(commands[commands.index("--output-dir") + 1]).startswith(str(scratch)))
            results = list((storage / "_david" / "output").iterdir())
            self.assertEqual(len(results), 1)
            self.assertTrue((results[0] / "submission.json").is_file())
            self.assertFalse((scratch / "DrugEx").exists())

    def test_walltime_and_interrupt_become_shell_exit_codes(self) -> None:
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            job.install_signal_handlers()
            for signum, code in ((signal.SIGTERM, 143), (signal.SIGINT, 130)):
                handler = signal.getsignal(signum)
                with self.assertRaises(SystemExit) as ctx:
                    handler(signum, None)
                self.assertEqual(ctx.exception.code, code)
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


if __name__ == "__main__":
    unittest.main()
