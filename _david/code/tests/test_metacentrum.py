from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from _david.code.metacentrum import submit
from _david.code.metacentrum.submit import (
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

    def test_defaults_are_cpu_only_and_match_the_runbook(self) -> None:
        spec = JobSpec(name="egcg_default")
        self.assertEqual(spec.ncpus, 16)
        self.assertEqual(spec.mem_gb, 32)
        self.assertEqual(spec.scratch_gb, 15)
        self.assertEqual(spec.walltime, "24:00:00")
        self.assertEqual(spec.queue, "p2e_1d")
        self.assertEqual(spec.ngpus, 0)
        self.assertIn("spec=8.0", spec.resources)
        self.assertNotIn("ngpus", spec.resources)

    def test_gpu_request_adds_ngpus_and_cap(self) -> None:
        spec = JobSpec(name="gpu_run", ngpus=1, gpu_cap="sm_89", queue="gpu_long")
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
        self.assertIn("-q p2e_1d", command)
        self.assertIn(
            f"RUN_SPEC={STORAGE}/jobs/egcg/spec.json,STORAGE_DIR={STORAGE}", command
        )
        self.assertIn(f"{STORAGE}/releases/{REVISION}/_david/code/metacentrum/", command)
        self.assertIn("job_template.sh", command)
        # The pipeline arguments stay in spec.json; qsub only carries PBS state.
        self.assertNotIn("gmean", command)

    def test_split_cli_keeps_pipeline_flags_out_of_argparse(self) -> None:
        cli, pipeline = split_cli(
            ["submit", "--name", "x", "--dry-run", "--", "--criteria", "gmean"]
        )
        self.assertEqual(cli, ["submit", "--name", "x", "--dry-run"])
        self.assertEqual(pipeline, ["--criteria", "gmean"])
        self.assertEqual(split_cli(["status"]), (["status"], []))

    def test_storage_root_uses_the_canonical_double_site_path(self) -> None:
        self.assertEqual(storage_root("kolardavidcz", "brno2"), STORAGE)
        self.assertEqual(
            storage_root("kolardavidcz", "praha1"),
            "/storage/praha1/home/kolardavidcz/DrugEx",
        )


class TestCliDefaults(unittest.TestCase):
    def test_submit_parser_defaults_to_cpu_rocs_resources(self) -> None:
        args = build_parser().parse_args(["submit", "--name", "egcg"])
        self.assertEqual(args.ncpus, 16)
        self.assertEqual(args.mem, 32)
        self.assertEqual(args.scratch, 15)
        self.assertEqual(args.walltime, "24:00:00")
        self.assertEqual(args.queue, "p2e_1d")
        self.assertEqual(args.spec, 8.0)
        self.assertEqual(args.ngpus, 0)
        self.assertFalse(args.dry_run)


class TestQstatParsing(unittest.TestCase):
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


class TestPull(unittest.TestCase):
    def test_missing_remote_output_is_reported_and_no_empty_directory_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "_david" / "output"
            with patch.object(submit, "remote", return_value=SimpleNamespace(stdout="")):
                with self.assertRaises(RuntimeError) as ctx:
                    submit.pull_results("meta", STORAGE, dest)
            self.assertIn("Nothing to pull", str(ctx.exception))
            self.assertFalse(dest.exists())


class TestJobTemplateContract(unittest.TestCase):
    """The PBS script owns the release/scratch contract; these are its invariants."""

    @classmethod
    def setUpClass(cls) -> None:
        template = Path(submit.__file__).with_name(submit.TEMPLATE_NAME)
        cls.text = template.read_text()

    def test_template_fails_fast_on_missing_pbs_environment(self) -> None:
        self.assertIn("set -euo pipefail", self.text)
        for variable in ("PBS_JOBID", "SCRATCHDIR", "STORAGE_DIR", "RUN_SPEC"):
            self.assertIn(f": \"${{{variable}:?", self.text)

    def test_template_claims_a_free_run_directory_like_the_local_pipeline(self) -> None:
        self.assertIn("date +%Y%m%d_%H%M", self.text)
        self.assertIn('suffix=2', self.text)
        self.assertIn('_${suffix}', self.text)

    def test_template_copies_back_on_every_exit_path(self) -> None:
        self.assertIn("trap cleanup EXIT", self.text)
        self.assertIn("trap 'exit 143' TERM", self.text)
        self.assertIn("trap 'exit 130' INT", self.text)
        self.assertIn("rsync -a \"$OUT_DIR/\" \"$RESULTS_DIR/\"", self.text)
        self.assertIn("submission.json", self.text)

    def test_template_passes_arguments_as_json_not_as_a_shell_string(self) -> None:
        self.assertIn("json.load(handle)", self.text)
        self.assertIn('"--output-dir"', self.text)
        self.assertNotIn("eval ", self.text)


if __name__ == "__main__":
    unittest.main()
