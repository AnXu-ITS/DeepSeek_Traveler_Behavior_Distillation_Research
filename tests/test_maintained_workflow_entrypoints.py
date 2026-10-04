"""Offline regression checks for maintained and retired workflow entry points.

These tests never query the Teacher, read respondent inputs or run MATSim.
PowerShell checks are static; run the shell scripts on a configured host to
validate their runtime integration.
"""
import argparse
import ast
import hashlib
import json
import shutil
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'archive/development_history/author_workflows'
ORIGINAL_HASHES = {
    'scripts/revision_20260921/complete_prompt_pilot.ps1': 'f4ae8954e59bd5719d784e527116e7a34582801e81b578ec1b9d31c3f4c9e57b',
    'scripts/revision_20260921/complete_teacher.ps1': 'eb92c2876b03a2f712cf36c9cb03f60ecf77729eaee9c53285b3d47ed9f15e46',
    'scripts/revision_20260921/human_complete_resident_sensitivity.py': 'd73fffa7a517b9ebc4baa97bd6cb294e23abe12ed6122cdf10e264eb897df38c',
    'cvpr_workspace/analysis/statistics/finalize_closure.py': 'f2d51a4a99257dc46eaa3bdc37efcac445a291498cb52aa05873499f82c58dbd',
    'cvpr_workspace/analysis/statistics/seal_closure.py': '58104565f1618fe2dc24bdb1ccf1733a1806befd3ec15e366608fbbb021738ff',
}


class MaintainedWorkflowTests(unittest.TestCase):
    def test_original_sources_are_preserved_byte_for_byte(self):
        for path, expected in ORIGINAL_HASHES.items():
            with self.subTest(path=path):
                self.assertEqual(hashlib.sha256((ARCHIVE / path).read_bytes()).hexdigest(), expected)

    def test_retired_entrypoints_stop_before_any_output(self):
        for name in ('finalize_closure.py', 'seal_closure.py'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                result = subprocess.run(
                    [sys.executable, '-B', str(ROOT / 'cvpr_workspace/analysis/statistics' / name)],
                    cwd=tmp, capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn('retired', result.stderr)
                self.assertIn('No results were generated or validated', result.stderr)
                self.assertIn('python scripts/check_repository_docs.py', result.stderr)
                self.assertEqual(result.stdout, '')
                self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_retired_entrypoints_only_import_standard_library(self):
        for name in ('finalize_closure.py', 'seal_closure.py'):
            tree = ast.parse((ROOT / 'cvpr_workspace/analysis/statistics' / name).read_text())
            imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
            self.assertEqual(len(imports), 1)
            self.assertIsInstance(imports[0], ast.Import)
            self.assertEqual(imports[0].names[0].name, 'sys')

    def test_teacher_wrapper_commands_are_unchanged(self):
        for name in ('complete_prompt_pilot.ps1', 'complete_teacher.ps1'):
            path = Path('scripts/revision_20260921') / name
            before, after = (ARCHIVE / path).read_text(), (ROOT / path).read_text()
            command_lines = lambda text: [line.strip() for line in text.splitlines() if line.strip().startswith('& ')]
            self.assertEqual(command_lines(before), command_lines(after))

    def test_teacher_wrappers_use_caller_credential_without_mutating_it(self):
        for name in ('complete_prompt_pilot.ps1', 'complete_teacher.ps1'):
            text = (ROOT / 'scripts/revision_20260921' / name).read_text()
            self.assertIn('[string]::IsNullOrWhiteSpace($env:DEEPSEEK_API_KEY)', text)
            self.assertLess(text.index('IsNullOrWhiteSpace'), text.index('& $taskPython'))
            self.assertNotIn('ReadAllText', text)
            self.assertNotIn('Codex', text)
            self.assertNotIn('Remove-Item Env:', text)
            self.assertNotIn('$env:DEEPSEEK_API_KEY =', text)

    def test_human_sensitivity_calculation_is_unchanged(self):
        path = Path('scripts/revision_20260921/human_complete_resident_sensitivity.py')
        before, after = (ARCHIVE / path).read_text(), (ROOT / path).read_text()
        start = '    sources=['
        original_calculation = before[before.index(start):before.index("    (PAPER/")]
        maintained_calculation = after[after.index(start):after.index('    supplement_output.parent.mkdir')]
        self.assertEqual(original_calculation, maintained_calculation)

    def test_supplement_output_is_local_by_default_and_configurable(self):
        path = ROOT / 'scripts/revision_20260921/human_complete_resident_sensitivity.py'
        tree = ast.parse(path.read_text())
        parser = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'parse_args')
        output = Path('outputs/revision_20260921/survey/human_complete_resident_sensitivity')
        namespace = {'argparse': argparse, 'Path': Path, 'DEST': output, '__doc__': ''}
        exec(compile(ast.Module(body=[parser], type_ignores=[]), str(path), 'exec'), namespace)
        self.assertEqual(namespace['parse_args']([]).supplement_output, output / 'supp_sample_flow.tex')
        self.assertEqual(namespace['parse_args'](['--supplement-output', 'custom/fragment.tex']).supplement_output,
                         Path('custom/fragment.tex'))

    def test_e1_requires_an_explicit_manuscript_destination(self):
        env = os.environ.copy()
        env.pop('AIT_PAPER_REPO', None)
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/finalize_e1.py')],
                                    cwd=tmp, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('provide --paper-repo or set AIT_PAPER_REPO', result.stderr)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_e1_help_does_not_run_an_experiment(self):
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/finalize_e1.py'), '--help'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('AIT_PAPER_REPO', result.stdout)

    def test_survey_builder_uses_preserved_template_in_an_isolated_directory(self):
        source_dir = ROOT / 'docs/plans/shanghai_survey_v2'
        template = source_dir / 'participant_sections.source.txt'
        data = template.read_bytes()
        git_blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        self.assertEqual(git_blob, '7ec1bb6bafbc0faf6e6f24f79070f99a479419af')
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'docs/plans/shanghai_survey_v2'
            directory.mkdir(parents=True)
            shutil.copy2(source_dir / 'build_package.py', directory / 'build_package.py')
            shutil.copy2(template, directory / template.name)
            result = subprocess.run([sys.executable, '-B', str(directory / 'build_package.py')],
                                    cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(json.loads((directory / 'cards.json').read_text())['cards']), 10)
            self.assertEqual(len(json.loads((directory / 'forms.json').read_text())['forms']), 10)
            questionnaire = directory.parent / 'Shanghai_Travel_Intention_Survey.md'
            self.assertTrue(questionnaire.read_text().startswith(template.read_text()))
            self.assertEqual(len(list(directory.glob('*_template.csv'))), 4)
            for csv_file in directory.glob('*_template.csv'):
                self.assertEqual(len(csv_file.read_text(encoding='utf-8-sig').splitlines()), 1)

    def test_matsim_paths_are_configurable_and_validated(self):
        for name in ('build_matsim_launcher.ps1', 'run_matsim.ps1'):
            text = (ROOT / 'scripts' / name).read_text()
            self.assertIn('$BuildDir = $env:MATSIM_BUILD_DIR', text)
            self.assertIn('$ReleaseDir = $env:MATSIM_RELEASE_DIR', text)
            self.assertIn('[System.IO.Path]::PathSeparator', text)
            self.assertIn("'matsim-2026.0.jar'", text)
            self.assertIn('Test-Path -LiteralPath $jar -PathType Leaf', text)
            self.assertNotIn('C:\\Users\\', text)
        build = (ROOT / 'scripts/build_matsim_launcher.ps1').read_text()
        self.assertIn("'tools/java/RunMatsimPreloaded.java'", build)
        self.assertIn('Copy-Item -LiteralPath $source -Destination $stagedSource', build)
        self.assertIn('if ($LASTEXITCODE -ne 0)', build)


if __name__ == '__main__':
    unittest.main()
