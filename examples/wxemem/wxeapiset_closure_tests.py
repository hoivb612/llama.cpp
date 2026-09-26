"""Standalone closure tests: python wxeapiset_closure_tests.py --exe <path> --arch x64."""

import argparse
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import unittest


def make_pe(path, imports=(), delayed=(), arch="x64"):
    """Create non-executable PE import-table fixtures; the tool must never load them."""
    data = bytearray(0x8000)
    is64 = arch == "x64"
    optional_size = 240 if is64 else 224
    struct.pack_into("<H", data, 0, 0x5A4D)
    struct.pack_into("<I", data, 0x3C, 0x80)
    struct.pack_into("<I", data, 0x80, 0x4550)
    struct.pack_into("<HHIIIHH", data, 0x84, 0x8664 if is64 else 0x14C,
                     1, 0, 0, 0, optional_size, 0x2022)
    struct.pack_into("<H", data, 0x98, 0x20B if is64 else 0x10B)
    struct.pack_into("<Q" if is64 else "<I", data, 0x98 + (24 if is64 else 28), 0x10000000)
    struct.pack_into("<I", data, 0x98 + 60, 0x200)
    directory_offset = 112 if is64 else 96
    struct.pack_into("<I", data, 0x98 + directory_offset - 4, 16)
    section = 0x98 + optional_size
    struct.pack_into("<8sIIII", data, section, b".idata", 0x7E00, 0x1000, 0x7E00, 0x200)
    cursor = 0x200

    def allocate(payload):
        nonlocal cursor
        offset = cursor
        data[offset:offset + len(payload)] = payload
        cursor = (cursor + len(payload) + 7) & ~7
        return offset, offset - 0x200 + 0x1000

    for modules, index, width in ((imports, 1, 20), (delayed, 13, 32)):
        if not modules:
            continue
        descriptors, rva = allocate(bytes((len(modules) + 1) * width))
        struct.pack_into("<II", data, 0x98 + directory_offset + index * 8,
                         rva, (len(modules) + 1) * width)
        for i, module in enumerate(modules):
            _, name_rva = allocate(module.encode("ascii") + b"\0")
            _, symbol_rva = allocate(b"\0\0Required\0")
            _, thunk_rva = allocate(struct.pack("<QQ" if is64 else "<II", symbol_rva, 0))
            if index == 1:
                struct.pack_into("<IIIII", data, descriptors + i * width,
                                 thunk_rva, 0, 0, name_rva, thunk_rva)
            else:
                struct.pack_into("<IIIIIIII", data, descriptors + i * width,
                                 1, name_rva, 0, thunk_rva, thunk_rva, 0, 0, 0)
    path.write_bytes(data)


class ClosureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="wxeapiset-closure-")
        self.root = Path(self.temp.name)
        self.cwd = self.root / "cwd"
        self.first = self.root / "first path"
        self.second = self.root / "second"
        for path in (self.cwd, self.first, self.second):
            path.mkdir()
        self.env = os.environ.copy()
        self.env["PATH"] = f'"{self.first}";;{self.second};{self.first}'

    def tearDown(self):
        self.temp.cleanup()

    def pe(self, directory, name, imports=(), delayed=(), arch=None):
        path = directory / name
        make_pe(path, imports, delayed, arch or ARGS.arch)
        return path

    def run_closure(self, binary, code=0, include_delay=False):
        mode = "--closure-delay" if include_delay else "--closure"
        result = subprocess.run([ARGS.exe, mode, str(binary)], cwd=self.cwd,
                                env=self.env, text=True, capture_output=True, timeout=20)
        self.assertEqual(code, result.returncode, result.stderr)
        paths = result.stdout.splitlines()
        self.assertEqual(paths, sorted(set(paths), key=str.casefold), result.stdout)
        self.assertTrue(all(Path(path).is_absolute() for path in paths), result.stdout)
        return {str(Path(path)).casefold() for path in paths}, result

    def test_cycle_diamond_delay_and_search_precedence(self):
        root = self.pe(self.cwd, "root.exe", ["a.dll", "B.dll"], ["late.dll"])
        a = self.pe(self.cwd, "a.dll", ["shared.dll", "root.exe"], ["child-late.dll"])
        b = self.pe(self.first, "b.dll", ["SHARED.dll"])
        shared = self.pe(self.second, "shared.dll", ["a.dll"])
        late = self.pe(self.first, "late.dll", ["late-normal.dll"], ["late-delay.dll"])
        child_late = self.pe(self.first, "child-late.dll")
        late_normal = self.pe(self.second, "late-normal.dll")
        late_delay = self.pe(self.second, "late-delay.dll", ["root.exe"])
        (self.first / "a.dll").write_bytes(b"Wrong CWD precedence")
        (self.second / "b.dll").write_bytes(b"Wrong PATH precedence")
        paths, result = self.run_closure(root)
        self.assertEqual(paths, {str(p).casefold() for p in (a, b, shared)})
        self.assertIn("Normal imports only", result.stderr)
        paths, result = self.run_closure(root, include_delay=True)
        self.assertEqual(paths, {str(p).casefold() for p in
                                (a, b, shared, late, child_late, late_normal, late_delay)})
        self.assertIn("potential delay", result.stderr)

    def test_missing_continues_and_marks_partial(self):
        root = self.pe(self.cwd, "root.exe", ["missing.dll", "ok.dll"], ["late-missing.dll"])
        ok = self.pe(self.first, "ok.dll")
        paths, result = self.run_closure(root, 1)
        self.assertEqual(paths, {str(ok).casefold()})
        self.assertIn("MISSING [normal] missing.dll", result.stderr)
        self.assertNotIn("late-missing.dll", result.stderr)
        paths, result = self.run_closure(root, 1, include_delay=True)
        self.assertEqual(paths, {str(ok).casefold()})
        self.assertIn("MISSING [normal] missing.dll", result.stderr)
        self.assertIn("MISSING [delay] late-missing.dll", result.stderr)
        self.assertIn("INCOMPLETE", result.stderr)

    def test_delay_only_failures_do_not_affect_normal_closure(self):
        root = self.pe(self.cwd, "root.exe", ["ok.dll"],
                       ["late-missing.dll", "ext-ms-test-never-present-l1-1-0.dll"])
        ok = self.pe(self.first, "ok.dll", delayed=["child-missing.dll"])
        paths, result = self.run_closure(root)
        self.assertEqual(paths, {str(ok).casefold()})
        self.assertNotIn("MISSING", result.stderr)
        self.assertNotIn("UNRESOLVED", result.stderr)
        paths, result = self.run_closure(root, 1, include_delay=True)
        self.assertEqual(paths, {str(ok).casefold()})
        self.assertIn("MISSING [delay] late-missing.dll", result.stderr)
        self.assertIn("MISSING [delay] child-missing.dll", result.stderr)
        self.assertIn("UNRESOLVED [delay] ext-ms-test-never-present", result.stderr)

    def test_malformed_delay_directory_ignored_only_in_normal_closure(self):
        root = self.pe(self.cwd, "root.exe", ["ok.dll"])
        ok = self.pe(self.first, "ok.dll")
        data = bytearray(ok.read_bytes())
        directory_offset = 112 if ARGS.arch == "x64" else 96
        struct.pack_into("<II", data, 0x98 + directory_offset + 13 * 8, 0xFFFFFF00, 32)
        ok.write_bytes(data)
        paths, _ = self.run_closure(root)
        self.assertEqual(paths, {str(ok).casefold()})
        _, result = self.run_closure(root, 2, include_delay=True)
        self.assertIn("Malformed PE: unmapped RVA", result.stderr)
        result = subprocess.run([ARGS.exe, "--bin", str(ok)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Malformed PE: unmapped RVA", result.stderr)

    def test_no_implicit_input_directory(self):
        root = self.pe(self.root, "root.exe", ["beside.dll"])
        self.pe(self.root, "beside.dll")
        paths, result = self.run_closure(root, 1)
        self.assertFalse(paths)
        self.assertIn("beside.dll", result.stderr)

    def test_root_name_search_and_empty_path(self):
        self.pe(self.first, "root.exe")
        self.run_closure("root.exe")
        self.env["PATH"] = ""
        self.pe(self.cwd, "cwd.exe")
        self.run_closure("cwd.exe")
        self.run_closure("root.exe", 2)

    def test_malformed_and_wrong_architecture_continue(self):
        root = self.pe(self.cwd, "root.exe", ["bad.dll", "other.dll", "ok.dll"])
        bad = self.first / "bad.dll"
        bad.write_bytes(b"not a PE")
        other = self.pe(self.first, "other.dll", arch="x86" if ARGS.arch == "x64" else "x64")
        ok = self.pe(self.second, "ok.dll")
        paths, result = self.run_closure(root, 2)
        self.assertEqual(paths, {str(p).casefold() for p in (bad, other, ok)})
        self.assertIn("Architecture mismatch", result.stderr)
        self.assertIn("ERROR reading", result.stderr)

    def test_whitespace_path_entries_and_padded_quotes(self):
        root = self.pe(self.cwd, "root.exe", ["first.dll", "second.dll"])
        first = self.pe(self.first, "first.dll")
        second = self.pe(self.second, "second.dll")
        self.env["PATH"] = f' ;\t;"";"   ";  "{self.first}" \t; \t{self.second}  ; '
        paths, result = self.run_closure(root)
        self.assertEqual(paths, {str(first).casefold(), str(second).casefold()})
        self.assertNotIn("GetFullPathName", result.stderr)
        self.env["PATH"] = " ; \t "
        self.pe(self.cwd, "empty-path.exe")
        self.run_closure("empty-path.exe")

    def test_bad_path_reports_offending_entry(self):
        root = self.pe(self.cwd, "root.exe")
        self.env["PATH"] = '"C:\\invalid'
        paths, result = self.run_closure(root, 2)
        self.assertFalse(paths)
        self.assertIn('malformed quoted PATH entry: "C:\\invalid', result.stderr)

    def test_cross_architecture_root_and_transitive_dependencies(self):
        arch = "x86" if ARGS.arch == "x64" else "x64"
        root = self.pe(self.cwd, "root.exe", ["first.dll"], arch=arch)
        first = self.pe(self.first, "first.dll", ["second.dll"], arch=arch)
        second = self.pe(self.second, "second.dll", ["first.dll"], arch=arch)
        paths, result = self.run_closure(root)
        self.assertEqual(paths, {str(first).casefold(), str(second).casefold()})
        self.assertIn(f"WARNING: root binary is {arch}, tool is {ARGS.arch}", result.stderr)
        self.assertIn("tool's PEB schema", result.stderr)
        self.assertNotIn("Architecture mismatch", result.stderr)

    def test_cross_architecture_dependency_must_match_root_not_tool(self):
        arch = "x86" if ARGS.arch == "x64" else "x64"
        root = self.pe(self.cwd, "root.exe", ["wrong.dll", "right.dll"], arch=arch)
        wrong = self.pe(self.first, "wrong.dll", ["not-traversed.dll"])
        right = self.pe(self.second, "right.dll", arch=arch)
        paths, result = self.run_closure(root, 2)
        self.assertEqual(paths, {str(wrong).casefold(), str(right).casefold()})
        self.assertIn(f"dependency is {ARGS.arch}, root binary requires {arch}", result.stderr)
        self.assertNotIn("not-traversed.dll", result.stderr)

    def test_contract_resolution_and_missing_contract(self):
        contract = "api-ms-win-core-synch-l1-2-0.dll"
        query = subprocess.run([ARGS.exe, contract], text=True, capture_output=True, timeout=10)
        self.assertEqual(query.returncode, 0, query.stderr)
        host = re.search(r" -> ([^\r\n]+\.dll)", query.stdout)[1]
        root = self.pe(self.cwd, "root.exe", [contract, "ext-ms-test-never-present-l1-1-0.dll"])
        mapped = self.pe(self.first, host)
        paths, result = self.run_closure(root, 1)
        self.assertEqual(paths, {str(mapped).casefold()})
        self.assertIn("UNRESOLVED", result.stderr)

    def test_directory_is_not_a_binary(self):
        root = self.pe(self.cwd, "root.exe", ["thing.dll"])
        (self.cwd / "thing.dll").mkdir()
        real = self.pe(self.first, "thing.dll")
        paths, _ = self.run_closure(root)
        self.assertEqual(paths, {str(real).casefold()})

    def test_no_arguments_and_missing_root(self):
        self.run_closure(self.root / "missing.exe", 2)
        for mode in ("--closure", "--closure-delay"):
            for args in ([], [""], ["one.exe", "two.exe"]):
                result = subprocess.run([ARGS.exe, mode, *args], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(result.stdout)
                self.assertIn(f"Usage: wxeapiset {mode}", result.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--arch", choices=("x86", "x64"), required=True)
    ARGS, remaining = parser.parse_known_args()
    ARGS.exe = str(Path(ARGS.exe).resolve())
    unittest.main(argv=[__file__, *remaining])
