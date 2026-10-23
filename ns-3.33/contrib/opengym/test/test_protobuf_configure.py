"""Configuration regressions for mixed system/local Protobuf installations."""
from collections import defaultdict
import importlib.machinery
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

WSCRIPT = Path(__file__).resolve().parents[1] / "wscript"
loader = importlib.machinery.SourceFileLoader("opengym_wscript", str(WSCRIPT))
module = types.ModuleType(loader.name)
loader.exec_module(module)


class Env(defaultdict):
    def __init__(self):
        super().__init__(list)

    def append_unique(self, key, values):
        for value in values:
            if value not in self[key]:
                self[key].append(value)


class Conf:
    def __init__(self, directory):
        self.env = Env()
        self.env['LIBPATH'] = ['/older/system/library']
        self.path = types.SimpleNamespace(abspath=lambda: directory)
        self.probes = []

    def check(self, **kwargs):
        return True

    def check_cfg(self, **kwargs):
        self.package = kwargs

    def msg(self, *args):
        pass

    def fatal(self, message):
        raise RuntimeError(message)

    def check_cxx(self, **kwargs):
        self.probes.append(kwargs)


class ProtobufConfigurationTests(unittest.TestCase):
    def configure(self, directory, compiler_version='3.20.3', library_version='3.20.3'):
        conf = Conf(directory)
        def output(command):
            option = command[1]
            return {'--version': 'libprotoc ' + compiler_version,
                    '--modversion': library_version,
                    '--variable=includedir': directory,
                    '--variable=libdir': directory}[option].encode()
        with patch.dict(os.environ, {'PROTOC': '/selected/bin/protoc'}), \
             patch.object(module.subprocess, 'check_output', side_effect=output), \
             patch.object(module.subprocess, 'check_call') as generate:
            module.configure(conf)
        return conf, generate

    def test_selected_compiler_headers_linker_and_runtime_are_consistent(self):
        with tempfile.TemporaryDirectory(prefix='protobuf with spaces ') as folder:
            conf, generate = self.configure(folder)
            self.assertEqual(conf.env['LIBPATH'][0], folder)
            self.assertEqual(conf.env['RPATH'][0], folder)
            self.assertEqual(conf.env['LIB'], ['zmq', 'protobuf'])
            self.assertEqual(len(generate.call_args_list), 2)
            for call in generate.call_args_list:
                self.assertEqual(call.args[0][0], '/selected/bin/protoc')
                self.assertEqual(call.kwargs, {})  # argv, never shell interpolation
            self.assertIn('3020003', conf.probes[0]['fragment'])
            self.assertIn('messages.pb.cc', conf.probes[1]['fragment'])
            self.assertTrue(all(probe['execute'] for probe in conf.probes))
            self.assertNotIn('LINKFLAGS', conf.env)

    def test_mixed_compiler_and_library_fail_before_generation(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, 'Protobuf mismatch'):
                self.configure(folder, compiler_version='3.6.1')


if __name__ == '__main__':
    unittest.main()
