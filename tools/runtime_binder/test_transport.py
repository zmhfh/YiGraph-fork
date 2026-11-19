import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'aag/engine'))
import runtime_binder_transport as transport


class Transport(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.worker = self.root / "worker.py"
        self.worker.write_text("import json,sys,os\nfor line in sys.stdin:\n e=json.loads(line)\n r=e['request']\n if r.get('bad'): print('unexpected output',flush=True); continue\n print(json.dumps({'request_id':e['request_id'],'result':{'input_refs':[r['ref']]}}),flush=True)\n", encoding="utf-8")
        self.env = patch.dict(os.environ, YIGRAPH_BINDER_COMMAND=json.dumps([sys.executable, str(self.worker)]),
            YIGRAPH_BINDER_LOG=str(self.root / "model.log"), YIGRAPH_BINDER_CUDA_VISIBLE_DEVICES="GPU-test", YIGRAPH_BINDER_TIMEOUT="3")
        self.env.start()

    def tearDown(self):
        transport.close()
        self.env.stop()
        self.temp.cleanup()

    def test_persistent_process_and_references(self):
        self.assertEqual(transport.request_local({"ref": "G0"}), {"input_refs": ["G0"]})
        pid = transport._process.pid
        self.assertEqual(transport.request_local({"ref": "working_graph"}), {"input_refs": ["working_graph"]})
        self.assertEqual(pid, transport._process.pid)
        self.assertEqual(transport._process.poll(), None)

    def test_bad_output_closes_worker(self):
        with self.assertRaises(ValueError):
            transport.request_local({"bad": True})
        self.assertIsNone(transport._process)

    def test_invalid_command_is_rejected(self):
        os.environ["YIGRAPH_BINDER_COMMAND"] = '"not a list"'
        with self.assertRaises(ValueError):
            transport.request_local({"ref": "G0"})

    def test_close_releases_process(self):
        transport.request_local({"ref": "G0"})
        process = transport._process
        transport.close()
        self.assertIsNotNone(process.poll())


if __name__ == "__main__":
    unittest.main()
