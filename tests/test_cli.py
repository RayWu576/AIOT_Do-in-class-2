"""CLI integration without modifying historical outputs."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def test_entrypoint_exports_and_never_overwrites(self):
        rng=np.random.default_rng(33);x=rng.normal(size=(40,3))
        frame=pd.DataFrame(x,columns=['area','rooms','distance'])
        frame['score']=3*x[:,0]+rng.normal(size=40)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'new.csv';frame.to_csv(source,index=False)
            out=root/'new-run'
            args=[sys.executable,'-X','utf8','main.py',str(source),'--target','score',
                  '--features','area','rooms','distance','--counts','1','--inner-folds','2','--outer-folds','2',
                  '--max-features','5','--output',str(out)]
            run=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,timeout=180)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertTrue((out/'models.joblib').exists())
            self.assertTrue((out/'linearity.png').read_bytes().startswith(b'\x89PNG'))
            self.assertTrue((out/'qq.png').exists())
            data=json.loads((out/'analysis.json').read_text(encoding='utf-8'))
            self.assertEqual(len(data['models']),5)
            prior=(out/'analysis.json').read_bytes()
            run=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,timeout=60)
            self.assertNotEqual(run.returncode,0)
            self.assertEqual((out/'analysis.json').read_bytes(),prior)


if __name__=='__main__':unittest.main()
