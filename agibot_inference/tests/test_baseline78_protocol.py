"""Run in either client repository, with its existing TCP fixture (no hardware)."""
import os,sys,unittest
from pathlib import Path
root=Path.cwd();sys.path.insert(0,str(root/'tests'))
from test_deployed78 import Endpoint,Deployed78Tests
if os.environ.get('BASELINE_COMMON'):sys.path.insert(0,os.environ['BASELINE_COMMON'])

class BaselineNames(Deployed78Tests):
    def test_new_baseline_model_contracts(self):
        for model in ('alpha','pi05'):
            endpoint=Endpoint(8869,info_edit=lambda info:info.update(model=model))
            try:
                client=self.client(endpoint,model)
                try:self.assertEqual(client.info['model'],model)
                finally:client.close()
            finally:endpoint.close()

if __name__=='__main__':unittest.main()
