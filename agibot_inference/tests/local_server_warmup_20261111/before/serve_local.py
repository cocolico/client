"""Local-only entry over the unchanged admitted server; relocate asset IO only."""
from pathlib import Path
import argparse,fcntl,hashlib,json,logging,os,socket,sys,time,types

def relocate(value,rootfs):
    if isinstance(value,str) and value.startswith('/mnt/'):
        local=rootfs/value.lstrip('/')
        if local.exists():return str(local)
    if isinstance(value,dict):return {k:relocate(v,rootfs) for k,v in value.items()}
    if isinstance(value,list):return [relocate(v,rootfs) for v in value]
    return value

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--port',type=int,default=19011);p.add_argument('--check-assets',action='store_true');a=p.parse_args()
    b=a.bundle.resolve();m=json.loads((b/'BUNDLE_MANIFEST.json').read_text());rootfs=b/'rootfs'
    sys.path[:0]=[str(b/'code'),str(b/'external')]
    # Keep package-level training engines outside this inference entry.
    for name in ('openwam.deploy','openwam.dataloader'):
        module=types.ModuleType(name);module.__path__=[str((b/'code').joinpath(*name.split('.')))];sys.modules[name]=module
    from openwam.deploy.real_posttrain import runtime,aligned_runtime
    original=runtime.checked_json
    def checked_json(path,expected=None):
        path=relocate(str(path),rootfs)
        return relocate(original(path,expected),rootfs)
    runtime.checked_json=aligned_runtime.checked_json=checked_json
    from openwam.deploy.real_posttrain.midfour_final_runtime import create_runtime
    from openwam.deploy.real_posttrain.server import Server,claim_output
    checkpoint=Path(relocate(m['checkpoint'],rootfs))
    if a.check_assets:
        for row in m['payload']:
            local=b/row['relative']
            assert local.stat().st_size==row['bytes'] and runtime.sha256(local)==row['sha256'],str(local)
        assert checkpoint.stat().st_size==m['checkpoint_bytes'] and runtime.sha256(checkpoint)==m['checkpoint_sha256']
        binding=checked_json(checkpoint.parent/'data_binding.json')
        for key in ('catalog_path','text_manifest_path','normalizer_path'):assert Path(binding[key]).is_file()
        texts=checked_json(binding['text_manifest_path'])['tasks']
        assert all(Path(v['path']).is_file() for v in texts.values())
        assets=checked_json(checkpoint.parent/'pretrained_assets.json');assert Path(assets['vae']['path']).is_file()
        print(json.dumps(dict(passed=True,all_runtime_assets_local=True,checkpoint_sha256=m['checkpoint_sha256'],robot_io=False)),flush=True);return
    if not 1<=a.port<=65535:p.error('port must be1..65535')
    output=b/'logs'/(time.strftime('%Y%m%d-%H%M%S')+'-'+str(os.getpid()));output.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    with (output/'server.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);claim_output(output)
        rt=create_runtime(checkpoint,output,model='c0_lr',checkpoint_sha256=m['checkpoint_sha256'],steps=10,compile_enabled=True)
        identity=json.loads((output/'identity.json').read_text());identity['portable_entry_sha256']=runtime.sha256(__file__)
        identity['portable_manifest_sha256']=runtime.sha256(b/'BUNDLE_MANIFEST.json')
        rt.info.update(local_bundle=str(b),portable_entry_sha256=identity['portable_entry_sha256'],deployment_id=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest())
        for name,value in [('identity.json',identity),('info.json',rt.info)]: (output/name).write_text(json.dumps(value,indent=2))
        with Server(('127.0.0.1',a.port),rt) as server:
            receipt=dict(port=a.port,pod_ip='127.0.0.1',deployment_id=rt.info['deployment_id'],weights_loaded=True,robot_io=False,output=str(output))
            (output/'listening.json').write_text(json.dumps(receipt,indent=2));(b/'LATEST_SERVER.json').write_text(json.dumps(receipt,indent=2))
            print('LOCAL_MID_A1_READY '+json.dumps(receipt),flush=True)
            try:server.serve_forever()
            except KeyboardInterrupt:pass
if __name__=='__main__':main()
