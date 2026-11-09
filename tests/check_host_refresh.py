import argparse
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch

torch.set_num_threads(1)
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--report',type=Path,required=True)
args=p.parse_args()

def function(path, name='_maybe_refresh'):
    tree=ast.parse(path.read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='LocksMetadataBuilder')
    method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name==name)
    mod=ast.Module(body=[method],type_ignores=[])
    space={'torch':torch,'os':os,'_seq_lens_host':lambda cam:cam.sl}
    exec(compile(mod,str(path),'exec'),space)
    return space[name]

old=function(root/'src/locks/backend/builder.py')
new=function(root/'src/locks/backend/builder.py')
single=function(root/'src/locks/backend/builder.py', '_refresh_single')
class Dummy:
    def __init__(self):
        self.block_size=16
        self._cfg=SimpleNamespace(is_mem=False)
        self._prev_seq_lens=self._prev_settled=None
        self._built_once=False
        self.calls=[]
    def _r8_refresh(self,*args):self.calls.append(('refresh',))
    def _bulk_refresh(self,st,md,n,max_fin,**kw):self.calls.append(('bulk',n,max_fin))
    def _delta_refresh(self,st,md,n,max_fin,**kw):self.calls.append(('delta',n,max_fin))
    def _tail_refresh(self,st,md,n,rows):self.calls.append(('tail',n,tuple(rows.tolist())))
    def _settled_dev(self,md,n):return None
    def _rows_to_device(self,rows):return rows

Dummy._maybe_refresh = new
checks=[]
for max_pages in (1033,2057,4105,8201,32777):
    for max_reqs in (1,2,8):
        for memory in (False,True):
            a,b=Dummy(),Dummy()
            a._cfg.is_mem=b._cfg.is_mem=memory
            st=SimpleNamespace(max_pages=max_pages,max_reqs=max_reqs,r=4,G=8,d=128)
            sequence=[(x,1) for x in range(16385,16520)]
            sequence += [(16519,1)]*3+[(0,0)]*3+[(16385,1),(16386,1),(32769,1)]
            sequence += [(x,1) for x in range(32770,32810)]
            sequence += [(32809,0),(32809,16),(32810,1),None,(16385,1),(16386,1)]
            for i,pair in enumerate(sequence):
                if pair is None:
                    cam=SimpleNamespace(sl=None,query_start_loc_cpu=None)
                else:
                    cur,qlen=pair
                    cam=SimpleNamespace(sl=torch.tensor([cur],dtype=torch.int32),
                        query_start_loc_cpu=torch.tensor([0,qlen],dtype=torch.int32))
                ra=old(a,st,None,cam,1)
                eligible=max_reqs==1 and not memory and max_pages<=8192
                rb=(single if eligible else new)(b,st,None,cam,1)
                assert ra==rb,(max_pages,max_reqs,memory,i,ra,rb)
                assert a.calls==b.calls,(max_pages,max_reqs,memory,i,a.calls,b.calls)
            checks.append(dict(max_pages=max_pages,max_reqs=max_reqs,offloaded=memory,
                steps=len(sequence),derive_flags_equal=True,refresh_actions_equal=True))
output=args.report
assert not output.exists()
output.write_text(json.dumps(dict(passed=True,checks=checks),indent=2)+'\n')
print('Passed',len(checks),'configurations and',sum(x['steps'] for x in checks),'state transitions')
