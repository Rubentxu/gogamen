#!/usr/bin/env python3
import unittest, import_backlog as b

class Seed(unittest.TestCase):
    def setUp(self): self.tasks=b.load('tasks.json')
    def test_graph(self):
        c=b.validate(self.tasks); self.assertEqual(sum(c.values()),117)
        self.assertEqual(sum(len(x['acceptance']) for x in self.tasks),351)
    def test_flags(self):
        for t in self.tasks:
            ids={k:f'ACTUAL-{i}' for i,k in enumerate(t['dependencies'])}
            a=b.argv_for(t,ids)
            self.assertEqual(a.count('--ac'),3); self.assertEqual(a.count('--dod'),2)
            if ids: self.assertEqual(a[a.index('--dep')+1],','.join(ids.values()))
            else: self.assertNotIn('--dep',a)
            self.assertNotIn('--check-ac',a); self.assertNotIn('--check-dod',a)
    def test_no_cycle(self):
        bad=[dict(self.tasks[0],dependencies=[self.tasks[1]['key']]),*self.tasks[1:]]
        with self.assertRaises(AssertionError): b.validate(bad)
    def test_revision(self):
        keys={t['key'] for t in self.tasks}
        self.assertTrue({f'NOE-{i:03}' for i in range(1,118)}==keys)
        by={t['key']:t for t in self.tasks}
        def closure(k):
            out=set(by[k]['dependencies'])
            for dep in by[k]['dependencies']: out |= closure(dep)
            return out
        # Release must require core capabilities and Kotlin parity, not optional providers/hardware.
        deps=closure('NOE-054')
        self.assertTrue({f'NOE-{i:03}' for i in (63,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,93,94,95,96,97,98,99,100,101,102,103,104,108,109,110,113,117)} <= deps)
        self.assertFalse({f'NOE-{i:03}' for i in (29,43,45,47,50,90,91,92,105,106,107,111,112,114,115,116)} & deps)
    def test_docs(self):
        for d in b.load('documents.json'):
            self.assertIn(d['kind'],('guide','other','specification','readme'))
            self.assertTrue((b.HERE/'seed/docs'/d['file']).read_text().strip())
        self.assertTrue(all(d['status']=='proposed' for d in b.load('decisions.json')))
    def test_delta(self):
        delta=b.load('v2-reconciliation.json')
        self.assertEqual(len(delta['newKeys']),25)
        self.assertEqual(len(delta['changedTasks']),6)
        by={t['key']:t for t in self.tasks}
        self.assertEqual(set(delta['newKeys']),{f'NOE-{i:03}' for i in range(93,118)})
        for ch in delta['changedTasks']:
            self.assertEqual(ch['after'],by[ch['key']]); self.assertNotEqual(ch['before'],ch['after'])
    def test_preflight(self):
        t=self.tasks[0]
        v=dict(id='ACTUAL-1',title=b.title(t),dependencies=[],description=b.desc(t),acceptanceCriteria=[{'text':c,'checked':True} for c in t['acceptance']],definitionOfDone=[{'text':d} for d in t['dod']])
        class Fake:
            def json(self,*args): return {'task':v}
        b.preflight_existing(Fake(),self.tasks,[v])
        v['description']='modified'
        with self.assertRaises(RuntimeError): b.preflight_existing(Fake(),self.tasks,[v])
    def test_duplicate(self):
        t={'title':'[NOE-001] A','id':'A-1'}
        with self.assertRaises(RuntimeError): b.match([t,t],'NOE-001')
    def test_native_docs_and_decisions_mock(self):
        class Fake:
            def __init__(self): self.docs=[]; self.decisions=[]; self.updated=[]
            def json(self,kind,*args):
                self.assert_kind=kind
                if kind=='decision-list': return {'decisions':self.decisions}
                raise AssertionError(kind)
            def run(self,*args):
                if args[:2]==('doc','list'): return '\n'.join(self.docs)
                if args[:2]==('doc','create'):
                    ident=f'doc-{len(self.docs)+1}'; self.docs.append(ident+' '+args[2]); return ident
                if args[:2]==('doc','update'):
                    self.updated.append(args[2]); return ''
                if args[:2]==('decision','create'):
                    assert args[args.index('-s')+1]=='proposed'
                    self.decisions.append({'id':f'decision-{len(self.decisions)+1}','title':args[2]}); return ''
                raise AssertionError(args)
        fake=Fake()
        for _ in range(2):
            b.docs(fake,b.load('documents.json')); b.decisions(fake,b.load('decisions.json'))
        self.assertEqual(len(fake.docs),6); self.assertEqual(len(fake.updated),6)
        self.assertEqual(len(fake.decisions),13)
    def test_mock_import(self):
        class Fake:
            def __init__(self): self.items=[]; self.created=0
            def list(self): return self.items
            def json(self,kind,*args):
                if kind=='search': return {'results':[]}
                if kind=='task-view': return {'task':next(t for t in self.items if t['id']==args[-1])}
                raise AssertionError(kind)
            def run(self,*args):
                self.created+=1
                self.items.append({'id':f'ACTUAL-{self.created}','title':args[2], 'dependencies':args[args.index('--dep')+1].split(',') if '--dep' in args else []})
        c=Fake()
        for _ in range(2):
            ids={}
            for t in self.tasks: b.reconcile_task(c,t,ids)
            self.assertEqual(len(ids),117)
        self.assertEqual(c.created,117)

if __name__=='__main__': unittest.main()
