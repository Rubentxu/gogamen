#!/usr/bin/env python3
"""One-shot seed importer. No filesystem writes in Backlog-managed directories."""
import argparse, collections, json, pathlib, re, shlex, shutil, subprocess, sys
HERE=pathlib.Path(__file__).resolve().parent

def load(name): return json.loads((HERE/'seed'/name).read_text(encoding='utf-8'))
def validate(tasks):
    seen=set()
    for t in tasks:
        k=t['key']; assert re.fullmatch(r'NOE-\d{3}',k) and k not in seen,k
        assert set(t['dependencies'])<=seen,(k,t['dependencies'])
        assert len(set(t['dependencies']))==len(t['dependencies']),k
        assert len(t['acceptance'])>=2 and all(t['acceptance']) and t['dod'],k
        assert t['size'] in ('sprint','standard','endurance') and t['priority'] in ('high','medium','low'),k
        seen.add(k)
    return collections.Counter(t['phase'] for t in tasks)

def title(t): return f"[{t['key']}] {t['title']}"
def desc(t):
    minutes={'sprint':15,'standard':30,'endurance':50}[t['size']]
    return f"{t['description']}\n\nSlice orientativo: {t['size']} (≤{minutes} min, no SLA). Contexto: {t['phase']}. Design key: {t['key']} (no es ID Backlog). Refinar el alcance antes de delegar si se estima oversized."
def argv_for(t,ids):
    cmd=['task','create',title(t),'-d',desc(t),'--priority',t['priority'],'-l',f"noesis,phase:{t['phase'].lower()}"]
    for ac in t['acceptance']: cmd+=['--ac',ac]
    for dod in t['dod']: cmd+=['--dod',dod]
    if t['dependencies']: cmd+=['--dep',','.join(ids[d] for d in t['dependencies'])]
    return cmd

class CLI:
    def __init__(self,root): self.root=root
    def run(self,*args):
        p=subprocess.run(['backlog',*args],cwd=self.root,text=True,encoding='utf-8',capture_output=True)
        if p.returncode: raise RuntimeError(f"{shlex.join(['backlog',*args[:3]])} falló ({p.returncode}): {p.stderr.strip()}\n{p.stdout.strip()}")
        return p.stdout.strip()
    def json(self,kind,*args):
        j=json.loads(self.run(*args,'--json'))
        if j.get('schemaVersion')!=1 or j.get('kind')!=kind: raise ValueError(f'Esquema inesperado: {kind}')
        return j
    def list(self): return self.json('task-list','task','list')['tasks']

def match(items,key):
    results=[x for x in items if str(x.get('title','')).startswith('['+key+'] ')]
    if len(results)>1: raise RuntimeError(f'Duplicado {key}: reconciliar por CLI')
    return results[0] if results else None

def reconcile_task(cli,t,ids):
    found=match(cli.list(),t['key'])
    # Search may also surface archived records; do not resurrect them.
    search=cli.json('search','search',t['key'])['results']
    archived=[x.get('data',{}) for x in search if x.get('type')=='task' and x.get('data',{}).get('archived')]
    if match(archived,t['key']): raise RuntimeError(f"{t['key']} archivada: reconciliar manualmente")
    if not found:
        print('CREATE',t['key'],flush=True); cli.run(*argv_for(t,ids)); found=match(cli.list(),t['key'])
        if not found: raise RuntimeError(f"No se encontró {t['key']} tras creación: NO repetir a ciegas")
    else: print('EXISTS',t['key'],found['id'],flush=True)
    # Existing records are never overwritten. Check material dependency drift.
    view=cli.json('task-view','task','view',found['id'])['task']
    if view.get('title')!=title(t): raise RuntimeError(f"Título divergente en {found['id']}; revisar")
    actual={str(d).casefold() for d in view.get('dependencies',[]) or []}
    expected={ids[d].casefold() for d in t['dependencies']}
    if actual!=expected: raise RuntimeError(f"Dependencias distintas en {found['id']}: {actual} vs {expected}; reconciliar sin sobrescribir")
    ids[t['key']]=found['id']


def preflight_existing(cli,tasks,initial):
    """Read-only: detect drift in existing tasks before creating anything."""
    existing={t['key']:match(initial,t['key']) for t in tasks}
    for t in tasks:
        current=existing[t['key']]
        if not current: continue
        view=cli.json('task-view','task','view',current['id'])['task']
        if view.get('archived'): raise RuntimeError(f"{t['key']} archivada; revisar")
        expected={existing[d]['id'].casefold() for d in t['dependencies'] if existing[d]}
        actual={str(d).casefold() for d in view.get('dependencies',[]) or []}
        if any(not existing[d] for d in t['dependencies']) or actual!=expected:
            raise RuntimeError(f"Dependencias divergentes en {t['key']}; revisar v2-reconciliation.json y conciliar por CLI")
        if view.get('description')!=desc(t):
            raise RuntimeError(f"Descripción divergente en {t['key']}; no se sobrescribe")
        if [x.get('text') for x in view.get('acceptanceCriteria',[])]!=t['acceptance']:
            raise RuntimeError(f"AC divergentes en {t['key']}; conservar evidencia y revisar por CLI")
        actual_dod=[x.get('text') for x in view.get('definitionOfDone',[])]
        if not all(d in actual_dod for d in t['dod']):
            raise RuntimeError(f"DoD divergente en {t['key']}; revisar por CLI")

def decisions(cli,items):
    for d in items:
        t=title(d)
        existing=match(cli.json('decision-list','decision','list')['decisions'],d['key'])
        if existing:
            if existing.get('title')!=t: raise RuntimeError(f'Decisión divergente: {t}')
            print('EXISTS decision',existing['id']); continue
        cli.run('decision','create',t,'-s','proposed','--context',d['context'],'--decision',d['decision'],'--alternatives',d['alternatives'],'--consequences',d['consequences'])
        if not match(cli.json('decision-list','decision','list')['decisions'],d['key']): raise RuntimeError(f'No se pudo verificar decisión {t}')
        print('CREATE decision',t)


def docs(cli,items):
    # doc list is not a documented v1 JSON envelope. Use exact output marker to
    # avoid guessing an ID, and never overwrite an existing document.
    for d in items:
        t=title(d)
        listing=cli.run('doc','list','--plain')
        if '['+d['key']+']' in listing:
            print('EXISTS doc (verificar contenido, no se sobreescribe)',t); continue
        out=cli.run('doc','create',t,'-p',d['path'],'-t',d['kind'],'--plain')
        candidates=set(re.findall(r'\bdoc-\d+\b',out,re.I))
        if len(candidates)!=1:
            raise RuntimeError(f'Doc posiblemente creado. Resolver ID real por doc list --plain y ejecutar doc update manual. Salida: {out}')
        ident=candidates.pop()
        cli.run('doc','update',ident,'--content',(HERE/'seed/docs'/d['file']).read_text(encoding='utf-8'),'--tags',','.join(d['tags']))
        print('CREATE doc',ident,t)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--home',type=pathlib.Path,help='Raíz privada de Backlog.md previamente inicializada')
    p.add_argument('--apply',action='store_true',help='Autoriza mutaciones en Backlog.md')
    p.add_argument('--confirm-config',action='store_true',help='Confirma revisión manual de configuración, permisos y alcance')
    p.add_argument('--section',choices=['all','tasks','docs','decisions'],default='all')
    a=p.parse_args(); tasks=load('tasks.json'); dd=load('decisions.json'); docs_data=load('documents.json')
    counts=validate(tasks)
    if not a.apply:
        print('DRY RUN: no CLI, no writes; alias NOE no son IDs reales.')
        print('Fases:',dict(counts)); print('Tareas',len(tasks),'decisiones',len(dd),'docs',len(docs_data))
        if a.section in ('all','tasks'):
            for t in tasks: print(t['key'],t['phase'],t['size'],t['title'],'deps:',','.join(t['dependencies']) or '-')
        return
    if not a.home or not a.confirm_config: p.error('--apply requiere --home y --confirm-config tras revisar efectos')
    home=a.home.resolve(strict=True)
    if not home.is_dir() or not shutil.which('backlog'): p.error('Home o CLI no disponible; no instalar ni inicializar automáticamente')
    cli=CLI(home)
    print(cli.run('instructions','overview')); print(cli.run('instructions','task-creation'))
    print(cli.run('config','list'))
    initial=cli.list()
    unrelated=[x for x in initial if not str(x.get('title','')).startswith('[NOE-')]
    if unrelated: raise RuntimeError('Este paquete solo se aplica a un tablero Noesis nuevo o parcialmente importado; existen otras tareas')
    # Reject an older or divergent seed before any mutations if a changed design key is present.
    # Not a migration utility; existing AC/DoD/notes remain untouched.
    known={t['key']:t for t in tasks}
    for item in initial:
        m=re.match(r'^\[(NOE-\d{3})\] ',item.get('title',''))
        if m and (m.group(1) not in known or item['title']!=title(known[m.group(1)])):
            raise RuntimeError('Versión de semilla divergente: conciliar por CLI antes de importar v3')
    if a.section in ('all','tasks'):
        preflight_existing(cli,tasks,initial)
        ids={}
        for t in tasks: reconcile_task(cli,t,ids)
        print('MAPA DE IDS NATIVOS (referencia temporal):',json.dumps(ids,ensure_ascii=False))
    if a.section in ('all','docs'): docs(cli,docs_data)
    if a.section in ('all','decisions'): decisions(cli,dd)
    print('Importación finalizada; verificar contenido, AC y DoD por CLI; ninguna tarea marcada Done.')

if __name__=='__main__':
    try: main()
    except (RuntimeError,ValueError,AssertionError,KeyError,OSError) as e:
        print('STOP:',e,file=sys.stderr); sys.exit(1)
