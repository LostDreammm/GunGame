"""Executable, bounded HTTP/JSON methods learned only after task acceptance.

The compiler deliberately supports a declarative cultural-heritage summary
contract, not arbitrary APIs. It binds current task parameters and credentials
on every execution. Recipes contain structure and operations, never answers.
"""
import copy
import hashlib
import json
import posixpath
import re
import shlex
from urllib.parse import parse_qsl, urlsplit, urlunsplit

SCHEMA = 'heritage_summary/v1'
RESULT_MARKER = '__FW_API_RESULT__='
ANSWER_MARKER = '__FW_TASK_ANSWER__='
# UTF-8 byte bounds apply to generated commands and the full sandbox output.
MAX_COMMAND_BYTES = 24000
MAX_OUTPUT_BYTES = 60000
MAX_AMBIGUITY_CANDIDATES = 64
_ERA_RANKS = {'旧石器':-2500000,'新石器':-12000,'夏':-2070,'商':-1600,'西周':-1046,'东周':-770,'周':-1046,'春秋':-770,'战国':-475,'秦':-221,'西汉':-206,'东汉':25,'汉':-206,'三国':220,'西晋':266,'东晋':317,'晋':266,'南北朝':420,'北魏':386,'东魏':534,'西魏':535,'北齐':550,'北周':557,'南齐':479,'南梁':502,'南陈':557,'隋':581,'唐':618,'五代十国':907,'五代':907,'辽':907,'北宋':960,'南宋':1127,'宋':960,'金':1115,'西夏':1038,'元':1271,'明':1368,'清':1644,'民国':1912,'现代':1949,'当代':1949}
OPERATIONS = [
    {'op': 'count', 'output': 'total_count'},
    {'op': 'filter_equals_count', 'field': 'level', 'value': '世界遗产', 'output': 'world_heritage_count'},
    {'op': 'distinct', 'field': 'type', 'output': 'types'},
    {'op': 'earliest_chronology', 'field': 'era', 'select': 'name', 'output': 'oldest_era'},
]

# This source is emitted into a separate task sandbox; it cannot import us.
_SOURCE = r'''
import hashlib,json,re,signal,time,urllib.request,urllib.error,urllib.parse
B=__BINDING__
D=time.monotonic()+11
requests=0
repairs=[]
diagnostic={}
class Failure(Exception): pass
def fail(reason): raise Failure(reason)
def alarm(*args): fail('time_limit')
signal.signal(signal.SIGALRM,alarm)
signal.setitimer(signal.ITIMER_REAL,11)
def pairs(items):
 d={}
 for k,v in items:
  if k in d: fail('duplicate_json_key')
  d[k]=v
 return d
def strict(s):
 return json.loads(s,object_pairs_hook=pairs,parse_constant=lambda x:fail('invalid_json_number'))
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args,**kwargs): return None
opener=urllib.request.build_opener(NoRedirect)
def get(auth,param,offset,limit):
 global requests
 requests+=1
 diagnostic.clear()
 diagnostic['step']={'offset':offset,'parameter':param,'auth':auth}
 if requests>24: fail('request_limit')
 remain=D-time.monotonic()
 if remain<=0: fail('time_limit')
 query={param:B['city'],'offset':offset,'limit':limit}
 head={'Accept':'application/json'}
 head['Authorization' if auth=='bearer' else 'X-API-Key']=('Bearer ' if auth=='bearer' else '')+B['key']
 req=urllib.request.Request(B['service']+'?'+urllib.parse.urlencode(query),headers=head,method='GET')
 try:
  response=opener.open(req,timeout=min(2.5,remain))
 except urllib.error.HTTPError as e: response=e
 with response:
  status=response.code
  raw=response.read(524289)
 if len(raw)>524288: fail('response_size_limit')
 try: value=strict(raw.decode('utf-8'))
 except (ValueError,UnicodeError): fail('invalid_json_response')
 code=status
 msg=''
 if isinstance(value,dict):
  envelopes=[value]+[value[k] for k in ('data','result') if isinstance(value.get(k),dict)]
  for obj in envelopes:
   c=obj.get('code',obj.get('status_code'))
   if isinstance(c,str) and c.isdigit(): c=int(c)
   if type(c) is int and c>=400: code=c
   bad=obj.get('success') is False or str(obj.get('status','')).lower() in ('error','failed','failure') or obj.get('error')
   if bad or (type(c) is int and c>=400):
    msg=str(obj.get('message',obj.get('error','')))
    if code<400: code=500
  if not msg: msg=str(value.get('message',value.get('error','')))
 if status>=400 or code>=400:
  safe=msg.replace(B['key'],'[REDACTED]')
  safe=re.sub(r'(?i)(Bearer\s+|X-API-Key\s*:\s*)[A-Za-z0-9._~+/=-]+',r'\1[REDACTED]',safe)
  diagnostic.update({'http_status':status,'api_status':code,'message':safe[:600],'step':{'offset':offset,'parameter':param,'auth':auth}})
  return None,code,msg
 if not 200<=status<300: fail('http_status_'+str(status))
 return value,None,''
def path(value,p):
 for k in p:
  if not isinstance(value,dict) or k not in value: return None
  value=value[k]
 return value
def discover(value):
 candidates=[]
 for rp in ([],['records'],['items'],['results'],['data'],['data','records'],['data','items'],['data','results'],['result','records']):
  if isinstance(path(value,rp),list): candidates.append(rp)
 if len(candidates)!=1: fail('unsupported_records_schema')
 rp=candidates[0]
 pps=[['pagination'],['paging'],['data','pagination'],['data','paging'],['result','pagination']]
 pps=[p for p in pps if isinstance(path(value,p),dict)]
 if len(pps)!=1: fail('missing_pagination_proof')
 return rp,pps[0]
def integer(value,label):
 if type(value) is not int or value<0: fail('invalid_'+label)
 return value
def fieldmap(records):
 aliases={'name':['name','heritage_name'],'era':['era','dynasty','period'],'type':['type','category'],'level':['protected_level','protection_level','level','protection'],'id':['id','heritage_id']}
 keys=set(records[0]) if records else set()
 fm={}
 for role,opts in aliases.items():
  hits=[k for k in opts if k in keys]
  if len(hits)>1: fail('ambiguous_field_'+role)
  if not hits and role!='id' and records: fail('missing_field_'+role)
  fm[role]=hits[0] if hits else (None if role=='id' else opts[0])
 return fm
def schema(records,rp,pp,fm):
 shapes=[]
 for row in records:
  if not isinstance(row,dict): fail('invalid_record')
  shape=sorted((k,type(v).__name__) for k,v in row.items())
  if shapes and shape!=shapes[0]: fail('record_schema_drift')
  shapes=[shape]
 value={'records':rp,'pagination':pp,'fields':fm,'shape':shapes}
 return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def era_key(s):
 if not isinstance(s,str) or not s.strip(): fail('unknown_era')
 eras=__ERA_RANKS__
 s=s.strip()
 year=re.fullmatch(r'(公元前|前|公元)?\s*(\d{1,6})年?',s)
 if year: return int(year[2])*(-1 if year[1] in ('公元前','前') else 1)
 pattern='|'.join(re.escape(k) for k in sorted(eras,key=len,reverse=True))
 hits=re.findall(pattern,s)
 rest=re.sub(pattern,'',s)
 rest=re.sub(r'时期|时代|年间|朝|代|至|到|[\s、，,\-—~～/]+','',rest)
 if not hits or rest: fail('unknown_era')
 return min(eras[k] for k in hits)
def main():
 auth=B.get('recipe',{}).get('auth',B['auth'])
 param=B.get('recipe',{}).get('parameter',B['parameter'])
 offset=0
 limit=100
 total=None
 records=[]
 seen=set()
 fingerprints=set()
 offsets=[]
 page_counts=[]
 rp=pp=fm=rs=None
 for page in range(20):
  while True:
   value,code,msg=get(auth,param,offset,limit)
   if code is None: break
   if code==401 and auth!='bearer' and re.search(r'authorization',msg,re.I) and re.search(r'bearer',msg,re.I):
    auth='bearer';repairs.append('auth:bearer');continue
   if code==400 and param!='location' and re.search(r'(?:missing|required|缺少|必需)',msg,re.I) and re.search(r'\blocation\b',msg,re.I):
    param='location';repairs.append('parameter:location');continue
   fail('api_error_'+str(code))
  newrp,newpp=discover(value)
  rows=path(value,newrp)
  pagination=path(value,newpp)
  keys=[k for k in ('total_count','total','total_records') if k in pagination]
  if len(keys)!=1: fail('missing_total_proof')
  n=integer(pagination[keys[0]],'total')
  start=integer(pagination.get('offset'),'offset')
  size=integer(pagination.get('limit'),'limit')
  if n>5000 or size<1 or size>5000 or len(rows)>size: fail('pagination_limit')
  if start!=offset: fail('pagination_offset_drift')
  if total is None: total=n
  if total!=n: fail('pagination_total_drift')
  if not all(isinstance(row,dict) for row in rows): fail('invalid_record')
  newfm=fieldmap(rows)
  newrs=schema(rows,newrp,newpp,newfm)
  if rp is None:
   rp,pp,fm,rs=newrp,newpp,newfm,newrs
   old=B.get('recipe',{})
   if old and old.get('response_schema')!=rs: repairs.append('schema:rediscovered')
  elif rows and (newrp,newpp,newfm,newrs)!=(rp,pp,fm,rs): fail('pagination_schema_drift')
  fingerprint=hashlib.sha256(json.dumps(rows,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
  if rows and fingerprint in fingerprints: fail('repeated_page')
  fingerprints.add(fingerprint)
  if not rows and offset<total: fail('incomplete_records')
  offsets.append(offset)
  page_counts.append(len(rows))
  for row in rows:
   for role in ('name','era','type','level'):
    if not isinstance(row.get(fm[role]),str) or not row[fm[role]].strip(): fail('invalid_field_'+role)
   for cityfield in ('city','location'):
    if cityfield in row and row[cityfield] not in (B['city'],B['city']+'市'): fail('wrong_city_record')
   ident=row.get(fm['id']) if fm['id'] else None
   if fm['id'] and (not isinstance(ident,(str,int)) or isinstance(ident,bool) or ident==''): fail('invalid_id')
   ident=json.dumps(ident,ensure_ascii=False) if fm['id'] else json.dumps(row,sort_keys=True,ensure_ascii=False)
   if ident in seen: fail('duplicate_record')
   seen.add(ident)
   records.append(row)
  offset+=len(rows)
  if offset>total: fail('count_exceeds_total')
  if offset==total: break
  limit=size
 else: fail('page_limit')
 if len(seen)!=total: fail('incomplete_unique_records')
 answer={'city':B['city']}
 candidates=[]
 for operation in B['operations']:
  op=operation['op']; output=operation['output']
  if op=='count': answer[output]=len(records)
  elif op=='filter_equals_count': answer[output]=sum(r[fm[operation['field']]]==operation['value'] for r in records)
  elif op=='distinct': answer[output]=sorted(set(r[fm[operation['field']]] for r in records))
  elif op=='earliest_chronology':
   ranked=[(era_key(r[fm[operation['field']]]),r[fm[operation['select']]],r[fm[operation['field']]]) for r in records]
   first=min((k for k,n,e in ranked),default=None)
   names=set()
   for rank,name,era in ranked:
    if rank==first and name not in names:
     names.add(name);candidates.append({'name':name,'era':era})
   if len(candidates)>64: fail('ambiguity_candidate_limit')
   if len(candidates)<=1: answer[output]=candidates[0]['name'] if candidates else None
  else: fail('unsupported_operation')
 recipe={'version':1,'family':'http_json','service':B['service'],'schema':B['schema'],'auth':auth,'parameter':param,'records_path':rp,'pagination_path':pp,'fields':fm,'response_schema':rs,'operations':B['operations']}
 evidence={'complete':True,'declared_total':total,'unique_count':len(seen),'raw_count':len(records),'pages':len(offsets),'offsets':offsets,'page_counts':page_counts,'response_schema':rs,'request_count':requests,'repairs':repairs}
 if len(candidates)>1:
  return {'ok':False,'error':'ambiguous_earliest_era','partial_answer':answer,'candidates':candidates,'recipe':recipe,'evidence':evidence,'service':B['service'],'schema':B['schema'],'city':B['city']}
 return {'ok':True,'answer':answer,'recipe':recipe,'evidence':evidence,'service':B['service'],'schema':B['schema'],'city':B['city']}
try:
 result=main()
except Exception as e:
 result={'ok':False,'error':str(e) if isinstance(e,Failure) else type(e).__name__,'service':B['service'],'schema':B['schema'],'city':B['city'],'evidence':{'complete':False,'request_count':requests,'repairs':repairs},'diagnostic':diagnostic}
signal.setitimer(signal.ITIMER_REAL,0)
def render(value):
 text='__FW_API_RESULT__='+json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n'
 if value['ok']: text+='__FW_TASK_ANSWER__='+json.dumps({'answer':value['answer']},ensure_ascii=False,separators=(',',':'))+'\n'
 return text
output=render(result)
if len(output.encode('utf-8'))>60000:
 output=render({'ok':False,'error':'output_size_limit','evidence':{'complete':False,'request_count':requests,'repairs':repairs}})
print(output,end='')
'''


def _url(value):
    if not isinstance(value, str):
        return None
    try:
        parts = urlsplit(value)
        if (parts.scheme not in ('http', 'https') or not parts.hostname or
                parts.username or parts.password or parts.fragment):
            return None
        return urlunsplit((parts.scheme, parts.netloc, parts.path or '', '', ''))
    except ValueError:
        return None


def _binding(context, notice):
    if not isinstance(context, dict) or context.get('status', 'ok') != 'ok':
        return None
    task = context.get('task_document', '')
    if not isinstance(task, str) or not task.strip() or len(task) > 50000:
        return None
    truncated = context.get('truncated', {})
    if isinstance(truncated, dict) and any(truncated.get(k) for k in ('task_document', 'refs', 'output')):
        return None
    required = ('city', 'total_count', 'world_heritage_count', 'types', 'oldest_era')
    if not all(re.search(r'\b' + key + r'\b', task) for key in required):
        return None
    # The automatic contract must match the requested operations, not merely
    # the output key names. Unknown filters/semantics belong to general reasoning.
    def clauses(key):
        parts = []
        for match in re.finditer(r'(?<![A-Za-z0-9_])' + key + r'(?![A-Za-z0-9_])', task):
            rest = task[match.end():match.end() + 300]
            parts.append(re.split(r'city|total_count|world_heritage_count|types|oldest_era|[\r\n;；]', rest)[0])
        return parts
    oldest = clauses('oldest_era')
    world = clauses('world_heritage_count')
    if not any(re.search(r'(?:年代|时代|时期).{0,12}(?:最早|最古老).{0,12}(?:名称|名字)', part) for part in oldest):
        return None
    if any(re.search(r'(?:不是|不返回|非).{0,8}(?:名称|名字)|(?<!不)返回年代|(?<!不)是年代文本', part) for part in oldest):
        return None
    if not any('世界遗产' in part and re.search(r'级别|等于|为|恰好', part) for part in world):
        return None
    if any(re.search(r'包含|含有|contains|substring', part, re.I) for part in world):
        return None
    if not any(re.search(r'不重复|去重|distinct|unique', part, re.I) for part in clauses('types')):
        return None
    if not re.search(r'(?:全部|所有)(?:的)?(?:文化遗产)?记录|全部文化遗产|全部遗产', task.replace('*', '')):
        return None
    if re.search(r'(?:只|仅)(?:统计|查询|获取|拉取|计算)|筛选|过滤|限定|\b(?:exclude|filter|only count)\b', task, re.I):
        return None

    cities = set(re.findall(r'["\x27]city["\x27]\s*:\s*["\x27]([^"\x27\r\n<>]{1,80})["\x27]', task))
    if len(cities) != 1:
        return None
    city = next(iter(cities))
    # A copied output example must never override the actual requested city.
    narrative = re.sub(r'```.*?```', '', task, flags=re.S).replace('*', '')
    named = set(re.findall(r'(?:查询|拉取|获取)\s*([\u4e00-\u9fff]{2,12}?)(?:市)?(?:的)?(?:全部)?文化遗产', narrative))
    named.difference_update({'该城', '该城市', '该市', '所有城', '所有城市'})
    for title in re.findall(r'^\s*#+\s*(.+)$', narrative, re.M):
        title = re.split(r'[:：]', title)[-1].strip()
        match = re.fullmatch(r'(?:查询)?([\u4e00-\u9fff]{2,12}?)(?:市)?(?:的)?文化遗产(?:统计|查询|报告)?', title)
        if match:
            named.add(match[1])
    if named and (len(named) != 1 or re.sub('市$', '', next(iter(named))) != re.sub('市$', '', city)):
        return None
    docs = []
    for item in context.get('referenced_documents', []):
        if not isinstance(item, dict) or item.get('truncated') or item.get('error'):
            return None
        content = item.get('content', '')
        if not isinstance(content, str):
            return None
        docs.append(content)
    source = task + '\n' + '\n'.join(docs)
    if len(source) > 100000:
        return None
    urls = re.findall(r'https?://[^\s`"\x27<>\\]+', source)
    urls = [u.rstrip('。；，,);]') for u in urls]
    normalized = {_url(u) for u in urls}
    normalized.discard(None)
    origins = {urlunsplit((urlsplit(u).scheme, urlsplit(u).netloc, '', '', '')) for u in normalized}
    if len(origins) != 1:
        return None
    base = next(iter(origins))
    endpoints = {u for u in normalized if urlsplit(u).path not in ('', '/')}
    for endpoint in re.findall(r'\bGET\s+(/[^\s`"\x27<>?]+)', source, re.I):
        normalized_endpoint = _url(base + endpoint)
        if normalized_endpoint:
            endpoints.add(normalized_endpoint)
    if len(endpoints) != 1:
        return None
    service = next(iter(endpoints))
    credentials = []
    for auth, pattern in (
            ('bearer', r'Authorization\s*:\s*Bearer\s+([A-Za-z0-9._~+/=-]+)'),
            ('x-api-key', r'X-API-Key\s*:\s*([A-Za-z0-9._~+/=-]+)'),
            ('x-api-key', r'(?:API[ _-]?Key|api_key)\s*[:=：]\s*[`"\x27]?([A-Za-z0-9._~+/=-]+)')):
        for key in re.findall(pattern, source, re.I):
            if len(key) >= 3 and key.lower() not in ('your-api-key', 'your_api_key', 'api_key', 'here', 'key'):
                credentials.append((auth, key))
    if not credentials:
        # Markdown credential tables, restricted to the explicitly labelled key section.
        match = re.search(r'API[ _-]?Key[^\n]*\n((?:[^\n]*\n){0,8})', source, re.I)
        if match:
            keys = re.findall(r'^\s*\|\s*`([A-Za-z0-9._~+/=-]{3,256})`\s*\|', match[1], re.M)
            credentials.extend(('x-api-key', key) for key in keys)
    keys = {key for auth, key in credentials}
    if len(keys) != 1:
        return None
    key = next(iter(keys))
    if len(key) > 256:
        return None
    auth = 'bearer' if ('bearer', key) in credentials else 'x-api-key'
    params = {k for u in urls for k, v in parse_qsl(urlsplit(u).query) if k in ('city', 'location')}
    parameter = next(iter(params)) if len(params) == 1 else 'city'
    return {'service': service, 'schema': SCHEMA, 'city': city, 'key': key,
            'auth': auth, 'parameter': parameter, 'operations': copy.deepcopy(OPERATIONS)}


def _safe_recipe(recipe):
    if not isinstance(recipe, dict):
        return False
    allowed = {'version', 'family', 'service', 'schema', 'auth', 'parameter', 'records_path',
               'pagination_path', 'fields', 'response_schema', 'operations'}
    if set(recipe) != allowed or type(recipe.get('version')) is not int or recipe.get('version') != 1 or recipe.get('family') != 'http_json':
        return False
    if recipe.get('schema') != SCHEMA or not _url(recipe.get('service', '')) or _url(recipe['service']) != recipe['service']:
        return False
    if recipe.get('auth') not in ('bearer', 'x-api-key') or recipe.get('parameter') not in ('city', 'location'):
        return False
    if recipe.get('operations') != OPERATIONS:
        return False
    if not isinstance(recipe.get('response_schema'), str) or not re.fullmatch('[0-9a-f]{64}', recipe['response_schema']):
        return False
    allowed_paths = ([], ['records'], ['items'], ['results'], ['data'], ['data', 'records'], ['data', 'items'], ['data', 'results'], ['result', 'records'])
    if recipe.get('records_path') not in allowed_paths:
        return False
    if recipe.get('pagination_path') not in (['pagination'], ['paging'], ['data', 'pagination'], ['data', 'paging'], ['result', 'pagination']):
        return False
    fields = recipe.get('fields')
    if not isinstance(fields, dict) or set(fields) != {'name', 'era', 'type', 'level', 'id'}:
        return False
    options = {'name': ('name', 'heritage_name'), 'era': ('era', 'dynasty', 'period'),
               'type': ('type', 'category'), 'level': ('protected_level', 'protection_level', 'level', 'protection'),
               'id': ('id', 'heritage_id', None)}
    return all(fields[k] in choices for k, choices in options.items())


def compile_task(context, notice='', recipe=None):
    """Compile this supported task contract, returning None for unsupported tasks."""
    binding = _binding(context, notice)
    if binding is None:
        return None
    reused = bool(_safe_recipe(recipe) and recipe['service'] == binding['service'])
    if reused:
        # Only mutable transport hints and schema identity need sandbox binding.
        # Every response is rediscovered and verified; this is not a cached answer.
        binding['recipe'] = {k: copy.deepcopy(recipe[k]) for k in ('auth', 'parameter', 'response_schema')}
        binding['operations'] = copy.deepcopy(recipe['operations'])
    source = _SOURCE.replace('__BINDING__', repr(binding)).replace('__ERA_RANKS__', repr(_ERA_RANKS))
    command = 'python3 -c ' + shlex.quote(source)
    if len(command.encode('utf-8')) > MAX_COMMAND_BYTES:
        return None
    return {'kind': 'api_skill', 'command': command, 'answer_from_stdout': True,
            'recipe': copy.deepcopy(recipe) if reused else None,
            'service': binding['service'], 'schema': SCHEMA,
            'city': binding['city'], 'reused': reused,
            'document_sources': _document_sources(context, binding['service'])}


def _safe_document_source(source, service):
    if (not isinstance(source, dict) or set(source) != {'path', 'task_directory', 'service'}
            or source.get('service') != service or not _url(service) or _url(service) != service):
        return False
    path, directory = source.get('path'), source.get('task_directory')
    if not all(isinstance(value, str) and value.startswith('/') and len(value) <= 2048
               and not any(ord(c) < 32 for c in value) and posixpath.normpath(value) == value
               for value in (path, directory)):
        return False
    return (posixpath.dirname(path) == directory and
            bool(re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*\.md', posixpath.basename(path))))


def _document_sources(context, service):
    task_path = context.get('task_path')
    if not isinstance(task_path, str) or not task_path.startswith('/'):
        return []
    directory = posixpath.dirname(task_path)
    sources = []
    for doc in context.get('referenced_documents', []):
        if not isinstance(doc, dict) or doc.get('error') or doc.get('truncated'):
            continue
        source = {'path': doc.get('path'), 'task_directory': directory, 'service': service}
        if _safe_document_source(source, service) and source not in sources:
            sources.append(source)
    return sources[:3]


def _strict_loads(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('nonfinite value')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def _valid_success(value):
    if not isinstance(value, dict) or value.get('ok') is not True:
        return False
    recipe, answer, evidence = (value.get(k) for k in ('recipe', 'answer', 'evidence'))
    if not _safe_recipe(recipe) or not isinstance(answer, dict) or not isinstance(evidence, dict):
        return False
    if value.get('service') != recipe['service'] or value.get('schema') != SCHEMA:
        return False
    if set(answer) != {'city', 'total_count', 'world_heritage_count', 'types', 'oldest_era'}:
        return False
    if not isinstance(answer['city'], str) or not answer['city'] or value.get('city') != answer['city']:
        return False
    total, world = answer['total_count'], answer['world_heritage_count']
    if type(total) is not int or not 0 <= total <= 5000 or type(world) is not int or not 0 <= world <= total:
        return False
    types = answer['types']
    if not isinstance(types, list) or not all(isinstance(t, str) and t.strip() for t in types):
        return False
    if len(types) != len(set(types)) or len(types) > total or bool(types) != bool(total):
        return False
    oldest = answer['oldest_era']
    if (total == 0 and oldest is not None) or (total > 0 and (not isinstance(oldest, str) or not oldest.strip())):
        return False
    if evidence.get('complete') is not True or any(type(evidence.get(k)) is not int or evidence[k] != total for k in ('declared_total', 'unique_count', 'raw_count')):
        return False
    if evidence.get('response_schema') != recipe['response_schema']:
        return False
    pages, offsets, requests = (evidence.get(k) for k in ('pages', 'offsets', 'request_count'))
    if type(pages) is not int or not 1 <= pages <= 20 or not isinstance(offsets, list) or len(offsets) != pages:
        return False
    if any(type(n) is not int or n < 0 for n in offsets) or offsets[0] != 0 or offsets != sorted(set(offsets)):
        return False
    if (total and offsets[-1] >= total) or (not total and offsets != [0]):
        return False
    counts = evidence.get('page_counts')
    if not isinstance(counts, list) or len(counts) != pages or any(type(n) is not int or n < 0 for n in counts):
        return False
    if sum(counts) != total or any(offsets[i] != sum(counts[:i]) for i in range(pages)):
        return False
    if total and not all(counts):
        return False
    if type(requests) is not int or not pages <= requests <= 24:
        return False
    repairs = evidence.get('repairs')
    return isinstance(repairs, list) and len(repairs) <= 3 and len(set(repairs)) == len(repairs) and all(r in ('auth:bearer', 'parameter:location', 'schema:rediscovered') for r in repairs)


def _candidate_rank(era):
    """Recognize the same chronology language as the compiled executor."""
    if not isinstance(era, str) or not era.strip():
        return None
    era = era.strip()
    year = re.fullmatch(r'(公元前|前|公元)?\s*(\d{1,6})年?', era)
    if year:
        return int(year[2]) * (-1 if year[1] in ('公元前', '前') else 1)
    pattern = '|'.join(re.escape(k) for k in sorted(_ERA_RANKS, key=len, reverse=True))
    hits = re.findall(pattern, era)
    rest = re.sub(pattern, '', era)
    rest = re.sub(r'时期|时代|年间|朝|代|至|到|[\s、，,\-—~～/]+', '', rest)
    return min(_ERA_RANKS[k] for k in hits) if hits and not rest else None


def _valid_ambiguity(value):
    """A complete query with unresolved tie semantics is not a failed query."""
    expected = {'ok', 'error', 'service', 'schema', 'city', 'recipe', 'evidence',
                'partial_answer', 'candidates'}
    if (not isinstance(value, dict) or set(value) != expected or
            value.get('ok') is not False or value.get('error') != 'ambiguous_earliest_era'):
        return False
    if len((RESULT_MARKER + json.dumps(value, ensure_ascii=False, separators=(',', ':')) +
            '\n').encode('utf-8')) > MAX_OUTPUT_BYTES:
        return False
    partial, candidates = value['partial_answer'], value['candidates']
    if (not isinstance(partial, dict) or set(partial) !=
            {'city', 'total_count', 'world_heritage_count', 'types'} or
            not isinstance(candidates, list) or not 2 <= len(candidates) <= MAX_AMBIGUITY_CANDIDATES):
        return False
    names, ranks = set(), set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != {'name', 'era'}:
            return False
        name = candidate['name']
        if not isinstance(name, str) or not name.strip() or name in names:
            return False
        rank = _candidate_rank(candidate['era'])
        if rank is None:
            return False
        names.add(name)
        ranks.add(rank)
    if len(ranks) != 1:
        return False
    # Reuse every existing count, transport schema, and pagination proof check.
    # This temporary validation envelope is never returned as a unique answer.
    validation = dict(value, ok=True, answer=dict(partial, oldest_era=candidates[0]['name']))
    return _valid_success(validation) and len(candidates) <= partial['total_count']


def ambiguity_answers(plan, result):
    """Return bounded candidate answers only for this plan's verified tie.

    API order is retained solely as a deterministic attempt order. It does not
    prove which tied name an external judge expects. No recipe stores answers.
    """
    try:
        if (not isinstance(plan, dict) or not _valid_ambiguity(result) or
                any(result.get(k) != plan.get(k) for k in ('city', 'service', 'schema'))):
            return []
        return [dict(copy.deepcopy(result['partial_answer']), oldest_era=item['name'])
                for item in result['candidates']]
    except (ValueError, TypeError, KeyError, RecursionError, OverflowError):
        return []


def parse_api_result(raw):
    """Validate the sandbox envelope and completeness evidence before submission."""
    invalid = {'ok': False, 'error': 'invalid_api_result'}
    if not isinstance(raw, str) or len(raw) > 100000 or not raw.startswith('[exitCode:0]\n'):
        return invalid
    if any(tag in raw for tag in ('[TRUNCATED]', '[TIMEOUT]', '[JUDGER_ERROR]')):
        return invalid
    try:
        if len(raw.split('\n', 1)[1].encode('utf-8')) > MAX_OUTPUT_BYTES:
            return invalid
    except UnicodeError:
        return invalid
    lines = [line.strip() for line in raw.splitlines()[1:] if line.strip()]
    results = [line[len(RESULT_MARKER):] for line in lines if line.startswith(RESULT_MARKER)]
    answers = [line[len(ANSWER_MARKER):] for line in lines if line.startswith(ANSWER_MARKER)]
    if len(results) != 1:
        return invalid
    try:
        value = _strict_loads(results[0])
        if not isinstance(value, dict):
            return invalid
        if value.get('ok') is False and not answers and isinstance(value.get('error'), str):
            if value['error'] == 'ambiguous_earliest_era':
                return value if _valid_ambiguity(value) else invalid
            return {'ok': False, 'error': value['error'][:160], 'evidence': value.get('evidence', {}),
                    'diagnostic': value.get('diagnostic', {})}
        if len(answers) != 1 or not lines[-1].startswith(ANSWER_MARKER) or not _valid_success(value):
            return invalid
        marker = _strict_loads(answers[0])
        if not isinstance(marker, dict) or set(marker) != {'answer'} or marker['answer'] != value['answer']:
            return invalid
        return value
    except (ValueError, TypeError, KeyError, RecursionError):
        return invalid


def validate_result(plan, result):
    """Bind a parsed execution to the current plan, never an earlier city's result."""
    try:
        return (isinstance(plan, dict) and _valid_success(result) and
                all(result.get(k) == plan.get(k) for k in ('city', 'service', 'schema')))
    except (ValueError, TypeError, KeyError):
        return False


class SkillMemory:
    """Small within-match recipe cache; successful execution alone cannot promote."""
    def __init__(self):
        self.recipes = {}
        self._documents = {}

    def document_hints(self):
        """Return sources for fresh sandbox reads, never credential or answer bytes."""
        return copy.deepcopy(list(reversed(list(self._documents.values()))))

    def propose(self, context, notice=''):
        plan = compile_task(context, notice)
        if plan is None:
            return None
        for recipe in reversed(list(self.recipes.values())):
            if recipe['service'] == plan['service'] and recipe['schema'] == plan['schema']:
                return compile_task(context, notice, recipe)
        return plan

    def learn(self, plan, result, confirmed=False):
        if confirmed is not True or not validate_result(plan, result):
            return False
        recipe = copy.deepcopy(result['recipe'])
        key = (recipe['service'], recipe['schema'], recipe['response_schema'])
        self.recipes.pop(key, None)
        self.recipes[key] = recipe
        for source in plan.get('document_sources', []):
            if _safe_document_source(source, recipe['service']):
                source_key = (source['service'], source['path'])
                self._documents.pop(source_key, None)
                self._documents[source_key] = copy.deepcopy(source)
        while len(self.recipes) > 8:
            del self.recipes[next(iter(self.recipes))]
        services = {item['service'] for item in self.recipes.values()}
        self._documents = {key: value for key, value in self._documents.items() if key[0] in services}
        while len(self._documents) > 8:
            del self._documents[next(iter(self._documents))]
        return True

    def snapshot(self):
        return {'count': len(self.recipes), 'recipes': copy.deepcopy(list(self.recipes.values())),
                'document_sources': self.document_hints()}
