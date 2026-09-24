"""Narrow deployment-spec repairs and evidence from the task's own ./check.

The generated commands run in the platform workspace, never in this process.
Unsupported specifications remain a model task; a partial plan is never emitted.
"""
import json
import posixpath
import re
import shlex


MARKER = '__FW_CHECK_RESULT__='


def _absolute(value):
    return (isinstance(value, str) and value.startswith('/') and value != '/'
            and '\x00' not in value and posixpath.normpath(value) == value)


def engineering_workspace(context):
    """Recognize the explicit repair/spec/check/token contract from its document."""
    if not isinstance(context, dict) or context.get('status') != 'ok':
        return None
    workspace, document = context.get('workspace'), context.get('task_document')
    truncated = context.get('truncated') or {}
    if (not _absolute(workspace) or not isinstance(document, str)
            or not isinstance(truncated, dict) or truncated.get('task_document')
            or truncated.get('output') or not isinstance(context.get('spec'), str)
            or not context['spec'].strip()
            or context.get('spec_path') != posixpath.join(workspace, 'spec.md')):
        return None
    requirements = (r'进入(?:工作区|该目录)', r'修复', r'(?:阅读|根据)\s*`?spec\.md',
                    r'(?:运行|执行)\s*`?\./check', r'提交',
                    r'\{\s*"token"\s*:\s*"xxx"\s*\}')
    if not all(re.search(pattern, document) for pattern in requirements):
        return None
    references = []
    for raw in re.findall(r'`cd\s+([^`\r\n]+)`', document):
        try:
            parts = shlex.split(raw)
        except ValueError:
            return None
        if len(parts) != 1 or not parts[0].startswith('/'):
            return None
        references.append(posixpath.normpath(parts[0]))
    return workspace if set(references) == {workspace} else None


def _relative(value):
    value = value.rstrip('/')
    if (not re.fullmatch(r'[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*', value)
            or any(part in ('.', '..') for part in value.split('/'))
            or value in ('check', 'spec.md')):
        return None
    return value


def _plan(spec):
    lines = [line.strip() for line in spec.splitlines() if line.strip()]
    if not lines or not re.fullmatch(r'#\s+应用\s+[A-Za-z0-9_.-]+\s+部署规范', lines[0]):
        return None
    plan = {'directories': [], 'configs': [], 'scripts': []}
    section, used, config = None, set(), None
    for line in lines[1:]:
        if line in ('## 目录要求', '## 脚本要求'):
            if line in used:
                return None
            used.add(line)
            section = 'directories' if line == '## 目录要求' else 'scripts'
            continue
        match = re.fullmatch(r'##\s+配置文件\s+([^\s]+)', line)
        if match:
            path = _relative(match.group(1))
            if not path:
                return None
            config = {'path': path, 'lines': []}
            plan['configs'].append(config)
            section = 'configs'
            continue
        if section == 'directories':
            match = re.fullmatch(r'-\s+([^\s]+)\s+必须存在\s*[,，]\s*权限为\s*([0-7]{3})', line)
            if not match or not _relative(match.group(1)):
                return None
            plan[section].append({'path': _relative(match.group(1)), 'mode': int(match.group(2), 8)})
        elif section == 'scripts':
            match = re.fullmatch(r'-\s+([^\s]+)\s+必须存在且可执行\s*[（(]\s*权限\s+([0-7]{3})\s*[）)]', line)
            if not match or not _relative(match.group(1)) or not int(match.group(2), 8) & 0o111:
                return None
            plan[section].append({'path': _relative(match.group(1)), 'mode': int(match.group(2), 8)})
        elif section == 'configs':
            match = re.fullmatch(r'-\s+第\s*([1-9][0-9]{0,3})\s*行\s*[:：]\s*`([^`\r\n]+)`', line)
            if not match or int(match.group(1)) in [item[0] for item in config['lines']]:
                return None
            config['lines'].append([int(match.group(1)), match.group(2)])
        else:
            return None
    paths = [item['path'] for group in plan.values() for item in group]
    if (not all(plan.values()) or any(not item['lines'] for item in plan['configs'])
            or len(paths) != len(set(paths))):
        return None
    return plan


# Kept self-contained: the platform sandbox does not import the contestant files.
_PRELUDE = r'''
import hashlib,json,os,pathlib,selectors,shlex,signal,stat,subprocess,time
root=pathlib.Path(workspace)
result={'workspace':workspace,'returncode':None,'stdout':'','stderr':'','timed_out':False,'truncated':False}
class CheckError(Exception):
    def __init__(self,code,message):
        self.code=code
        super().__init__(message)
def script_text(data):
    if not data.startswith(b'#!') or b'\x00' in data:
        return None
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        return None
def safe(relative):
    path=root/relative
    path.resolve().relative_to(root)
    part=path
    while part != root:
        if part.is_symlink():
            raise ValueError('symlink target is not repairable: '+relative)
        part=part.parent
    return path
def checker():
    if not root.is_dir() or root.resolve()!=root:
        raise CheckError('CHECK_WORKSPACE_INVALID','workspace is missing or no longer the resolved directory')
    path=safe('check')
    if not path.is_file() or not os.access(path,os.X_OK):
        raise CheckError('CHECK_UNAVAILABLE','./check must exist and be executable')
    with path.open('rb') as stream:
        data=stream.read(1048577)
    if len(data)>1048576:
        raise CheckError('CHECK_TOO_LARGE','./check exceeds bounded read size')
    if expected_hash is not None:
        if hashlib.sha256(data).hexdigest()!=expected_hash:
            if not (expected_lf_hash and script_text(data) is not None and
                    hashlib.sha256(data.replace(b'\r\n',b'\n')).hexdigest()==expected_lf_hash):
                raise CheckError('CHECK_INTEGRITY','./check content changed beyond line endings; verification refused')
            result['integrity']='line_endings_only'
    return path,data
def run_check():
    path,data=checker()
    command=['./check']
    text=script_text(data)
    if text is not None and '\r\n' in text:
        text=text.replace('\r\n','\n')
        interpreter=shlex.split(text.split('\n',1)[0][2:].strip())
        supported=(interpreter in (['/bin/sh'],['/bin/bash'],['/usr/bin/sh'],['/usr/bin/bash'],
                                   ['/usr/bin/env','sh'],['/usr/bin/env','bash']))
        if not supported or len(text.encode('utf-8'))>64000:
            raise CheckError('CHECK_INTERPRETER_UNSUPPORTED','CRLF checker requires a supported shell interpreter and bounded script')
        # Preserve cwd and $0 while keeping the original checker bytes intact.
        command=interpreter+['-c',text,'./check']
        result['execution_mode']='crlf_shell_in_memory'
    try:
        process=subprocess.Popen(command,cwd=str(root),stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    except OSError as error:
        raise CheckError('CHECK_LAUNCH_ERROR',str(error))
    selector=selectors.DefaultSelector()
    buffers={'stdout':bytearray(),'stderr':bytearray()}
    limits={'stdout':2048,'stderr':512}
    for pipe,name in ((process.stdout,'stdout'),(process.stderr,'stderr')):
        os.set_blocking(pipe.fileno(),False)
        selector.register(pipe,selectors.EVENT_READ,name)
    deadline=time.monotonic()+7
    try:
        while selector.get_map():
            remaining=deadline-time.monotonic()
            if remaining<=0:
                result['timed_out']=True
                break
            for key,_ in selector.select(min(remaining,.1)):
                chunk=os.read(key.fd,65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                name=key.data
                room=limits[name]-len(buffers[name])
                buffers[name].extend(chunk[:room])
                result['truncated']|=len(chunk)>room
        if not result['timed_out']:
            try:
                process.wait(timeout=max(.001,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                result['timed_out']=True
    finally:
        if result['timed_out'] or process.poll() is None:
            try:
                os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=.5)
        selector.close()
        process.stdout.close()
        process.stderr.close()
    result['returncode']=process.returncode
    for name,data in buffers.items():
        result[name]=data.decode('utf-8','replace')
'''

_REPAIR = r'''
def repair():
    check,_=checker()
    specification=safe('spec.md')
    with specification.open('rb') as stream:
        current=stream.read(20001).decode('utf-8')
    if current!=spec:
        raise ValueError('spec.md changed or is incomplete; reread requirements')
    targets=[]
    edits=[]
    for kind,entries in plan.items():
        for entry in entries:
            path=safe(entry['path'])
            if path.exists() and (os.path.samefile(path,check) or os.path.samefile(path,specification)):
                raise ValueError('target aliases check or spec.md: '+entry['path'])
            if kind=='directories':
                if path.exists() and not path.is_dir():
                    raise ValueError('directory target is not a directory: '+entry['path'])
                ancestor=path.parent
                while not ancestor.exists():
                    ancestor=ancestor.parent
                if not ancestor.is_dir():
                    raise ValueError('directory parent is not a directory: '+entry['path'])
            else:
                if not path.is_file():
                    raise ValueError('required existing file missing: '+entry['path'])
                if kind=='configs':
                    with path.open('rb') as stream:
                        data=stream.read(65537)
                    if len(data)>65536:
                        raise ValueError('configuration is too large: '+entry['path'])
                    lines=data.decode('utf-8').splitlines(keepends=True)
                    for number,value in entry['lines']:
                        if number>len(lines):
                            raise ValueError('configuration lacks required line: '+entry['path'])
                        old=lines[number-1]
                        ending='\r\n' if old.endswith('\r\n') else '\n' if old.endswith('\n') else '\r' if old.endswith('\r') else ''
                        lines[number-1]=value+ending
                    edits.append((path,''.join(lines).encode('utf-8')))
            targets.append((kind,path,entry))
    existing=[path for _,path,_ in targets if path.exists()]
    if len({(path.stat().st_dev,path.stat().st_ino) for path in existing})!=len(existing):
        raise ValueError('repair targets alias each other')
    for kind,path,entry in targets:
        if kind=='directories':
            path.mkdir(parents=True,exist_ok=True)
            path.chmod(entry['mode'])
    for path,data in edits:
        path.write_bytes(data)
    for kind,path,entry in targets:
        if kind=='scripts':
            path.chmod(entry['mode'])
'''


def _command(workspace, expected_hash=None, plan=None, spec=None, expected_lf_hash=None):
    source = 'workspace=' + repr(workspace) + '\nexpected_hash=' + repr(expected_hash) + '\n'
    source += 'expected_lf_hash=' + repr(expected_lf_hash) + '\n'
    source += _PRELUDE
    if plan is not None:
        source += '\nplan=' + repr(plan) + '\nspec=' + repr(spec) + '\n' + _REPAIR
    source += '\ntry:\n'
    if plan is not None:
        source += '    repair()\n'
    source += ('    run_check()\nexcept Exception as error:\n'
               '    result["error"]=str(error)[:600]\n'
               '    result["error_code"]=getattr(error,"code","CHECK_OR_REPAIR_ERROR")\n'
               'print("' + MARKER + '"+json.dumps(result,ensure_ascii=False,separators=(",",":")))\n')
    return 'python3 -c ' + shlex.quote(source)


def repair_command(context):
    """Return one bounded repair-and-check command only for a complete known spec."""
    workspace = engineering_workspace(context)
    if not workspace or any((context.get('truncated') or {}).get(name)
                            for name in ('task_document', 'spec', 'output')):
        return None
    if any(tag in context['spec'] for tag in ('TRUNCATED', 'TIMEOUT', 'JUDGER_ERROR')):
        return None
    plan = _plan(context['spec'])
    if plan is None:
        return None
    command = _command(workspace, context.get('checker_sha256'), plan, context['spec'],
                       context.get('checker_lf_sha256'))
    # The 8k limit is for model-authored commands. The platform already accepts
    # larger bounded automatic commands (the bootstrap is about 11k).
    return command if len(command) <= 14000 else None


def check_command(workspace, expected_sha256=None, expected_lf_sha256=None):
    """Independently execute the unchanged checker, including after a model repair."""
    return _command(workspace, expected_sha256, expected_lf_hash=expected_lf_sha256)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate result key')
        result[key] = value
    return result


def _check_result(output, workspace):
    """Read an actual generated checker envelope, including internal failures."""
    if (not isinstance(output, str) or not output.startswith('[exitCode:0]\n')
            or len(output) > 24000 or not _absolute(workspace)
            or any(tag in output for tag in ('[TIMEOUT]', '[TRUNCATED]', '[JUDGER_ERROR]', '[NO_RESULT]'))):
        return None
    lines = [line for line in output.splitlines()[1:] if line.strip()]
    if len(lines) != 1 or not lines[0].startswith(MARKER):
        return None
    try:
        result = json.loads(lines[0][len(MARKER):], object_pairs_hook=_unique_object,
                            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (TypeError, ValueError, RecursionError):
        return None
    if (not isinstance(result, dict) or result.get('workspace') != workspace or 'returncode' not in result
            or (result.get('returncode') is not None and type(result['returncode']) is not int)
            or type(result.get('timed_out')) is not bool or type(result.get('truncated')) is not bool
            or not isinstance(result.get('stdout'), str)
            or not isinstance(result.get('stderr'), str)):
        return None
    return result


def check_failure(output, workspace):
    """Expose internal failure even though the Python wrapper itself exited 0."""
    result = _check_result(output, workspace)
    if result is None:
        return None
    if result.get('error'):
        code = result.get('error_code')
        return code if isinstance(code, str) and code else 'CHECK_ERROR'
    if result['timed_out']:
        return 'CHECK_TIMEOUT'
    if result['truncated']:
        return 'CHECK_TRUNCATED'
    if result['returncode'] != 0:
        return 'CHECK_FAILED'
    return None


def check_answer(output, workspace):
    """Accept one real token only from a successful, complete checker envelope."""
    result = _check_result(output, workspace)
    if (result is None or result['returncode'] != 0 or result['timed_out']
            or result['truncated'] or result.get('error')):
        return None
    stdout_lines = [line.strip() for line in result['stdout'].splitlines() if line.strip()]
    token_lines = [line for line in stdout_lines if line.startswith('TOKEN:')]
    if len(token_lines) != 1 or token_lines[0] != stdout_lines[-1] or 'TOKEN:' in result['stderr']:
        return None
    match = re.fullmatch(r'TOKEN:[ \t]*([A-Za-z0-9][A-Za-z0-9._:+/=-]{0,511})[ \t]*', token_lines[0])
    if not match or match.group(1).casefold() in ('xxx', 'timeout', 'truncated', 'null', 'none', 'example'):
        return None
    return {'kind': 'answer', 'answer': json.dumps({'token': match.group(1)}, separators=(',', ':')),
            'sop': '定位任务工作区，完整读取规范并修复指定目录、配置行和权限；运行原始 ./check，仅提交成功校验实际输出的 TOKEN。'}
