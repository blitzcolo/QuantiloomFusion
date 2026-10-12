"""Single-writer lock for a dataset directory and live-process checks."""

import ctypes
import os
import socket
import uuid
from contextlib import contextmanager

from .jsonio import atomic_json, load_json


def alive(pid):
    if not isinstance(pid,int) or pid<=0:return False
    if os.name=='nt':
        k=ctypes.WinDLL('kernel32',use_last_error=True)
        k.OpenProcess.restype=ctypes.c_void_p
        h=k.OpenProcess(0x1000,False,pid)
        if not h:return ctypes.get_last_error()!=87  # access denied is not proof of exit
        try:
            code=ctypes.c_ulong()
            if not k.GetExitCodeProcess(ctypes.c_void_p(h),ctypes.byref(code)):return True
            return code.value==259
        finally:k.CloseHandle(ctypes.c_void_p(h))
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False
    except PermissionError:return True


@contextmanager
def runner_lock(out):
    path=out/'.generator.lock';token=uuid.uuid4().hex
    payload={'pid':os.getpid(),'host':socket.gethostname(),'token':token,'child_pid':None}
    mutex=(out/'.writer.mutex').open('a+b')
    mutex.seek(0);mutex.write(b'0');mutex.flush();mutex.seek(0)
    try:
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(mutex.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(mutex.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as e:raise ValueError('a dataset planner/runner is active; no second writer is allowed') from e
        if path.exists():
            prior=load_json(path)
            if prior.get('host')!=socket.gethostname() or alive(prior.get('child_pid')):
                raise ValueError('an orphaned render process or another host may still be active')
        atomic_json(path,payload)
        try:yield payload,path
        finally:
            if path.exists() and load_json(path).get('token')==token:path.unlink()
    finally:
        mutex.close()  # operating system releases the lock even after a crash
