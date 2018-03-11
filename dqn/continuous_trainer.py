import os
import subprocess
import threading

import sys
from signal import *



cur_path = os.path.dirname(os.path.realpath(__file__))
env = dict()
train_thread = None
enjoy_thread = None


def popenAndCall(onExit, *popenArgs, **popenKWArgs):
    def runInThread(onExit, popenArgs, popenKWArgs):
        proc = subprocess.Popen(*popenArgs, **popenKWArgs)
        proc.wait()
        onExit()
        return
    thread = threading.Thread(target=runInThread,
                              args=(onExit, popenArgs, popenKWArgs))
    thread.start()
    return thread


def train():
    global train_thread, enjoy_thread
    if enjoy_thread is not None:
        try: enjoy_thread.terminate()
        except Exception: pass
        enjoy_thread = None

    file_to_execute = cur_path+'/train_cms_32_32_auto.py'
    args = [file_to_execute] # , more args
    train_thread = popenAndCall(enjoy, [sys.executable] + args, cwd=cur_path, env=dict(os.environ, **env))


def enjoy():
    global train_thread, enjoy_thread
    if train_thread is not None:
        try: train_thread.terminate()
        except Exception: pass
        train_thread = None

    file_to_execute = cur_path+'/enjoy_cms_32_32_auto.py'
    args = [file_to_execute] # , more args
    enjoy_thread = popenAndCall(train, [sys.executable] + args, cwd=cur_path, env=dict(os.environ, **env))

def clean(*args):
    print("-- clean was called --")
    global train_thread, enjoy_thread
    if train_thread is not None:
        try: train_thread.terminate()
        except Exception: pass
        train_thread = None
    if enjoy_thread is not None:
        try: enjoy_thread.terminate()
        except Exception: pass
        enjoy_thread = None
    os._exit(0)

def main():
    for sig in (SIGABRT, SIGIOT, SIGINT, SIGBUS, SIGTERM, SIGFPE, SIGHUP, SIGTSTP, SIGTTIN, SIGTTOU,
                SIGILL, SIGQUIT, SIGSEGV, SIGALRM, SIGPIPE, SIGPROF, 
                SIGSYS, SIGTRAP, SIGUSR1, SIGUSR2, SIGVTALRM, SIGXCPU, SIGXFSZ):
        signal(sig, clean)

    train()
    # while True:
    #     train()
    #     enjoy()

if __name__ == '__main__':
    main()
