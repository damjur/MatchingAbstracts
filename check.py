# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
import subprocess
import time


def run(popenargs):
    with subprocess.Popen(popenargs, stdout=subprocess.PIPE) as process:
        try:
            output, unused_err = process.communicate()
        except subprocess.TimeoutExpired:
            process.kill()
            output, unused_err = process.communicate()
        except:
            process.kill()
            process.wait()
            raise
        retcode = process.poll()
        return output.decode("utf-8").splitlines()

def checkDir(commanddir, path, N):
    with open(os.path.join(path, 'correct.txt'), 'rt') as f:
        correct = f.read().splitlines()
    
    cmd = ['./run.sh', path, str(N)]
    cwd = os.getcwd()
    os.chdir(commanddir)
    start = time.time()
    output = run(cmd)
    stop = time.time()
    os.chdir(cwd)
    
    size = 6
    result = np.zeros(size)
    
    if len(output) != len(correct):
        return result, stop-start;
    
    for line, c in zip(output, correct):
        line = line.split()
        try:
            idx = line.index(c)
        except:
            idx = N - 1
        if idx < size-1:
            result[idx] += 1
        result[-1] += 1.0 / (1.0 + idx)
    
    return result, stop-start;

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(os.path.basename(sys.argv[0]) + " <katalog_z_programem_run.sh> <sciezka_do_katalogu_ze_zbiorami_danych>")
        exit(1)
    
    programdir_path = os.path.abspath(sys.argv[1])
    data_path = os.path.abspath(sys.argv[2])
    
    print("AUTORZY:")
    with open(os.path.join(programdir_path, 'autorzy.txt'), 'rt') as f:
        for l in f.readlines():
            print(l.strip())
    
    program_path = os.path.join(programdir_path, 'run.sh')
    os.chmod(program_path, os.stat(program_path).st_mode | 0o100)#stat.S_IEXEC)
    
    dirs = [('set0', 6),
            ('set1', 20),
            ('set2', 20),
            ('set3', 20),
            ('set4', 20),
            ('set5', 200),
            ('set6', 200),
            ('set7', 20),
            ('set8', 100)]
    
    print("WYNIKI:")
    total = []
    times = []
    for d in dirs:
        res, t = checkDir(programdir_path, os.path.join(data_path, d[0]), d[1])
        total.append(res)
        times.append(t)
        print(d[0],'=', res[:-1], 'score =', res[-1], "[%dsec]" % t)
    
    print("----")
    summary = np.array(total).sum(axis=0)
    print(summary[:-1], 'score =', summary[-1], "[%dsec]" % sum(times))
