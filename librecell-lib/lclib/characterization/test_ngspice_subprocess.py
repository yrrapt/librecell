import tempfile
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os
import asyncio
import logging
import threading, queue
import time


def test_simple_simulation():
    """Run a simple transient simulation in a ngspice subprocess and retreive the results from a file."""

    sim_file = tempfile.mktemp()
    sim_output_file = tempfile.mktemp()

    spice_simulation_netlist = f"""
*
.title Simple RC circuit.

R1 VDD Y 1k
C1 Y GND 1u ic=0V
Vsrc_vdd VDD GND PWL(0 0 1ms 0V 2ms 1V)

.control
*option abstol=10e-15
*option reltol=10e-11
set filetype=ascii
* Enable output of vector names in the first line.
set wr_vecnames
tran 1ms 10ms
wrdata {sim_output_file} v(VDD) v(Y)
exit
.endc

.end
"""
    print(f"Write simulation file: {sim_file}")
    open(sim_file, 'w').write(spice_simulation_netlist)

    # Run simulation.
    ret = subprocess.run(["ngspice", sim_file])
    print(f"Subprocess return value: {ret}")
    if ret.returncode != 0:
        print(f"ngspice simulation failed: {ret}")
    assert ret.returncode == 0

    print(f"Read output data: {sim_output_file}")
    data = np.loadtxt(sim_output_file, skiprows=1)  # Skip the header.

    a_time = data[:, 0]
    a = data[:, 1]

    y_time = data[:, 2]
    y = data[:, 3]

    assert all(a_time == y_time)

    plt.plot(a_time, a, 'x-')
    plt.plot(y_time, y, 'x-')
    plt.show()

    os.remove(sim_output_file)
    os.remove(sim_file)


def test_interactive_subprocess():
    import subprocess
    p = subprocess.Popen(['ngspice', '-p'],
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         universal_newlines=True)

    p.stdin.write("a\n")
    p.stdin.flush()
    print(p.stderr.readline())

    p.stdin.write("quit\n")
    p.stdin.flush()
    print(p.stdout.readlines())
    print(p.stderr.readlines())


class NgSpiceInteractive:
    """
    Interface to ngspice as a subprocess.
    """

    def __init__(self, logger: logging.Logger = None):
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger
        self.ngspice_cmd = "ngspice"
        self.ngspice_args = ['-p']
        "Run ngspice in interactive pipe mode."

        self.proc: subprocess.Popen[str] = None
        "ngspice process handle."

        self._from_stdout = queue.Queue()
        self._from_stderr = queue.Queue()

    def start(self):
        self.proc = subprocess.Popen(['ngspice', '-p'],
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     universal_newlines=True)

        def read_stdout():
            while True:
                line = self.proc.stdout.readline()
                if line is None:
                    break
                # print("stdout:", line)
                self._from_stdout.put(line)

        def read_stderr():
            while True:
                line = self.proc.stderr.readline()
                if line is None:
                    break
                # print("stderr:", line)
                self._from_stderr.put(line)

        stdout_thread = threading.Thread(target=read_stdout, daemon=True)
        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stdout_thread.start()
        stderr_thread.start()

    def readline(self, timeout=1) -> str:
        try:
            return self._from_stdout.get(block=True, timeout=timeout)
        except queue.Empty:
            return None

    def drop_stdout(self):
        """
        Delete the stdout buffer.
        """
        while not self._from_stdout.empty():
            self._from_stdout.get_nowait()

    def drop_stderr(self):
        """
        Delete the stderr buffer.
        """
        while not self._from_stdout.empty():
            self._from_stdout.get_nowait()

    def readline_err(self) -> str:
        try:
            return self._from_stderr.get(block=True, timeout=1)
        except queue.Empty:
            return None

    def cmd(self, cmd: str):
        self.proc.stdin.write(cmd)
        self.proc.stdin.write('\n')
        self.proc.stdin.flush()

    def source(self, file: str):
        """
        Load a netlist.
        """
        self.cmd(f"source {file}")

    def get_data(self, signals):
        self.drop_stdout()
        self.cmd(f"print {signals}")
        # time.sleep(1)

        num_rows = 0
        while True:
            line = self.readline(timeout=None)
            assert line is not None
            if 'No. of Data Rows' in line:
                _, num_rows_str = line.split(':', 2)
                num_rows = int(num_rows_str)
                break
        print(f"Number of rows: {num_rows}")
        assert num_rows > 0

        rows = []
        i = 0
        while True:
            line = self.readline()
            assert line is not None
            if line.startswith(str(i)):
                i = i + 1
                arr = np.fromstring(line, dtype=float, sep=' ')
                rows.append(arr)
            if i == num_rows:
                break

        data = np.array(rows)
        # Remove index.
        data = data[:, 1:]
        print(data.shape)


def test_ngspice_subprocess_class():
    sim_file = tempfile.mktemp()
    sim_output_file = tempfile.mktemp()

    spice_simulation_netlist = f"""
*
.title Simple RC circuit.

R1 VDD Y 1k
C1 Y GND 1u ic=0V
Vsrc_vdd VDD GND PWL(0 0 1ms 0V 2ms 1V)

*.control
*option abstol=10e-15
*option reltol=10e-11
*set filetype=ascii
* Enable output of vector names in the first line.
*set wr_vecnames
*tran 1ms 10ms
*wrdata {sim_output_file} v(VDD) v(Y)

*.endc

.end
    """
    print(f"Write simulation file: {sim_file}")
    open(sim_file, 'w').write(spice_simulation_netlist)

    ns = NgSpiceInteractive()
    ns.start()

    ns.source(sim_file)
    # ns.cmd(spice_simulation_netlist)
    ns.cmd('set filetype=ascii')
    ns.cmd('tran 1ms 10ms')
    # ns.cmd('wrdata /dev/stdout v(VDD) v(Y)')
    ns.get_data('v(VDD) v(Y)')

    line = ns.readline()
    while line is not None:
        print(line.strip())
        line = ns.readline()

    ns.cmd('quit')


class NgSpiceAsync:
    """
    Asynchronous interface to ngspice as a subprocess.
    """

    def __init__(self, logger: logging.Logger = None):
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger
        self.ngspice_cmd = "ngspice"
        self.ngspice_args = ['-p']
        "Run ngspice in interactive pipe mode."

        self.proc = None
        "ngspice process handle."

    async def start(self):
        self.proc = await asyncio.create_subprocess_exec(
            self.ngspice_cmd,
            *self.ngspice_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def cmd(self, cmd: bytes):
        self.logger.debug(f"Run cmd: {cmd}")
        print(f"Run cmd: {cmd}")
        # cmd = cmd.encode('utf-8')
        stdout, stderr = await self.proc.communicate(input=cmd)
        self.logger.debug(f"stdout: {stdout}")
        print(f"stdout: {stdout}")
        self.logger.debug(f"stderr: {stderr}")
        print(f"stderr: {stderr}")

        print(f"return code: {self.proc.returncode}")


def test_async_interactive_subprocess():
    async def run():
        ns = NgSpiceAsync()
        await ns.start()
        await ns.cmd(b"help")
        await ns.cmd(b"help")
        await ns.cmd(b"quit")

    asyncio.run(run())


def test_async_interactive_subprocess_raw():
    import asyncio

    async def run():
        cmd = 'ngspice -p'
        proc = await asyncio.create_subprocess_exec(
            'ngspice',
            '-p',
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate(input=b"help\n")
        print(stdout)
        print(stderr)
        stdout, stderr = await proc.communicate(input=b"quit\n")

        print(f'[{cmd!r} exited with {proc.returncode}]')
        if stdout:
            print(f'[stdout]\n{stdout.decode()}')
        if stderr:
            print(f'[stderr]\n{stderr.decode()}')

    asyncio.run(run())
