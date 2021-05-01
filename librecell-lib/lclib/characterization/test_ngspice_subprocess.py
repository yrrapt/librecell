import tempfile
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os
import asyncio
import logging


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
