import warnings
from typing import List


class Clock:
    def __init__(self):
        self.time = 0

    def increment(self, units: int = 1):
        self.time += units
        return self.time

    def current_time(self):
        return self.time


class Process:
    def __init__(
        self,
        process_num: int,
        arrival: int,
        cpu_times: List[int],
        io_times: List[int],
    ):
        self.process_num = process_num
        self.arrival_time = arrival
        self.cpu_times = cpu_times
        self.io_times = io_times
        self.service_time = sum(self.cpu_times)
        self.io_time = sum(self.io_times)
        self.blocked: bool = False
        self.finish_time = None

    def __str__(self):
        return (
            f"Process number {self.process_num}\n"
            f"Arrives at {self.arrival_time}\n"
            f"Requires {self.service_time} units of processing in bursts of {self.cpu_times}\n"
            f"Requires {self.io_time} units of I/O in periods of {self.io_times}\n"
        )


class CPU:
    def __init__(
        self,
        switch_time: int,
        initial_process: Process = None,
    ):
        self.context_switch_time = switch_time
        self.current_process = initial_process
        self.clock = Clock()

    # process the current process for given units of time
    # or until the current CPU burst is completed, whichever is shorter
    def process(self, units: int):
        if self.current_process.blocked:  # do not process a process waiting on I/O
            warnings.warn(f"Attempted to run the blocked process \"{self.current_process.process_num}\" at time {self.clock.current_time()}")
            return
        if units >= self.current_process.cpu_times[0]:
            self.clock.increment(self.current_process.cpu_times[0])  # advance clock
            self.current_process.cpu_times.pop(0)  # process the process
            self.current_process.blocked = True  # mark the process as waiting for I/O
        else:
            self.clock.increment(units)  # advance clock
            self.current_process.cpu_times[0] -= units  # process the process


if __name__ == '__main__':
    with open("input.txt", "r") as file:
        num_processes, switch_time = (int(s) for s in file.readline().split(" "))
        processes = []
        for i in range(1, num_processes + 1):
            process_num, arrival_time, num_cycles = [int(s) for s in file.readline().split(" ")]
            cpu_times = []
            io_times = []
            for j in range(1, num_cycles):
                cycle_num, cpu_time, io_time = [int(s) for s in file.readline().split(" ")]
                cpu_times.append(cpu_time)
                io_times.append(io_time)
            cycle_num, cpu_time = [int(s) for s in file.readline().split(" ")]  # handle the last cpu burst
            cpu_times.append(cpu_time)
            process = Process(process_num, arrival_time, cpu_times, io_times)
            processes.append(process)

    cpu = CPU(switch_time, processes[0])

    for process in processes:
        print(process)