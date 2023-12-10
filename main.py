import warnings
from collections import defaultdict
from enum import Enum
from typing import List, Dict


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
        self.blocked: bool = True  # Every process is considered blocked until it arrives
        self.finish_time = None

    def __str__(self):
        return (
            f"Process number {self.process_num}\n"
            f"Arrives at {self.arrival_time}\n"
            f"Requires {self.service_time} units of processing in bursts of {self.cpu_times}\n"
            f"Requires {self.io_time} units of I/O in periods of {self.io_times}\n"
        )


class State(Enum):
    BUSY = 0
    AVAILABLE = 1


class CPU:
    def __init__(
        self,
        switch_time: int,
        initial_process: Process = None,
    ):
        self.context_switch_time = switch_time
        self.current_process = initial_process
        self.state = State(1)  # CPU is available by default

    # process the current process for given units of time
    # or until the current CPU burst is completed, whichever is shorter.
    # returns the actual time units passed
    def process(self, units: int = None) -> int:
        if self.current_process.blocked:  # do not process a process waiting on I/O
            warnings.warn(f"Attempted to run the blocked process \"{self.current_process.process_num}\"")
            return 0
        if units is None or units >= self.current_process.cpu_times[0]:
            units = self.current_process.cpu_times[0]
            self.current_process.cpu_times.pop(0)  # process the process
            self.current_process.blocked = True  # mark the process as waiting for I/O
            return units
        else:
            self.current_process.cpu_times[0] -= units  # process the process
            return units


class EventType(Enum):
    WAITING = 0
    READY = 1


class Event:
    def __init__(
        self,
        process: Process,
    ):
        self.process = process
        # event occurring is the state the process is NOT currently in
        self.state = EventType(1) if process.blocked else EventType(0)


class Scheduler:
    def __init__(
        self,
        cpu: CPU = None,
        event_queue: Dict[int, List[Event]] = defaultdict(list),
        io_processes: List[Process] = [],
    ):
        self.cpu = cpu  # the CPU object the scheduler will be managing
        self.process_queue = []  # all processes ready for the CPU
        self.event_queue: Dict[int, Event] = event_queue  # maps clock times to Event object relevant to that time
        self.io_processes = io_processes  # all processes currently blocked and undergoing I/O operations
        self.clock = Clock()  # keeps track of the current time passed

    # performs I/O operations on all the io_processes for given units
    def perform_io(self, units: int):
        for process in self.io_processes:
            process.io_times[0] -= units
            if process.io_times[0] <= 0:  # i.e. I/O operations are completed
                process.io_times.pop(0)
                process.blocked = False
                self.process_queue.append(process)
                self.event_queue[self.clock.current_time()].append(Event(process))
                self.io_processes.remove(process)

    def switch_process(self, process: Process):
        units = self.cpu.context_switch_time
        self.clock.increment(units)
        self.perform_io(units)
        self.cpu.current_process = process

    # runs the simulation for given units of time
    def run(self, units: int = None):
        units = self.cpu.process(units)
        self.perform_io(units)
        self.clock.increment(units)


def first_come_first_serve(sch: Scheduler):
    while sch.process_queue or sch.io_processes:  # while there are still processes
        if sch.cpu.current_process.blocked:
            for process in sch.process_queue:
                if not process.blocked:
                    sch.switch_process(process)  # switch to first not blocked process
                    break
        print(sch.cpu.current_process)
        if sch.clock.current_time() < cpu.current_process.arrival_time:
            sch.run(sch.cpu.current_process.arrival_time - sch.clock.current_time())  # fast-forward to arrival time if needed

        sch.run()


if __name__ == '__main__':
    event_queue = defaultdict(list)
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
            event_queue[arrival_time].append(Event(process))
            processes.append(process)

    cpu = CPU(switch_time, processes[0])
    scheduler = Scheduler(cpu, processes)
    first_come_first_serve(scheduler)

    # for process in processes:
    #     print(process)
    # print(scheduler.cpu.current_process)
    # print(scheduler.clock.current_time())
    # print()
    # scheduler.run(5)
    # print(scheduler.cpu.current_process)
    # print(scheduler.clock.current_time())