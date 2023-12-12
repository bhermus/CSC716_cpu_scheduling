import random
import warnings
from collections import defaultdict
from contextlib import suppress
from enum import Enum
from typing import List, Dict


class Clock:
    def __init__(self, initial_time: int = 0):
        self.time = initial_time

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
        if self.current_process is None:  # if there is no current process, obviously don't run it
            return units
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
    QUANTUM = 2


class Event:
    def __init__(
        self,
        process: Process,
        state: EventType = None
    ):
        self.process = process
        if state:
            self.state = state
        else:
            # by default, event occurring is the state the process is NOT currently in
            self.state = EventType(1) if process.blocked else EventType(0)


class Scheduler:
    def __init__(
        self,
        cpu: CPU = None,
        event_queue: Dict[int, List[Event]] = defaultdict(list),
        io_processes: List[Process] = [],
    ):
        self.cpu = cpu  # the CPU object the scheduler will be managing
        self.process_queue: List[Process] = []  # all processes ready for the CPU
        self.event_queue: Dict[int, List[Event]] = event_queue  # maps clock times to Event object relevant to that time
        self.io_processes = io_processes  # all processes currently blocked and undergoing I/O operations
        self.clock = Clock(min(self.event_queue.keys()))  # keeps track of the current time, initialized to first Event

    # performs I/O operations on all the io_processes for given units
    def perform_io(self, units: int):
        for process in self.io_processes:
            with suppress(IndexError):  # ignore index errors that get thrown if the process has no more I/O to run
                process.io_times[0] -= units
                if process.io_times[0] <= 0:  # i.e. I/O operations are completed
                    process.io_times.pop(0)
                    process.blocked = False
                    self.io_processes.remove(process)

    def switch_process(self, process: Process):
        # units = self.cpu.context_switch_time  # TODO reincorporate context switch time
        # self.clock.increment(units)
        # self.perform_io(units)
        self.cpu.current_process = process

    # runs the simulation for given units of time
    def run(self, units: int = None):
        self.cpu.process(units)
        self.perform_io(units)
        self.clock.increment(units)

    def fcfs(self):
        while self.event_queue:
            to_print = ""
            for et, events in event_queue.items():
                for event in events:
                    to_print += f"|{et} {event.state.name} {event.process.process_num}|"
            print(self.clock.current_time(), to_print)

            event_time = min(self.event_queue.keys())  # currently occurring event
            event = self.event_queue[event_time][0]

            if event.state == EventType.READY:
                event.process.blocked = False  # mark the process as ready
                if self.cpu.state == State.AVAILABLE:  # if the CPU is available,
                    self.cpu.state = State.BUSY
                    self.switch_process(event.process)  # switch to the newly ready process
                    self.event_queue[self.clock.current_time() + event.process.cpu_times[0]].append(Event(event.process))  # create a new Event at which the process will be WAITING
                elif self.cpu.state == State.BUSY:
                    self.process_queue.append(event.process)  # else add process to BACK of queue
            elif event.state == EventType.WAITING:
                event.process.blocked = True  # mark the process as waiting
                self.io_processes.append(event.process)
                if event.process.io_times:  # if the process has I/O to handle
                    self.event_queue[self.clock.current_time() + event.process.io_times[0]].append(Event(event.process))  # create a new Event at which the process will be READY
                if self.process_queue:  # if there are processes ready
                    next_process = self.process_queue.pop(0)  # take process from FRONT of queue
                    self.switch_process(next_process)
                    self.event_queue[self.clock.current_time() + next_process.cpu_times[0]].append(Event(next_process))  # create a new Event at which the process will be WAITING
                else:
                    self.cpu.state = State.AVAILABLE
                    self.cpu.current_process = None

            event_queue[event_time].pop(0)  # remove this Event

            if len(event_queue[event_time]) == 0:  # if there are no more Events at this time
                event_queue.pop(event_time)

            if self.event_queue:
                next_event_time = min(self.event_queue.keys())  # soonest (i.e. next) occurring event
                self.run(next_event_time - event_time)
            else:  # this is the last event
                pass

            # print(self.clock.current_time(), event.process.process_num, event.process.cpu_times, event.process.io_times)

    def sjn(self):
        while self.event_queue:
            to_print = ""
            for et, events in event_queue.items():
                for event in events:
                    to_print += f"|{et} {event.state.name} {event.process.process_num}|"
            print(self.clock.current_time(), to_print)

            event_time = min(self.event_queue.keys())  # currently occurring event
            event = self.event_queue[event_time][0]

            if event.state == EventType.READY:
                event.process.blocked = False  # mark the process as ready
                if self.cpu.state == State.AVAILABLE:  # if the CPU is available,
                    self.cpu.state = State.BUSY
                    self.switch_process(event.process)  # switch to the newly ready process
                    self.event_queue[self.clock.current_time() + event.process.cpu_times[0]].append(Event(event.process))  # create a new Event at which the process will be WAITING
                elif self.cpu.state == State.BUSY:
                    # see which processes in the queue (if any) have shorter job length
                    index = 0
                    for p in self.process_queue:
                        if event.process.cpu_times[0] >= p.cpu_times[0]:
                            index += 1
                    self.process_queue.insert(index, event.process)  # insert process in queue AFTER all processes whose next CPU burst is shorter
            elif event.state == EventType.WAITING:
                event.process.blocked = True  # mark the process as waiting
                self.io_processes.append(event.process)
                if event.process.io_times:  # if the process has I/O to handle
                    self.event_queue[self.clock.current_time() + event.process.io_times[0]].append(Event(event.process))  # create a new Event at which the process will be READY
                if self.process_queue:  # if there are processes ready
                    next_process = self.process_queue.pop(0)  # take process from FRONT of queue
                    self.switch_process(next_process)
                    self.event_queue[self.clock.current_time() + next_process.cpu_times[0]].append(Event(next_process))  # create a new Event at which the process will be WAITING
                else:
                    self.cpu.state = State.AVAILABLE
                    self.cpu.current_process = None

            event_queue[event_time].pop(0)  # remove this Event

            if len(event_queue[event_time]) == 0:  # if there are no more Events at this time
                event_queue.pop(event_time)

            if self.event_queue:
                next_event_time = min(self.event_queue.keys())  # soonest (i.e. next) occurring event
                self.run(next_event_time - event_time)
            else:  # this is the last event
                pass

    def rr(self, quantum: int = 10):
        while self.event_queue:
            to_print = ""
            for et, events in event_queue.items():
                for event in events:
                    to_print += f"|{et} {event.state.name} {event.process.process_num}|"
            print(self.clock.current_time(), to_print)

            event_time = min(self.event_queue.keys())  # currently occurring event
            event = self.event_queue[event_time][0]

            if event.state == EventType.QUANTUM:
                if self.process_queue:  # if there are processes ready
                    self.process_queue.append(event.process)
                    next_process = self.process_queue.pop(0)  # take process from FRONT of queue
                    self.switch_process(next_process)
                    if next_process.cpu_times[0] > quantum:  # if it will take more than one quantum to complete processing
                        self.event_queue[self.clock.current_time() + quantum].append(Event(next_process, EventType.QUANTUM))  # create a new Event at which the QUANTUM ends
                    else:
                        self.event_queue[self.clock.current_time() + next_process.cpu_times[0]].append(Event(next_process))  # create a new Event at which the process will be WAITING
                else:  # no processes are waiting
                    if event.process.cpu_times[0] > quantum:
                        self.event_queue[self.clock.current_time() + quantum].append(Event(event.process, EventType.QUANTUM))  # create a new Event at which the QUANTUM ends
                    else:
                        self.event_queue[self.clock.current_time() + event.process.cpu_times[0]].append(Event(event.process))  # create a new Event at which the process will be WAITING

            elif event.state == EventType.READY:
                event.process.blocked = False  # mark the process as ready
                if self.cpu.state == State.AVAILABLE:  # if the CPU is available,
                    self.cpu.state = State.BUSY
                    self.switch_process(event.process)  # switch to the newly ready process
                    if event.process.cpu_times[0] > quantum:
                        self.event_queue[self.clock.current_time() + quantum].append(Event(event.process, EventType.QUANTUM))  # create a new Event at which the QUANTUM ends
                    else:
                        self.event_queue[self.clock.current_time() + event.process.cpu_times[0]].append(Event(event.process))  # create a new Event at which the process will be WAITING
                elif self.cpu.state == State.BUSY:
                    self.process_queue.append(event.process)  # else add process to BACK of queue

            elif event.state == EventType.WAITING:
                event.process.blocked = True  # mark the process as waiting
                self.io_processes.append(event.process)
                if event.process.io_times:  # if the process has I/O to handle
                    self.event_queue[self.clock.current_time() + event.process.io_times[0]].append(Event(event.process))  # create a new Event at which the process will be READY
                if self.process_queue:  # if there are processes ready
                    next_process = self.process_queue.pop(0)  # take process from FRONT of queue
                    self.switch_process(next_process)
                    if next_process.cpu_times[0] > quantum:  # if it will take more than one quantum to complete processing
                        self.event_queue[self.clock.current_time() + quantum].append(Event(next_process, EventType.QUANTUM))  # create a new Event at which the QUANTUM ends
                    else:
                        self.event_queue[self.clock.current_time() + next_process.cpu_times[0]].append(Event(next_process))  # create a new Event at which the process will be WAITING
                else:
                    self.cpu.state = State.AVAILABLE
                    self.cpu.current_process = None

            event_queue[event_time].pop(0)  # remove this Event

            if len(event_queue[event_time]) == 0:  # if there are no more Events at this time
                event_queue.pop(event_time)

            if self.event_queue:
                next_event_time = min(self.event_queue.keys())  # soonest (i.e. next) occurring event
                self.run(next_event_time - event_time)
            else:  # this is the last event
                pass

def generate_input_file():
    with open("input1.txt", "w") as file:
        num_processes = 50
        switch_time = 5

        file.write(f"{num_processes} {switch_time}\n")

        arrival_interval_mean = 50
        for i in range(1, num_processes + 1):
            arrival_time = round(random.expovariate(1 / arrival_interval_mean))
            num_cycles = random.randint(10, 30)
            if i == 1:
                file.write(f"{i} {0} {num_cycles}\n")
            else:
                file.write(f"{i} {arrival_time} {num_cycles}\n")

            for cycle in range(1, num_cycles):
                cpu_time = random.randint(5, 400)
                io_time = random.randint(30, 200)
                file.write(f"{cycle} {cpu_time} {io_time}\n")
            # Check if it's the last iteration before writing the last line
            file.write(f"{num_cycles} {cpu_time}\n")
            
if __name__ == '__main__':
    generate_input_file()
    event_queue = defaultdict(list)
    with open("input1.txt", "r") as file:
        num_processes, switch_time = (int(s) for s in file.readline().split(" "))
        processes = []
        for i in range(1, num_processes + 1):
            process_num, arrival_time, num_cycles = [int(s) for s in file.readline().split(" ")]
            cpu_times = []
            io_times = []
            for j in range(num_cycles - 1):  # Adjusted loop range
                cycle_num, cpu_time, io_time = [int(s) for s in file.readline().split()]
                cpu_times.append(cpu_time)
                io_times.append(io_time)
            # Reading the last cycle differently if there are only two values
            line = file.readline().split()
            if len(line) == 2:  # Check for two values
                cpu_times.append(int(line[1]))
            else:
                cycle_num, cpu_time, io_time = [int(s) for s in line]
                cpu_times.append(cpu_time)
            process = Process(process_num, arrival_time, cpu_times, io_times)
            event_queue[arrival_time].append(Event(process))
            processes.append(process)

    # for i, j in event_queue.items():
    #     print(i, j.process.process_num, j.state)

    for process in processes:
        print(process)

    cpu = CPU(switch_time)
    scheduler = Scheduler(cpu, event_queue)
    scheduler.rr()

    for process in processes:
        print(process)
    # print(scheduler.cpu.current_process)
    # print(scheduler.clock.current_time())
    # print()
    # scheduler.run(5)
    # print(scheduler.cpu.current_process)
    # print(scheduler.clock.current_time())
