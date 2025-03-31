import heapq  # Module for heap queue (priority queue) operations
import pandas as pd  # Module for handling data frames (used for reading the trace file)
import argparse

ETHERNET_MTU:int=1500 # MTU Ethernet é 1500 bits ou 188 bytes, mas manterei 1500 bytes pra não matar o esculapio
# Delay de Broadcast
# 0,805 s em datacenter (cross-silo)
# 800 s em mobile networks (cross-device)
CROSSSILO_BROADCAST_DELAY=0.805
CROSSDEVICE_BROADCAST_DELAY=800

class Packet:
    def __init__(self, arrival_time, size, packet_id):
        self.arrival_time = arrival_time  # Time when the packet arrives
        self.size = size  # Size of the packet in bytes
        self.start_service_time = None  # Time when the packet starts being processed
        self.departure_time = None  # Time when the packet leaves the queue
        self.id = packet_id

class EventQueueSimulator:
    def __init__(self, bandwidth_bps:int, mtu:int):
        self.queue:list[Packet] = []  # A list to store packets currently in the queue
        self.job_maxsize:int = mtu
        self.current_time:float = 0.0  # Current time in the simulation
        self.bandwidth_bps:int = bandwidth_bps  # Bandwidth of the server in bits per second
        self.events:list[tuple[float, int, int, int, Packet, str]] = []  # Priority queue (heap) to manage events
        self.results:list[tuple] = []
        self.results_filename = "metrics_network_"
        self.total_bytes:int = 0  # Total bytes processed
        self.total_delay:int = 0  # Total delay accumulated by all packets
        self.total_packets:int = 0  # Total number of packets processed


    def read_trace(self, trace_file:str, broadcast_delay:float):
        self.results_filename += "_".join(trace_file.split("_")[2:])

        df = pd.read_csv(trace_file)  # Read the trace CSV file
        packet_counter = 0
        current_time = 0.0

        rounds:int = df["round_number"].max()

        for round in range(1, rounds+1):
            clients = df[df['round_number'] == round]

            for _, row in clients.iterrows():  # Iterate through each row in the dataframe
                message_size:int = row['bytes_sended']

                packet = Packet(arrival_time=row['time'] + current_time, size=message_size, packet_id=packet_counter) # Create a packet object
                    
                heapq.heappush(self.events, (packet.arrival_time, round, row['client_id'], packet.id, packet, 'arrival'))  # Push packet arrival event into the heap
                packet_counter += 1
                
                current_time += float(clients['time'].max()) + broadcast_delay

    def process_events(self):
        while self.events:  # Continue until there are no more events
            time, round_number, client_id, pkid, packet, event_type = heapq.heappop(self.events)  # Pop the next event
            self.current_time = time  # Update current simulation time
            
            # Process the event based on the event type
            if event_type == 'arrival':  # If the event is an arrival                
                # Calculate the start service time based on the current queue status
                if not self.queue or self.queue[-1].departure_time <= packet.arrival_time: # If queue is empty or last packet has departed
                    packet.start_service_time = packet.arrival_time  # Start service immediately if queue is empty or last packet has departed
                else:
                    packet.start_service_time = self.queue[-1].departure_time  # Otherwise, wait until the last packet departs

                # Calculate the departure time based on the packet size and bandwidth
                packet.departure_time = packet.start_service_time + (packet.size * 8) / self.bandwidth_bps  # Compute departure time based on size and bandwidth
                heapq.heappush(self.events, (packet.departure_time, round_number, client_id, pkid, packet, 'departure'))  # Push departure event into the heap
                self.queue.append(packet)  # Add packet to the queue
            
            # Process the departure event
            elif event_type == 'departure':  # If the event is a departure
                self.total_bytes += packet.size  # Update total bytes processed
                self.total_delay += packet.departure_time - packet.arrival_time  # Update total delay
                self.total_packets += 1  # Increment total packet count
                delay = packet.departure_time - packet.arrival_time  # Calculate individual packet delay
                self.results.append((client_id, round_number, round(self.current_time, 3), round(delay, 3), packet.size))
                print(f"Arrived {packet.arrival_time} : Departed {packet.departure_time} : Delay {delay} : Size {packet.size}")

    def calculate_metrics(self):
        mean_delay = self.total_delay / self.total_packets if self.total_packets > 0 else 0
        throughput = (self.total_bytes * 8) / (self.current_time if self.current_time > 0 else 1)
        return mean_delay, throughput

    def run_simulation(self, trace_file, broadcast_delay:float):
        self.read_trace(trace_file, broadcast_delay)
        self.process_events()
        results_df = pd.DataFrame(self.results, columns=["client-id", "round_number", "time", "delay", "size"])
        results_df.to_csv(self.results_filename)
        return self.calculate_metrics()
    
def main():
    parser = argparse.ArgumentParser(
        prog='Trace-Driven Simulator',
        description='A simulator that receive a trace file and generate a simulation of the network behavior during the trace snapshot period'
    )

    parser.add_argument('-b', '--bandwidth', help='bandwidth of the simulated network', dest='bandwidth_bps', default=40000000)
    parser.add_argument('-t', '--trace-file', help='trace file that describe the network workload during the simulation', dest='trace_file', default=None, required=True)
    parser.add_argument('-mt', '--mtu', help='MTU of the packets in the network', dest='mtu', default=ETHERNET_MTU)
    parser.add_argument('-fs', '--federated-scenario', help='type of federated learning scenario: cross silo or cross-device', dest='federated_scenario', default='CROSSSILO')

    arguments = parser.parse_args()
    
    BANDWIDTH_BPS:int = int(arguments.bandwidth_bps)
    TRACEFILE:str = str(arguments.trace_file)
    MTU:int = int(arguments.mtu)
    BROADCAST_DELAY:float = CROSSDEVICE_BROADCAST_DELAY if arguments.federated_scenario and arguments.federated_scenario == "CROSSDEVICE" else CROSSSILO_BROADCAST_DELAY

    simulator = EventQueueSimulator(bandwidth_bps=BANDWIDTH_BPS, mtu=MTU)
    mean_delay, throughput = simulator.run_simulation(TRACEFILE, BROADCAST_DELAY)

    print(f"Mean Delay: {mean_delay} seconds")
    print(f"Throughput: {throughput} bits per second")

if __name__ == "__main__":
    main()