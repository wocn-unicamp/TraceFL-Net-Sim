package main

import (
	"flag"
	"log"

	"github.com/Marco-Guerra/Federated-Learning-Network-Workload/trace_driven_simulator/internal/simulator"
)

func main() {
	workloadBackgroundClients := flag.Uint("bg-workload", 4500000, "Workload traffic (b/s) generated from each background client")
	clientsBandwidthBps := flag.Uint("clients-b", 4500000, "Clients network devices bandwidth (b/s) in the simulated network")
	serverBandwidthBps := flag.Uint("server-b", 4000000, "Server network bandwidth (b/s) in the simulated network")
	traceFile := flag.String("t", "", "Trace file that describe the network workload during the simulation")

	flag.Parse()

	if *traceFile == "" {
		log.Panic("Trace file path must be given")
	}

	traceDrivenSimulator := simulator.New(&simulator.GlobalOptions{
		ClientsBandwidth:          uint32(*clientsBandwidthBps),
		ServerBandwidth:           uint32(*serverBandwidthBps),
		WorkloadBackgroundClients: uint32(*workloadBackgroundClients),
	})

	traceDrivenSimulator.RunSimulation(*traceFile)
}
