package main

import (
	"flag"
	"log"

	"github.com/wocn-unicamp/TraceFL-Net-Sim/trace_driven_simulator/internal/simulator"
)

func main() {
	backgroundTrafficLoad := flag.Float64("bg-workload", 0.3, "Workload traffic (percentage) generated from each background client")
	clientsBandwidthBps := flag.Uint("clients-b", 4500000, "Clients network devices bandwidth (b/s) in the simulated network")
	serverBandwidthBps := flag.Uint("server-b", 4000000, "Server network bandwidth (b/s) in the simulated network")
	seed := flag.Uint64("seed", 0, "Seed for the random number generator. If 0, the current time will be used as seed")
	earlyStopping := flag.Int("early-stop", -1, "Max number of rounds to simulate. If -1, the simulation will run until the end of the trace")
	backgroundTrafficType := flag.String("bg-model", "POISSON", "Probalistic model for background traffic generation. Options: POISSON, PARETO, ONOFF")
	bufferSize := flag.Uint("buffer-size", 0, "Maximum buffer size (in packets) for the network devices. If 0, the buffer is considered infinite")
	traceFile := flag.String("t", "", "Trace file that describe the network workload during the simulation")
	retransmission := flag.Bool("retransmission", false, "Enable packet retransmission on packet drop")
	retransmissionBackoff := flag.Float64("retransmission-backoff", 1, "Explicit backoff time (seconds) for packet retransmission. If 0, the RTT is used as backoff time")

	flag.Parse()

	if *traceFile == "" {
		log.Panic("Trace file path must be given")
	}

	if *backgroundTrafficType != "POISSON" && *backgroundTrafficType != "PARETO" && *backgroundTrafficType != "ONOFF" {
		log.Panic("Invalid background traffic model. Please choose from: POISSON, PARETO, ONOFF")
	}

	traceDrivenSimulator := simulator.New(&simulator.GlobalOptions{
		ClientsBandwidth:       uint32(*clientsBandwidthBps),
		Seed:                   *seed,
		ServerBandwidth:        uint32(*serverBandwidthBps),
		MaxNumberOfRounds:      *earlyStopping,
		BackgroundTrafficLoad:  *backgroundTrafficLoad,
		BackgroundTrafficModel: simulator.TrafficModel(*backgroundTrafficType),
		InfiniteBuffer:         *bufferSize == 0,
		MaxQueueSize:           uint16(*bufferSize),
		EnableRetransmission:   *retransmission,
		RetransmissionBackoff:  *retransmissionBackoff,
	})

	traceDrivenSimulator.RunSimulation(*traceFile)
}
