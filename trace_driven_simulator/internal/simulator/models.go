package simulator

import (
	"github.com/Marco-Guerra/Federated-Learning-Network-Workload/trace_driven_simulator/packages/writer"
)

const (
	SERVER_AGG_TIME    float32 = 60        // 60 s
	DOWNLINK_TIME      float32 = 30        // 30 s
	ETHERNET_HEADER    uint8   = 14        // 14 Bytes
	ETHERNET_MIN_FRAME uint8   = 64        // 64 Bytes
	ETHERNET_MTU       uint16  = 1500      // 1500 Bytes
	PROP_SPEED         float32 = 300000000 // 3 * 10**8 m/s
	CHANN_LEN          float32 = 1000      // 1 km
	EVAL_TIME          float64 = 1         // 1 s
)

type GlobalOptions struct {
	MaxNumberOfRounds         int
	ClientsBandwidth          uint32
	ServerBandwidth           uint32
	WorkloadBackgroundClients uint32
}

type TraceDriven struct {
	options        *GlobalOptions
	resultsWritter *writer.Writer
}
