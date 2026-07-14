package simulator

import (
	"github.com/wocn-unicamp/TraceFL-Net-Sim/trace_driven_simulator/packages/writer"
)

const (
	SERVER_AGG_TIME      float32 = 60        // 60 s
	DOWNLINK_TIME        float32 = 30        // 30 s
	ETHERNET_HEADER      uint8   = 14        // 14 Bytes
	ETHERNET_MIN_FRAME   uint8   = 64        // 64 Bytes
	ETHERNET_MTU         uint16  = 1500      // 1500 Bytes
	PROP_SPEED           float32 = 300000000 // 3 * 10**8 m/s
	CHANN_LEN            float32 = 1000      // 1 km
	EVAL_TIME            float64 = 1         // 1 s
	ALPHA_BG             float64 = 1.5       // Heavy-tail web browsing model parameter for Pareto distribution
	ON_MEAN_BG           float64 = 0.1       // Mean ON time for ON/OFF background traffic model (in seconds)
	OFF_MEAN_BG          float64 = 0.9       // Mean OFF time for ON/OFF background traffic model (in seconds)
	INTERNET_JITTER_MEAN float64 = 0.05      // 50ms average router queueing jitter for Shifted Exponential
)

type TrafficModel string

const (
	POISSON TrafficModel = "POISSON"
	PARETO  TrafficModel = "PARETO"
	ONOFF   TrafficModel = "ONOFF"
)

type GlobalOptions struct {
	MaxNumberOfRounds      int
	ClientsBandwidth       uint32
	ServerBandwidth        uint32
	Seed                   uint64
	BackgroundTrafficLoad  float64
	BackgroundTrafficModel TrafficModel
}

type TraceDriven struct {
	options        *GlobalOptions
	resultsWritter *writer.Writer
}
