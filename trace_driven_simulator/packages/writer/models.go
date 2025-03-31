package writer

import "encoding/csv"

type WriterRegister struct {
	ClientID           uint16
	RoundNumber        uint16
	BackgroundWorkload uint32
	Workload           uint32
	ComputationTime    float64
	ClientQueueDelay   float64
	PropagationDelay   float64
	StationQueueDelay  float64
}

type Writer struct {
	maxBufferSize      uint32
	aggregationChannel chan *WriterRegister
	aggregationBuffer  []*WriterRegister
	filename           string
	csvWriter          *csv.Writer
}
