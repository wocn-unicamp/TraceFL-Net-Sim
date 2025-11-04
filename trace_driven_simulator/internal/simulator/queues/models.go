package queues

import (
	"github.com/Marco-Guerra/Federated-Learning-Network-Workload/trace_driven_simulator/packages/writer"
)

type EventType uint8

const (
	ARRIVAL   EventType = 0
	DEPARTURE EventType = 1
)

type NetType uint8

const (
	SERVER NetType = 0
	CLIENT NetType = 1
)

type PacketType uint8

const (
	FIRST    PacketType = 0
	FRAGMENT PacketType = 1
	LAST     PacketType = 2
)

type Packet struct {
	MSSSize          uint32
	Size             uint32
	MSSArrivalTime   float64
	ArrivalTime      float64
	StartServiceTime float64
	DepartureTime    float64
	Id               uint64
	Type             PacketType
}

type Event struct {
	Time             float64
	ComputationTime  float64
	ClientQueueDelay float64
	*Packet
	ClientID    uint16
	RoundNumber uint16
	Type        EventType
	_           [3]byte
}

type Output struct {
	NumPackets uint32
	Bandwidth  uint32
	SimTime    float64
	Delay      float64
	Workload   *EventHeap
}

type GlobalOptions struct {
	NetType
	PacketHeader       uint8
	MinPacketSize      uint8
	MaxPacketSize      uint16
	MaxQueue           uint16
	Bandwidth          uint32
	BackgroundWorkload uint32
	ChannelLength      float32
	PropagationSpeed   float32
	EvalTime           float64
}

type EventQueue struct {
	options        *GlobalOptions
	queue          []*Packet
	events         *EventHeap
	resultsWritter *writer.Writer
	currentTime    float64
}

// EventHeap implements heap.Interface and holds Events
type EventHeap []*Event

func (h EventHeap) Len() int { return len(h) }

func (h EventHeap) Less(i, j int) bool {
	if h[i].Time != h[j].Time {
		return h[i].Time < h[j].Time
	}
	if h[i].RoundNumber != h[j].RoundNumber {
		return h[i].RoundNumber < h[j].RoundNumber
	}
	if h[i].ClientID != h[j].ClientID {
		return h[i].ClientID < h[j].ClientID
	}

	if h[i].Packet.Type != h[j].Packet.Type {
		return h[i].Packet.Type < h[j].Packet.Type
	}

	return h[i].Packet.Id < h[j].Packet.Id
}

func (h EventHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func (h *EventHeap) Push(x interface{}) {
	*h = append(*h, x.(*Event))
}

func (h *EventHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[0 : n-1]
	return item
}
