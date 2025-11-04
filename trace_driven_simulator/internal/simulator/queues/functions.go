package queues

import (
	"container/heap"
	"fmt"
	"log"
	"math"
	"os"

	"github.com/Marco-Guerra/Federated-Learning-Network-Workload/trace_driven_simulator/packages/writer"
)

func New(options *GlobalOptions, workload *EventHeap, rwritter *writer.Writer) *EventQueue {
	return &EventQueue{
		options:        options,
		queue:          make([]*Packet, 0),
		events:         workload,
		resultsWritter: rwritter,
	}
}

func (evq *EventQueue) Start() *Output {
	numPackets, simTime, totalDelay, outWorkload := evq.processEvents()

	return &Output{
		SimTime:    simTime,
		Delay:      totalDelay,
		NumPackets: uint32(numPackets),
		Bandwidth:  evq.options.Bandwidth,
		Workload:   outWorkload,
	}
}

func (evq *EventQueue) cleanBuffer() {
	qlen := len(evq.queue)

	if qlen >= int(evq.options.MaxQueue) {
		index := -1
		low := 0
		high := qlen - 1

		for low <= high {
			mid := (low + high) / 2

			if evq.queue[mid].DepartureTime == evq.currentTime {
				index = mid
				break
			} else if evq.queue[mid].DepartureTime < evq.currentTime {
				low = mid + 1
			} else {
				high = mid - 1
			}
		}

		if index > 1 {
			evq.queue = evq.queue[index:]
		}
	}
}

func (evq *EventQueue) processEvents() (int, float64, float64, *EventHeap) {
	numPackets := evq.events.Len()
	var totalBytes uint64 = 0
	var totalDelay float64 = 0
	var outWorkload *EventHeap = nil
	simStartTime := (*evq.events)[0].Time

	if evq.options.NetType != SERVER {
		outWorkload = &EventHeap{}
	}

	for evq.events.Len() > 0 {
		event := heap.Pop(evq.events).(*Event)
		evq.currentTime = event.Time

		switch event.Type {
		case ARRIVAL:
			qlen := len(evq.queue)

			if qlen == 0 || evq.queue[qlen-1].DepartureTime < event.Packet.ArrivalTime {
				if event.Packet == nil {
					fmt.Println("Memory error: a nil packets was find on the queue")
					fmt.Println(evq.options.NetType, event)
					os.Exit(2)
				}
				event.Packet.StartServiceTime = event.Packet.ArrivalTime
			} else {
				event.Packet.StartServiceTime = evq.queue[qlen-1].DepartureTime
			}

			event.Packet.DepartureTime = event.Packet.StartServiceTime + (float64(event.Packet.Size)*8)/float64(evq.options.Bandwidth)

			heap.Push(evq.events, &Event{
				Time:             event.Packet.DepartureTime,
				RoundNumber:      event.RoundNumber,
				ComputationTime:  event.ComputationTime,
				ClientQueueDelay: event.ClientQueueDelay,
				ClientID:         event.ClientID,
				Packet:           event.Packet,
				Type:             DEPARTURE,
			})

			evq.queue = append(evq.queue, event.Packet)
		case DEPARTURE:
			totalBytes += uint64(event.Packet.Size)

			if event.Packet.Type == LAST {
				individualDelay := event.Packet.DepartureTime - event.Packet.MSSArrivalTime

				if event.ClientID != 4096 {
					switch evq.options.NetType {
					case CLIENT:
						event.ClientQueueDelay = individualDelay
					case SERVER:
						evq.resultsWritter.Write(&writer.WriterRegister{
							ClientID:           event.ClientID,
							ComputationTime:    event.ComputationTime,
							Workload:           uint32(math.Floor((float64(event.MSSSize) * 8 / event.ComputationTime))),
							PropagationDelay:   float64(evq.options.ChannelLength / evq.options.PropagationSpeed),
							BackgroundWorkload: evq.options.BackgroundWorkload,
							ClientQueueDelay:   event.ClientQueueDelay,
							StationQueueDelay:  individualDelay,
							RoundNumber:        event.RoundNumber,
						})
					}
				}

				totalDelay += float64(individualDelay)
			}

			event.Packet.ArrivalTime = event.Packet.DepartureTime + float64(evq.options.ChannelLength/evq.options.PropagationSpeed)
			event.Packet.MSSArrivalTime = event.Packet.ArrivalTime

			if evq.options.NetType != SERVER {
				heap.Push(outWorkload, &Event{
					ComputationTime:  event.ComputationTime,
					ClientQueueDelay: event.ClientQueueDelay,
					Time:             event.Packet.ArrivalTime,
					RoundNumber:      event.RoundNumber,
					ClientID:         event.ClientID,
					Packet:           event.Packet,
					Type:             ARRIVAL,
				})
			}

			evq.cleanBuffer()
		default:
			log.Fatal("Unkown Event on the Event list. ", event)
		}
	}

	if outWorkload != nil && (*outWorkload)[0].ClientID != 4096 {
		firstEvent := (*outWorkload)[0]
		for i := 1; i < outWorkload.Len(); i++ {
			event := (*outWorkload)[i]
			event.Packet.MSSArrivalTime = firstEvent.Packet.MSSArrivalTime
		}
	}

	return numPackets, evq.currentTime - simStartTime, totalDelay, outWorkload
}
