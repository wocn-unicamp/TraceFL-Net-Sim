package queues

import (
	"container/heap"
	"fmt"
	"log"
	"math"
	"os"

	"github.com/wocn-unicamp/TraceFL-Net-Sim/trace_driven_simulator/packages/writer"
)

func New(options *GlobalOptions, workload *EventHeap, rwritter *writer.Writer) *EventQueue {
	return &EventQueue{
		options:           options,
		events:            workload,
		resultsWritter:    rwritter,
		LastDepartureTime: -1.0,
		CurrentBufferSize: 0,
	}
}

func (evq *EventQueue) Start() *Output {
	numMessages, numPackets, simTime, totalDelay, outWorkload := evq.processEvents()

	return &Output{
		SimTime:     simTime,
		Delay:       totalDelay,
		NumPackets:  uint32(numPackets),
		NumMessages: uint32(numMessages),
		Bandwidth:   evq.options.Bandwidth,
		Workload:    outWorkload,
	}
}

func (evq *EventQueue) processEvents() (int, int, float64, float64, *EventHeap) {
	if evq.events == nil || evq.events.Len() == 0 {
		return 0, 0, 0, 0, nil
	}

	numPackets := evq.events.Len()
	var numMessages int = 0
	var totalBytes uint64 = 0
	var totalDelay float64 = 0
	var outWorkload *EventHeap = nil

	if evq.options.NetType != SERVER {
		tmp := make(EventHeap, 0, numPackets)
		outWorkload = &tmp
	}

	bandwidthFactor := 8.0 / float64(evq.options.Bandwidth)
	propagationDelay := float64(evq.options.ChannelLength / evq.options.PropagationSpeed)

	for evq.events.Len() > 0 {
		event := heap.Pop(evq.events).(*Event)
		evq.currentTime = event.Time

		switch event.Type {
		case ARRIVAL:
			if event.Packet == nil {
				fmt.Println("Memory error: a nil packet was found on the queue")
				fmt.Println(evq.options.NetType, event)
				os.Exit(2)
			}

			if !evq.options.InfiniteBuffer && evq.CurrentBufferSize >= int(evq.options.MaxQueue) {
				if evq.options.EnableRetransmission {
					// Retrieve Explicit Backoff, or fallback to dynamic network RTO
					baseBackoff := evq.options.RetransmissionBackoff
					if baseBackoff <= 0.0 {
						// RTO Fallback = Time to transmit packet + Round Trip Time
						transmitTime := float64(event.Packet.Size) * bandwidthFactor
						rtt := 2.0 * propagationDelay
						baseBackoff = transmitTime + rtt
					}

					// Apply Exponential Backoff: Base * (2 ^ Attempts)
					multiplier := math.Pow(2, float64(event.Packet.RetransmissionAttempts))
					backoff := baseBackoff * multiplier

					// Increment attempt counter for next time
					event.Packet.RetransmissionAttempts++

					// Schedule the re-transmission in the future relative to NOW.
					retryTime := evq.currentTime + backoff
					event.Time = retryTime
					event.Packet.ArrivalTime = retryTime

					heap.Push(evq.events, event)
				}
				// If EnableRetransmission is false, the packet is permanently dropped.
				continue
			}

			// Buffer accepted the packet
			evq.CurrentBufferSize++

			if evq.LastDepartureTime < event.Packet.ArrivalTime {
				event.Packet.StartServiceTime = event.Packet.ArrivalTime
			} else {
				event.Packet.StartServiceTime = evq.LastDepartureTime
			}

			event.Packet.DepartureTime = event.Packet.StartServiceTime + (float64(event.Packet.Size) * bandwidthFactor)
			evq.LastDepartureTime = event.Packet.DepartureTime

			event.Time = event.Packet.DepartureTime
			event.Type = DEPARTURE

			heap.Push(evq.events, event)

		case DEPARTURE:
			evq.CurrentBufferSize--

			// --- TRANSMISSION FAILURE LOGIC ---
			// Check if we have an RNG configured and success rate is less than 100%
			if evq.options.RNG != nil && evq.options.TransmissionSuccessRate < 1.0 {
				if evq.options.RNG.Float64() > evq.options.TransmissionSuccessRate {
					if evq.options.EnableRetransmission {

						// Generate X ~ U(0.016, 0.064) using the existing RNG instance
						minBackoff := 0.016
						maxBackoff := 0.064
						x := minBackoff + (maxBackoff-minBackoff)*evq.options.RNG.Float64()

						// Calculate backoff using the formula: X * 2^n
						n := float64(event.Packet.RetransmissionAttempts)
						backoff := x * math.Pow(2, n)

						// Increment attempt counter for next time
						event.Packet.RetransmissionAttempts++

						// Re-schedule the event as an ARRIVAL after the backoff timer expires
						retryTime := evq.currentTime + backoff
						event.Time = retryTime
						event.Type = ARRIVAL
						event.Packet.ArrivalTime = retryTime

						heap.Push(evq.events, event)
					}
					// If EnableRetransmission is false, the packet is dropped permanently.
					continue // Skip the rest of the DEPARTURE logic to prevent metrics logging & forwarding
				}
			}
			// ----------------------------------

			// Successful transmission: reset attempts so the penalty doesn't carry over to the next network hop
			event.Packet.RetransmissionAttempts = 0

			totalBytes += uint64(event.Packet.Size)

			if event.Packet.Type == LAST {
				individualDelay := event.Packet.DepartureTime - event.Packet.MSSArrivalTime

				if event.ClientID != 4096 {
					numMessages++
					switch evq.options.NetType {
					case CLIENT:
						event.ClientQueueDelay = individualDelay
					case SERVER:
						evq.resultsWritter.Write(&writer.WriterRegister{
							ClientID:           event.ClientID,
							ComputationTime:    event.ComputationTime,
							Workload:           uint32(math.Floor((float64(event.MSSSize) * 8))),
							PropagationDelay:   propagationDelay,
							BackgroundWorkload: evq.options.BackgroundWorkload,
							ClientQueueDelay:   event.ClientQueueDelay,
							StationQueueDelay:  individualDelay,
							RoundNumber:        event.RoundNumber,
						})
					}
				}

				switch evq.options.NetType {
				case CLIENT:
					totalDelay += individualDelay
				case SERVER:
					if event.ClientID != 4096 {
						totalDelay += individualDelay
					}
				}
			}

			event.Packet.ArrivalTime = event.Packet.DepartureTime + propagationDelay
			event.Packet.MSSArrivalTime = event.Packet.ArrivalTime

			if evq.options.NetType != SERVER {
				event.Time = event.Packet.ArrivalTime
				event.Type = ARRIVAL
				heap.Push(outWorkload, event)
			}

		default:
			log.Fatal("Unknown Event on the Event list. ", event)
		}
	}

	return numMessages, numPackets, evq.currentTime, totalDelay, outWorkload
}
