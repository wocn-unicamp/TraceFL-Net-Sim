package simulator

import (
	"container/heap"
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/wocn-unicamp/TraceFL-Net-Sim/trace_driven_simulator/internal/simulator/queues"
	"github.com/wocn-unicamp/TraceFL-Net-Sim/trace_driven_simulator/packages/writer"
	"golang.org/x/exp/rand"
)

func New(options *GlobalOptions) *TraceDriven {
	return &TraceDriven{
		options: options,
	}
}

func (td *TraceDriven) RunSimulation(trace_filename string) {
	td.readTrace(trace_filename)
	td.resultsWritter.Close()
}

func (td *TraceDriven) calculeMetrics(results *queues.Output, backgroundTraffic bool) float64 {
	var meanDelay float64 = 0.0

	if backgroundTraffic {
		if results.NumPackets > 0 {
			meanDelay = results.Delay / float64(results.NumPackets)
		}
	} else {
		if results.NumMessages > 0 {
			meanDelay = results.Delay / float64(results.NumMessages)
		}
	}

	return meanDelay
}

func (td *TraceDriven) readTrace(traceFilename string) {
	parts := strings.Split(traceFilename, "_")
	var leafExperimentMeta string

	if len(parts) > 2 {
		leafExperimentMeta = strings.Join(parts[4:], "_")
	} else {
		log.Fatal("Unexpected pattern in trace filename. ", traceFilename)
	}

	file, err := os.Open(traceFilename)
	if err != nil {
		log.Fatal("Error opening file:", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		log.Fatal("Error reading CSV file:", err)
	}

	seed := uint64(time.Now().Unix())
	if td.options.Seed != 0 {
		seed = td.options.Seed
	}
	rng := rand.New(rand.NewSource(seed))

	var packetCounter uint64 = 0
	var currentTime float64 = 0.0
	var previousTime float64 = 0.0
	var tmutex sync.Mutex = sync.Mutex{}

	// Calculate the mean arrival interval dynamically based on bandwidth load
	minFrameSize := float64(ETHERNET_MIN_FRAME)
	maxFrameSize := minFrameSize - float64(ETHERNET_HEADER) + float64(ETHERNET_MTU)
	averagePacketSizeBits := ((minFrameSize + maxFrameSize) / 2.0) * 8.0

	targetBandwidthBps := td.options.BackgroundTrafficLoad * float64(td.options.ServerBandwidth)
	var meanArrivalInterval float64
	if targetBandwidthBps > 0 {
		meanArrivalRate := targetBandwidthBps / averagePacketSizeBits
		meanArrivalInterval = 1.0 / meanArrivalRate
	} else {
		meanArrivalInterval = math.Inf(1)
	}

	// Find the maximum round number
	rounds := 0
	for i, record := range records {
		if i == 0 {
			continue // Skip header
		}
		roundNumber, _ := strconv.Atoi(record[1])
		if roundNumber > rounds {
			rounds = roundNumber
		}
	}

	// Find the number of clients
	nFLClients := 0
	lastNClients := 0
	for i, record := range records {
		if i == 0 {
			continue // Skip header
		}

		clientID, _ := strconv.Atoi(record[0])
		if clientID > nFLClients {
			lastNClients = nFLClients
			nFLClients = clientID
		}

		if lastNClients == nFLClients {
			break
		}
	}

	td.resultsWritter = writer.New(uint32(len(records)), "metrics_network_"+leafExperimentMeta)
	go td.resultsWritter.Start()

	nclient := nFLClients + 1
	queuesOPT := make([]*queues.GlobalOptions, nclient)

	for i := range nclient {
		queuesOPT[i] = &queues.GlobalOptions{
			Bandwidth:             td.options.ClientsBandwidth,
			NetType:               queues.CLIENT,
			EvalTime:              EVAL_TIME,
			PropagationSpeed:      PROP_SPEED,
			ChannelLength:         CHANN_LEN,
			InfiniteBuffer:        td.options.InfiniteBuffer,
			MaxQueue:              td.options.MaxQueueSize,
			EnableRetransmission:  td.options.EnableRetransmission,
			RetransmissionBackoff: td.options.RetransmissionBackoff,
		}

		if i == nFLClients {
			queuesOPT[i].Bandwidth = td.options.ServerBandwidth
		}
	}

	bgIsBursting := false
	bgMeanOn := ON_MEAN_BG
	bgMeanOff := OFF_MEAN_BG
	bgNextTransitionTime := make([]float64, nclient)
	for i := range nclient {
		bgNextTransitionTime[i] = -math.Log(1-rng.Float64()) * bgMeanOff
	}

	// State trackers BEFORE the rounds loop
	serverLastDepartureTime := -1.0
	serverCurrentBufferSize := 0

	clientLastDepartureTime := make([]float64, nclient)
	clientCurrentBufferSize := make([]int, nclient)
	for i := range clientLastDepartureTime {
		clientLastDepartureTime[i] = -1.0
	}

	for round := 1; round <= rounds; round++ {
		if round >= td.options.MaxNumberOfRounds && td.options.MaxNumberOfRounds != -1 {
			break
		}
		var clients [][]string
		dqueues := make([]*queues.EventQueue, nclient)
		workloads := make([]queues.EventHeap, nclient)
		serverWorkload := queues.EventHeap{}

		for i, record := range records {
			if i == 0 {
				continue // Skip header
			}
			roundNumber, _ := strconv.Atoi(record[1])
			if roundNumber == round {
				clients = append(clients, record)
			}
		}

		var messageSize int = 0
		var maxClientTime float64 = 0.0

		for _, row := range clients {
			messageSize, _ = strconv.Atoi(row[4])
			compTime, _ := strconv.ParseFloat(row[6], 64)
			clientID, _ := strconv.Atoi(row[0])

			if compTime > maxClientTime {
				maxClientTime = compTime
			}

			temp := messageSize

			for messageSize > int(ETHERNET_MTU) {
				packet := &queues.Packet{
					MSSSize:        uint32(temp),
					MSSArrivalTime: compTime + currentTime,
					ArrivalTime:    compTime + currentTime,
					Size:           uint32(ETHERNET_MTU) + uint32(ETHERNET_HEADER),
					Type:           queues.FRAGMENT,
					Id:             packetCounter,
				}

				if messageSize == temp {
					packet.Type = queues.FIRST
				}

				event := &queues.Event{
					Time:            packet.ArrivalTime,
					RoundNumber:     uint16(round),
					ClientID:        uint16(clientID),
					ComputationTime: compTime,
					Packet:          packet,
					Type:            queues.ARRIVAL,
				}

				heap.Push(&workloads[clientID-1], event)

				messageSize -= int(ETHERNET_MTU)
				packetCounter++
			}

			packet := &queues.Packet{
				MSSSize:        uint32(temp),
				MSSArrivalTime: compTime + currentTime,
				ArrivalTime:    compTime + currentTime,
				Type:           queues.LAST,
				Size:           uint32(messageSize),
				Id:             packetCounter,
			}

			if messageSize < int(ETHERNET_MIN_FRAME) {
				packet.Size = uint32(ETHERNET_MIN_FRAME)
			}

			event := &queues.Event{
				Time:            packet.ArrivalTime,
				RoundNumber:     uint16(round),
				ComputationTime: compTime,
				ClientID:        uint16(clientID),
				Packet:          packet,
				Type:            queues.ARRIVAL,
			}

			heap.Push(&workloads[clientID-1], event)
			packetCounter++
		}

		previousTime = currentTime
		bgEndTime := previousTime + maxClientTime

		if serverLastDepartureTime < currentTime {
			serverLastDepartureTime = currentTime
			serverCurrentBufferSize = 0
		}

		// Generate Background Traffic over the active client window [previousTime, bgEndTime]
		for i := nFLClients; i < nclient; i++ {
			if meanArrivalInterval == math.Inf(1) {
				continue
			}

			localtime := float64(previousTime)

			for {
				var arrivalInterval float64

				switch td.options.BackgroundTrafficModel {
				case PARETO:
					alpha := ALPHA_BG
					xm := meanArrivalInterval * ((alpha - 1.0) / alpha)
					u := rng.Float64()
					arrivalInterval = xm / math.Pow(1.0-u, 1.0/alpha)

				case ONOFF:
					for localtime >= bgNextTransitionTime[i] {
						bgIsBursting = !bgIsBursting
						if bgIsBursting {
							bgNextTransitionTime[i] += -math.Log(1-rng.Float64()) * bgMeanOn
						} else {
							bgNextTransitionTime[i] += -math.Log(1-rng.Float64()) * bgMeanOff
						}
					}

					if !bgIsBursting {
						localtime = bgNextTransitionTime[i]
						bgIsBursting = true
						bgNextTransitionTime[i] += -math.Log(1-rng.Float64()) * bgMeanOn
					}

					burstFactor := (bgMeanOn + bgMeanOff) / bgMeanOn
					burstArrivalInterval := meanArrivalInterval / burstFactor
					arrivalInterval = -math.Log(1-rng.Float64()) * burstArrivalInterval

				case POISSON:
					fallthrough
				default:
					arrivalInterval = -math.Log(1-rng.Float64()) * meanArrivalInterval
				}

				localtime += arrivalInterval
				if localtime > bgEndTime {
					break
				}

				mssSize := uint32(ETHERNET_MIN_FRAME) + rng.Uint32()%uint32(uint16(ETHERNET_MIN_FRAME-ETHERNET_HEADER)+ETHERNET_MTU+1)

				packet := &queues.Packet{
					MSSSize:        mssSize,
					ArrivalTime:    localtime,
					MSSArrivalTime: localtime,
					Size:           mssSize,
					Type:           queues.LAST,
					Id:             packetCounter,
				}

				event := &queues.Event{
					Time:        packet.ArrivalTime,
					RoundNumber: 1001,
					ClientID:    4096,
					Packet:      packet,
					Type:        queues.ARRIVAL,
				}

				heap.Push(&workloads[i], event)
				packetCounter++
			}
		}

		for i := range dqueues {
			queuesOPT[i].MaxQueue = uint16(math.Floor((float64(workloads[i].Len()) * 0.10)))
			dqueues[i] = queues.New(queuesOPT[i], &workloads[i], td.resultsWritter)

			dqueues[i].LastDepartureTime = clientLastDepartureTime[i]
			dqueues[i].CurrentBufferSize = clientCurrentBufferSize[i]
		}

		qwg := sync.WaitGroup{}
		qwg.Add(nclient)

		for i := range nclient {
			go func(qid int) {
				qout := dqueues[qid].Start()

				clientLastDepartureTime[qid] = dqueues[qid].LastDepartureTime
				clientCurrentBufferSize[qid] = dqueues[qid].CurrentBufferSize

				tmutex.Lock()
				if qout.SimTime > currentTime {
					currentTime = qout.SimTime
				}
				tmutex.Unlock()

				meanDelay := td.calculeMetrics(qout, qid == nFLClients)

				resultString := fmt.Sprintf("%d,%d,%f\n",
					round,
					qid+1,
					meanDelay,
				)
				fmt.Print(resultString)

				tmutex.Lock()
				if qout.Workload != nil {
					for qout.Workload.Len() > 0 {
						heap.Push(&serverWorkload, heap.Pop(qout.Workload))
					}
				}
				tmutex.Unlock()

				qwg.Done()
			}(i)
		}

		qwg.Wait()

		queueWorkloadMetric := uint32(math.Round(td.options.BackgroundTrafficLoad * 100))

		basePropDelay := float64(CHANN_LEN) / float64(PROP_SPEED)
		internetJitter := -math.Log(1-rng.Float64()) * INTERNET_JITTER_MEAN
		serverDelay := basePropDelay + internetJitter

		serverQueue := queues.New(&queues.GlobalOptions{
			InfiniteBuffer:        td.options.InfiniteBuffer,
			MaxQueue:              td.options.MaxQueueSize,
			EnableRetransmission:  td.options.EnableRetransmission,
			RetransmissionBackoff: td.options.RetransmissionBackoff,
			NetType:               queues.SERVER,
			Bandwidth:             td.options.ServerBandwidth,
			BackgroundWorkload:    queueWorkloadMetric,
			PacketHeader:          ETHERNET_HEADER,
			EvalTime:              EVAL_TIME,
			MinPacketSize:         ETHERNET_MIN_FRAME,
			MaxPacketSize:         ETHERNET_MTU,
			PropagationSpeed:      1.0,
			ChannelLength:         float32(serverDelay),
		},
			&serverWorkload,
			td.resultsWritter,
		)

		serverQueue.LastDepartureTime = serverLastDepartureTime
		serverQueue.CurrentBufferSize = serverCurrentBufferSize

		sqout := serverQueue.Start()

		serverLastDepartureTime = serverQueue.LastDepartureTime
		serverCurrentBufferSize = serverQueue.CurrentBufferSize

		if sqout.SimTime > currentTime {
			currentTime = sqout.SimTime + float64(SERVER_AGG_TIME+DOWNLINK_TIME)
		}

		meanDelay := td.calculeMetrics(sqout, false)

		resultString := fmt.Sprintf("%d,0,%f\n",
			round,
			meanDelay,
		)
		fmt.Print(resultString)
	}
}
