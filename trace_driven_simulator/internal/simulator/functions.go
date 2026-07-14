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

func (td *TraceDriven) calculeMetrics(results *queues.Output) float64 {
	// Prevent division by zero (NaN) if a client had no packets in this round
	if results.NumPackets == 0 {
		return 0.0
	}
	meanDelay := results.Delay / float64(results.NumPackets)
	return meanDelay
}

func (td *TraceDriven) readTrace(traceFilename string) {
	parts := strings.Split(traceFilename, "_")
	var leafExperimentMeta string

	if len(parts) > 2 {
		leafExperimentMeta = strings.Join(parts[4:], "_")
	} else {
		log.Fatal("Unexpected patten in trace filename. ", traceFilename)
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
			Bandwidth:        td.options.ClientsBandwidth,
			NetType:          queues.CLIENT,
			EvalTime:         EVAL_TIME,
			PropagationSpeed: PROP_SPEED,
			ChannelLength:    CHANN_LEN,
		}
	}

	// Setup independent Pareto On/Off state machines for every background client
	bgIsBursting := make([]bool, nclient)
	bgNextTransitionTime := make([]float64, nclient)

	// Pre-calculate Pareto Scale Parameters (Xm) to match the target Mean times
	alpha := ALPHA_BG
	xmOn := ON_MEAN_BG * ((alpha - 1.0) / alpha)
	xmOff := OFF_MEAN_BG * ((alpha - 1.0) / alpha)

	// Initialize all clients to start in a Pareto OFF state
	for i := range nclient {
		u := rng.Float64()
		bgNextTransitionTime[i] = xmOff / math.Pow(1.0-u, 1.0/alpha)
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

		for _, row := range clients {
			messageSize, _ = strconv.Atoi(row[4])
			time, _ := strconv.ParseFloat(row[6], 64)
			clientID, _ := strconv.Atoi(row[0])

			temp := messageSize

			for messageSize > int(ETHERNET_MTU) {
				packet := &queues.Packet{
					MSSSize:        uint32(temp),
					MSSArrivalTime: time + currentTime,
					ArrivalTime:    time + currentTime,
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
					ComputationTime: time,
					Packet:          packet,
					Type:            queues.ARRIVAL,
				}

				heap.Push(&workloads[clientID-1], event)

				messageSize -= int(ETHERNET_MTU)
				packetCounter++
			}

			packet := &queues.Packet{
				MSSSize:        uint32(temp),
				MSSArrivalTime: time + currentTime,
				ArrivalTime:    time + currentTime,
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
				ComputationTime: time,
				ClientID:        uint16(clientID),
				Packet:          packet,
				Type:            queues.ARRIVAL,
			}

			heap.Push(&workloads[clientID-1], event)
			packetCounter++
		}

		previousTime = currentTime

		for _, client := range clients {
			clientTime, _ := strconv.ParseFloat(client[6], 64)
			if clientTime > currentTime {
				currentTime = clientTime
			}
		}

		for i := nFLClients; i < nclient; i++ {
			if meanArrivalInterval == math.Inf(1) {
				continue
			}

			localtime := float64(previousTime)

			for {
				var arrivalInterval float64

				switch td.options.BackgroundTrafficModel {
				case PARETO:
					// Standard independent heavy-tail arrivals
					u := rng.Float64()
					xm := meanArrivalInterval * ((alpha - 1.0) / alpha)
					arrivalInterval = xm / math.Pow(1.0-u, 1.0/alpha)

				case ONOFF:
					// Classic Pareto On/Off process for self-similar traffic
					for localtime >= bgNextTransitionTime[i] {
						bgIsBursting[i] = !bgIsBursting[i]
						u := rng.Float64()
						if bgIsBursting[i] {
							bgNextTransitionTime[i] += xmOn / math.Pow(1.0-u, 1.0/alpha)
						} else {
							bgNextTransitionTime[i] += xmOff / math.Pow(1.0-u, 1.0/alpha)
						}
					}

					// Fast-forward local time over the idle OFF gap
					if !bgIsBursting[i] {
						localtime = bgNextTransitionTime[i]
						bgIsBursting[i] = true
						u := rng.Float64()
						bgNextTransitionTime[i] += xmOn / math.Pow(1.0-u, 1.0/alpha)
					}

					burstFactor := (ON_MEAN_BG + OFF_MEAN_BG) / ON_MEAN_BG
					burstArrivalInterval := meanArrivalInterval / burstFactor
					// During the ON burst, packets arrive at Poisson intervals matching the burst rate
					arrivalInterval = -math.Log(1-rng.Float64()) * burstArrivalInterval

				case POISSON:
					fallthrough
				default:
					arrivalInterval = -math.Log(1-rng.Float64()) * meanArrivalInterval
				}

				localtime += arrivalInterval
				if localtime > float64(currentTime) {
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
		}

		qwg := sync.WaitGroup{}
		qwg.Add(nclient)

		for i := range nclient {
			go func(qid int) {
				qout := dqueues[qid].Start()

				tmutex.Lock()
				if qout.SimTime > currentTime {
					previousTime = currentTime
					currentTime = qout.SimTime
				}
				tmutex.Unlock()

				meanDelay := td.calculeMetrics(qout)

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
			MaxQueue:           uint16(math.Floor((float64(serverWorkload.Len()) * 0.10))),
			NetType:            queues.SERVER,
			Bandwidth:          td.options.ServerBandwidth,
			BackgroundWorkload: queueWorkloadMetric,
			PacketHeader:       ETHERNET_HEADER,
			EvalTime:           EVAL_TIME,
			MinPacketSize:      ETHERNET_MIN_FRAME,
			MaxPacketSize:      ETHERNET_MTU,
			PropagationSpeed:   1.0,
			ChannelLength:      float32(serverDelay),
		},
			&serverWorkload,
			td.resultsWritter,
		)

		sqout := serverQueue.Start()
		meanDelay := td.calculeMetrics(sqout)

		resultString := fmt.Sprintf("%d,0,%f\n",
			round,
			meanDelay,
		)
		fmt.Print(resultString)
	}
}
