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

	"github.com/Marco-Guerra/Federated-Learning-Network-Workload/trace_driven_simulator/internal/simulator/queues"
	"github.com/Marco-Guerra/Federated-Learning-Network-Workload/trace_driven_simulator/packages/writer"
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

func (td *TraceDriven) calculeMetrics(results *queues.Output) (float64, uint32) {
	meanDelay := results.Delay / float64(results.NumPackets)

	if results.SimTime <= 0 {
		results.SimTime = 1
	}

	throughput := float64(results.TotalBytes*8) / results.SimTime

	return meanDelay, uint32(math.Floor(math.Min(float64(throughput), float64(results.Bandwidth))))
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
	rng := rand.New(rand.NewSource(seed))

	var packetCounter uint64 = 0
	var currentTime float64 = 0.0
	var previousTime float64 = 0.0
	var backgroundTrafficMean float64 = float64(td.options.WorkloadBackgroundClients) / (((float64(ETHERNET_HEADER) + float64(ETHERNET_MIN_FRAME) + float64(ETHERNET_MTU)) / 2) * 8)
	var tmutex sync.Mutex = sync.Mutex{}

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
			PropagationSpeed: PROP_SPEED,
			ChannelLength:    CHANN_LEN,
		}
	}

	for round := 1; round <= rounds; round++ {
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
				packet := queues.Packet{
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

				event := queues.Event{
					Time:            packet.ArrivalTime,
					RoundNumber:     uint16(round),
					ClientID:        uint16(clientID),
					ComputationTime: time,
					Packet:          &packet,
					Type:            queues.ARRIVAL,
				}

				heap.Push(&workloads[clientID-1], &event)

				messageSize -= int(ETHERNET_MTU)

				packetCounter++
			}

			packet := queues.Packet{
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

			event := queues.Event{
				Time:            packet.ArrivalTime,
				RoundNumber:     uint16(round),
				ComputationTime: time,
				ClientID:        uint16(clientID),
				Packet:          &packet,
				Type:            queues.ARRIVAL,
			}

			heap.Push(&workloads[clientID-1], &event)

			packetCounter++
		}

		previousTime = currentTime

		// Update current time
		for _, client := range clients {
			clientTime, _ := strconv.ParseFloat(client[6], 64)
			if clientTime > currentTime {
				currentTime = clientTime
			}
		}

		currentTime += float64(SERVER_AGG_TIME + DOWNLINK_TIME)

		for i := nFLClients; i < nclient; i++ {
			var arrivalInterval float64 = 0
			for localtime := float64(previousTime); localtime <= float64(currentTime); localtime += arrivalInterval {
				mssSize := uint32(ETHERNET_MIN_FRAME) + rng.Uint32()%uint32(uint16(ETHERNET_MIN_FRAME-ETHERNET_HEADER)+ETHERNET_MTU+1)

				packet := queues.Packet{
					MSSSize:        mssSize,
					ArrivalTime:    localtime,
					MSSArrivalTime: localtime,
					Size:           mssSize,
					Type:           queues.LAST,
					Id:             packetCounter,
				}

				event := queues.Event{
					Time:        packet.ArrivalTime,
					RoundNumber: 1001,
					ClientID:    4096,
					Packet:      &packet,
					Type:        queues.ARRIVAL,
				}

				heap.Push(&workloads[i], &event)

				packetCounter++

				arrivalInterval = (-math.Log(1-rand.Float64()) / backgroundTrafficMean)
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

				meanDelay, throughput := td.calculeMetrics(qout)

				resultString := fmt.Sprintf("%d,%d,%f,%d\n",
					round,
					qid+1,
					meanDelay,
					throughput,
				)

				fmt.Print(resultString)

				tmutex.Lock()
				for qout.Workload.Len() > 0 {
					heap.Push(&serverWorkload, heap.Pop(qout.Workload))
				}
				tmutex.Unlock()

				qwg.Done()
			}(i)
		}

		qwg.Wait()

		serverQueue := queues.New(&queues.GlobalOptions{
			MaxQueue:           uint16(math.Floor((float64(serverWorkload.Len()) * 0.10))),
			NetType:            queues.SERVER,
			Bandwidth:          td.options.ServerBandwidth,
			BackgroundWorkload: td.options.WorkloadBackgroundClients,
			PacketHeader:       ETHERNET_HEADER,
			MinPacketSize:      ETHERNET_MIN_FRAME,
			MaxPacketSize:      ETHERNET_MTU,
			PropagationSpeed:   PROP_SPEED,
			ChannelLength:      CHANN_LEN,
		},
			&serverWorkload,
			td.resultsWritter,
		)

		sqout := serverQueue.Start()

		meanDelay, throughput := td.calculeMetrics(sqout)

		resultString := fmt.Sprintf("%d,0,%f,%d\n",
			round,
			meanDelay,
			throughput,
		)

		fmt.Print(resultString)
	}
}
