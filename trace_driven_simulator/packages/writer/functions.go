package writer

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"strings"
)

func New(numberOfRegister uint32, filename string) *Writer {
	file, err := os.Create(filename)

	if err != nil {
		log.Fatalf("Unable to create file %s: %v", filename, err)
	}

	return &Writer{
		aggregationChannel: make(chan *WriterRegister),
		aggregationBuffer:  make([]*WriterRegister, 0),
		maxBufferSize:      uint32(math.Floor(float64(numberOfRegister) * float64(PER_OF_REGISTER_ON_MEMORY))),
		csvWriter:          csv.NewWriter(file),
		filename:           filename,
	}
}

func (w *Writer) Start() {
	err := w.csvWriter.Write(COLLUMN_LABELS)

	if err != nil {
		log.Fatalf("Unable to write collumn labels in file %s: %v", w.filename, err)
	}

	w.csvWriter.Flush()

	go w.aggregator()
}

func (w *Writer) Write(register *WriterRegister) {
	w.aggregationChannel <- register
}

func (w *Writer) Close() {
	close(w.aggregationChannel)
	w.write()
}

func (w *Writer) write() {
	stringBuffer := make([][]string, w.maxBufferSize)

	for rid := range w.aggregationBuffer {
		tempString := fmt.Sprintf("%d,%d,%d,%d,%.9f,%.9f,%.9f,%.9f",
			w.aggregationBuffer[rid].ClientID,
			w.aggregationBuffer[rid].RoundNumber,
			w.aggregationBuffer[rid].Workload,
			w.aggregationBuffer[rid].BackgroundWorkload,
			w.aggregationBuffer[rid].ComputationTime,
			w.aggregationBuffer[rid].ClientQueueDelay,
			w.aggregationBuffer[rid].PropagationDelay,
			w.aggregationBuffer[rid].StationQueueDelay,
		)

		stringBuffer[rid] = strings.Split(tempString, ",")
	}

	err := w.csvWriter.WriteAll(stringBuffer)

	if err != nil {
		log.Fatalf("Unable to write register on file %s: %v", w.filename, err)
		return
	}

	w.csvWriter.Flush()
}

func (w *Writer) aggregator() {
	for reg := range w.aggregationChannel {
		if len(w.aggregationBuffer) >= int(w.maxBufferSize) {
			w.write()
			w.aggregationBuffer = []*WriterRegister{}
		}

		w.aggregationBuffer = append(w.aggregationBuffer, reg)
	}
}
