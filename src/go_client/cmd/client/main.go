/*
A sample TensorFlow client in Go code.

Prerequisites:
```
~/mycode/ml/tensorflow/misc $ git clone https://github.com/lenjoy/tfserving_golang.git

export GOPATH=~/mycode/ml/tensorflow/misc/tfserving_golang:~/go
```

Usage:
```
~/mycode/ml/tensorflow/misc/tfserving_sample/go_client [master=]$ go build .

~/mycode/ml/tensorflow/misc/tfserving_sample/go_client [master=]$ ./go_client  --model_version 1532991213
====== yeah!!! ======
classes dtype:DT_STRING tensor_shape:<dim:<size:1 > dim:<size:2 > > string_val:"0" string_val:"1"
scores dtype:DT_FLOAT tensor_shape:<dim:<size:1 > dim:<size:2 > > float_val:0.40602925 float_val:0.5939707
[{0 0.40602925} {1 0.5939707}]
```
*/
package main

import (
	"flag"
	"fmt"

	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/grpclog"

	"github.com/golang/protobuf/proto"
	google_protobuf "github.com/golang/protobuf/ptypes/wrappers"

	tf_core_example "go_client/tensorflow/core/example"
	tf_core_framework "go_client/tensorflow/core/framework"
	pb "go_client/tensorflow_serving/apis"
)

var (
	serverAddr         = flag.String("server_addr", "127.0.0.1:8500", "The server address in the format of host:port")
	modelName          = flag.String("model_name", "wide_and_deep", "TensorFlow model name")
	modelVersion       = flag.Int64("model_version", 1, "TensorFlow model version")
	tls                = flag.Bool("tls", false, "Connection uses TLS if true, else plain TCP")
	caFile             = flag.String("ca_file", "testdata/ca.pem", "The file containning the CA root cert file")
	serverHostOverride = flag.String("server_host_override", "x.test.youtube.com", "The server name use to verify the hostname returned by TLS handshake")
)

type Prediction struct {
	Class string  `json:"class"`
	Score float32 `json:"score"`
}

func main() {
	flag.Parse()
	var opts []grpc.DialOption
	if *tls {
		var sn string
		if *serverHostOverride != "" {
			sn = *serverHostOverride
		}
		var creds credentials.TransportCredentials
		if *caFile != "" {
			var err error
			creds, err = credentials.NewClientTLSFromFile(*caFile, sn)
			if err != nil {
				grpclog.Fatalf("Failed to create TLS credentials %v", err)
			}
		} else {
			creds = credentials.NewClientTLSFromCert(nil, sn)
		}
		opts = append(opts, grpc.WithTransportCredentials(creds))
	} else {
		opts = append(opts, grpc.WithInsecure())
	}
	conn, err := grpc.Dial(*serverAddr, opts...)
	if err != nil {
		grpclog.Fatalf("fail to dial: %v", err)
	}
	defer conn.Close()
	client := pb.NewPredictionServiceClient(conn)

	req := newReq(modelName, modelVersion)
	resp, err := client.Predict(context.Background(), req)

	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println("====== yeah!!! ======")
	for k, v := range resp.Outputs {
		fmt.Println(k, v)
	}

	classesTensor, scoresTensor := resp.Outputs["classes"], resp.Outputs["scores"]
	if classesTensor == nil || scoresTensor == nil {
		fmt.Println("missing expected tensors in response")
	}

	classes := classesTensor.StringVal
	scores := scoresTensor.FloatVal
	var results []Prediction
	for i := 0; i < len(classes) && i < len(scores); i++ {
		results = append(results, Prediction{Class: string(classes[i]), Score: scores[i]})
	}
	fmt.Println(results)
}

func floatFeature(v ...float32) *tf_core_example.Feature {
	return &tf_core_example.Feature{
		Kind: &tf_core_example.Feature_FloatList{
			FloatList: &tf_core_example.FloatList{Value: v},
		},
	}
}

func bytesFeature(v ...[]byte) *tf_core_example.Feature {
	return &tf_core_example.Feature{
		Kind: &tf_core_example.Feature_BytesList{
			BytesList: &tf_core_example.BytesList{Value: v},
		},
	}
}

func newReq(modelName *string, modelVersion *int64) *pb.PredictRequest {
	featureMap := map[string]*tf_core_example.Feature{
		"age":            floatFeature(40),
		"education":      bytesFeature([]byte("Doctorate")),
		"education_num":  floatFeature(7),
		"capital_gain":   floatFeature(0),
		"capital_loss":   floatFeature(0),
		"hours_per_week": floatFeature(60),
		"workclass":      bytesFeature([]byte("Private")),
	}
	inputBytes, err := proto.Marshal(
		&tf_core_example.Example{
			Features: &tf_core_example.Features{
				Feature: featureMap,
			},
		})
	if err != nil {
		fmt.Println(".....fail.....")
		return nil
	}

	req := &pb.PredictRequest{
		ModelSpec: &pb.ModelSpec{
			Name: *modelName,
			Version: &google_protobuf.Int64Value{
				Value: *modelVersion,
			},
		},
		Inputs: map[string]*tf_core_framework.TensorProto{
			"inputs": &tf_core_framework.TensorProto{
				Dtype:     tf_core_framework.DataType_DT_STRING,
				StringVal: [][]byte{inputBytes},
				TensorShape: &tf_core_framework.TensorShapeProto{
					Dim: []*tf_core_framework.TensorShapeProto_Dim{{Size: 1}},
				},
			},
		},
	}
	return req
}
