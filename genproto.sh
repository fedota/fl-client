#!/bin/bash -e

protodir=../fl-proto
gen=.

# if [ ! -d $gen ]; then
#     mkdir $gen;
#     touch "$gen/__init__.py"
# fi

python3 -m grpc_tools.protoc -I=$protodir --python_out=$gen --grpc_python_out=$gen $protodir/fl_round/fl_round.proto

#python -m grpc_tools.protoc --python_out=genproto --grpc_python_out=genproto  -I ../fl-proto ../fl-proto/fl_round/fl_round.proto